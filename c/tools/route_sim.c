/* route_sim: offline eviction-quality scorer for the per-layer expert
 * streaming cache. Replays a ROUTE_TRACE dump (glm.c: one line per
 * position+layer, "call pos layer eid:w eid:w ...") through three policies
 * over the exact same request stream — LRU, the adaptive SLRU from tier.h
 * (cloxcache), and Belady's OPT with bypass (the offline lower bound on
 * misses) — reproducing the runtime's batch-union access pattern: per moe
 * call, unique experts in first-appearance order, probed in 64-blocks, the
 * last min(nmiss,cap) misses of each block admitted in reverse (the promo
 * loop), and routing recency stamped for every selection before the probes.
 *
 * Hit rate alone is a proxy: two policies can hit equally while sacrificing
 * very different experts. So besides hit%, each ONLINE policy's individual
 * eviction decisions are scored against hindsight:
 *   x-opt%   excess misses over OPT — the headroom the policy leaves
 *   sub%     evictions where the victim was re-requested BEFORE some other
 *            then-resident expert (a choice Belady would not have made)
 *   ret%     victims later re-requested at all (eviction regret rate)
 *   ttr      mean probes until an evicted victim returns (higher = better)
 *
 * Usage: route_sim -c CAP [-p NPIN] trace
 *   -c  streaming-cache slots per layer (mirror the runtime's ecap)
 *   -p  pin the N most-requested experts per layer (approximates VRAM pins:
 *       kept in the 64-block layout, never enter the streaming cache)
 * The clox column replays tier.h verbatim, including its built-in live-freq
 * decay; to sweep the decay cadence, rebuild with -DTIER_DECAY_EVERY=n.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../tier.h"

#define INF 0xffffffffu
#define PIN 0x80000000u
#define MAXLAYERS 4096

typedef struct {                 /* raw per-layer trace */
    int *routed; size_t nr, car; /* selected experts, dups included, line order */
    size_t *cs; size_t nc, cac;  /* per-call start offsets into routed */
} LT;

typedef struct {                 /* per-policy result */
    unsigned long long hits, miss, evict, subopt, returned, ttr;
} PS;

static void *xrealloc(void *p, size_t n){ p=realloc(p,n); if(!p){fprintf(stderr,"oom\n");exit(1);} return p; }

static void lt_call(LT *t){
    if(t->nc==t->cac){ t->cac=t->cac?t->cac*2:64; t->cs=xrealloc(t->cs,t->cac*sizeof *t->cs); }
    t->cs[t->nc++]=t->nr;
}
static void lt_push(LT *t, int e){
    if(t->nr==t->car){ t->car=t->car?t->car*2:256; t->routed=xrealloc(t->routed,t->car*sizeof *t->routed); }
    t->routed[t->nr++]=e;
}

/* next use of expert e strictly after probe index t (lazy monotone cursor) */
static uint32_t nu_at(uint32_t e, uint32_t t, const uint32_t *occ, const uint32_t *off, uint32_t *cur){
    uint32_t c=cur[e], end=off[e+1];
    while(c<end && occ[c]<=t) c++;
    cur[e]=c;
    return c<end ? occ[c] : INF;
}

static int lru_victim(const uint64_t *used, int nn){
    int v=0; for(int z=1;z<nn;z++) if(used[z]<used[v]) v=z; return v;
}

/* verbatim victim policy from glm.c ecache_victim, on the sim's slot arrays */
static int sim_victim(const int *seid, const uint64_t *used, int nn, const int8_t *freq, int k){
    int v=-1, fb=0; uint64_t vu=UINT64_MAX, fu=UINT64_MAX;
    for(int z=0;z<nn;z++){
        if(seid[z]<0) return z;
        if(freq[seid[z]]<=k && used[z]<vu){ v=z; vu=used[z]; }
        if(used[z]<fu){ fb=z; fu=used[z]; }
    }
    return v>=0 ? v : fb;
}

/* score one eviction decision against hindsight, then account it */
static void score_evict(PS *s, int vict, uint32_t t, const int *seid, int nn,
                        const uint32_t *occ, const uint32_t *off, uint32_t *cur){
    uint32_t nv=nu_at((uint32_t)vict,t,occ,off,cur);
    if(nv!=INF){
        s->returned++; s->ttr+=nv-t;
        for(int z=0;z<nn;z++){
            if(seid[z]<0 || seid[z]==vict) continue;
            if(nu_at((uint32_t)seid[z],t,occ,off,cur)>nv){ s->subopt++; break; }
        }
    }
    s->evict++;
}

/* one online policy over a layer's probe stream. clox!=0 runs the tier.h
 * adaptive SLRU (with routing-recency updates feeding tier_evict's ghost
 * turnover); clox==0 runs plain LRU on the same block/promo structure. */
static void run_online(const LT *t, const uint32_t *pr, const size_t *pcs, int nexp, int cap,
                       const uint32_t *occ, const uint32_t *off, uint32_t *cur,
                       int clox, PS *s, TierAdapt *ad_out, unsigned long long *prot_evict){
    int *seid=xrealloc(NULL,cap*sizeof *seid);
    uint64_t *used=xrealloc(NULL,cap*sizeof *used);
    int *rs=xrealloc(NULL,nexp*sizeof *rs);            /* expert -> slot | -1 */
    int8_t *freq=calloc(nexp,1);
    unsigned char *gb=calloc(nexp,1);   /* graduated-this-residency (clox==2) */
    uint32_t *last=calloc(nexp,sizeof *last);
    for(int e=0;e<nexp;e++) rs[e]=-1;
    TierAdapt ad; tier_adapt_init(&ad);
    memset(s,0,sizeof *s);
    int nn=0; uint64_t eclock=0; uint32_t aclock=0, tix=0;

    for(size_t c=0;c<t->nc;c++){
        if(clox)                                            /* glm.c:2967 — recency for every selection, pinned included */
            for(size_t i=t->cs[c];i<(c+1<t->nc?t->cs[c+1]:t->nr);i++) last[t->routed[i]]=++aclock;
        size_t ps=pcs[c], pn=pcs[c+1];
        for(size_t base=ps;base<pn;base+=64){               /* batch-union 64-blocks */
            size_t nb = pn-base<64 ? pn-base : 64;
            int missb[64]; int nmiss=0;
            uint32_t bt=0;
            for(size_t j=0;j<nb;j++){
                uint32_t v=pr[base+j]; bt=tix++;
                if(v&PIN) continue;                          /* pinned: resident elsewhere, not this cache */
                int e=(int)v;
                if(rs[e]>=0){
                    s->hits++; used[rs[e]]=++eclock;
                    if(clox==1){ tier_touch(&ad,&freq[e],nn>=cap); tier_probe(&ad,1); }
                    else if(clox==2){          /* -g: graduate once per residency */
                        int8_t fv=freq[e];
                        if(fv>=1&&fv<TIER_FMAX){
                            if(fv==ad.k && nn>=cap && !gb[e]){ ad.graduated++; gb[e]=1; }
                            freq[e]=(int8_t)(fv+1);
                        }
                        tier_probe(&ad,1);
                    }
                }else{
                    s->miss++; missb[nmiss++]=e;
                    if(clox) tier_probe(&ad,0);
                }
            }
            int promo = nmiss<cap ? nmiss : cap;             /* glm.c promo loop: last promo misses, reverse */
            for(int a=0;a<promo;a++){
                int e=missb[nmiss-1-a], slot;
                if(nn<cap) slot=nn++;
                else{
                    slot = clox ? sim_victim(seid,used,nn,freq,ad.k)
                                : lru_victim(used,nn);
                    int vict=seid[slot];
                    score_evict(s,vict,bt,seid,nn,occ,off,cur);
                    if(clox){
                        if(prot_evict && freq[vict]>ad.k) (*prot_evict)++;
                        tier_evict(&ad,freq,nexp,last,vict,cap);
                        tier_maybe_adapt(&ad);
                    }
                    rs[vict]=-1;
                }
                seid[slot]=e; rs[e]=slot; used[slot]=++eclock;
                if(clox){ tier_admit(&freq[e]); gb[e]=0; }
            }
        }
    }
    if(ad_out) *ad_out=ad;
    free(seid); free(used); free(rs); free(freq); free(gb); free(last);
}

/* Belady with bypass, probe by probe: the miss-count lower bound */
static void run_opt(const uint32_t *pr, size_t np, int nexp, int cap,
                    const uint32_t *occ, const uint32_t *off, uint32_t *cur, PS *s){
    int *seid=xrealloc(NULL,cap*sizeof *seid);
    uint32_t *nu=xrealloc(NULL,cap*sizeof *nu);
    int *rs=xrealloc(NULL,nexp*sizeof *rs);
    for(int e=0;e<nexp;e++) rs[e]=-1;
    memset(s,0,sizeof *s);
    int nn=0;
    for(size_t i=0;i<np;i++){
        uint32_t v=pr[i];
        if(v&PIN) continue;
        int e=(int)v;
        uint32_t nxt=nu_at((uint32_t)e,(uint32_t)i,occ,off,cur);
        if(rs[e]>=0){ s->hits++; nu[rs[e]]=nxt; continue; }
        s->miss++;
        if(nxt==INF) continue;                               /* never again: bypass for free */
        if(nn<cap){ seid[nn]=e; rs[e]=nn; nu[nn]=nxt; nn++; continue; }
        int zm=0; for(int z=1;z<nn;z++) if(nu[z]>nu[zm]) zm=z;
        if(nxt>=nu[zm]) continue;                            /* incoming is the farthest: bypass */
        rs[seid[zm]]=-1; s->evict++;
        seid[zm]=e; rs[e]=zm; nu[zm]=nxt;
    }
    free(seid); free(nu); free(rs);
}

static double pct(unsigned long long a, unsigned long long b){ return b? 100.0*(double)a/(double)b : 0.0; }
static double xopt(const PS *p, const PS *o){ return o->miss? 100.0*((double)p->miss-(double)o->miss)/(double)o->miss : 0.0; }

int main(int argc, char **argv){
    int cap=0, npin=0, gonce=0; const char *path=NULL;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"-c")&&i+1<argc) cap=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-p")&&i+1<argc) npin=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-g")) gonce=1;    /* clox: graduate once per residency */
        else path=argv[i];
    }
    if(!path||cap<1){ fprintf(stderr,"usage: route_sim -c CAP [-p NPIN] trace\n"); return 2; }
    FILE *f=fopen(path,"r");
    if(!f){ perror(path); return 1; }

    static LT lt[MAXLAYERS];
    int nexp=0, maxlayer=-1;
    long lastcall=-1; int lastlayer=-1;
    char line[1<<16];
    while(fgets(line,sizeof line,f)){
        char *p=line,*q;
        long call=strtol(p,&q,10); if(q==p) continue; p=q;
        strtol(p,&q,10); p=q;                        /* pos: unused */
        long layer=strtol(p,&q,10); p=q;
        if(layer<0||layer>=MAXLAYERS) continue;
        if(layer>maxlayer) maxlayer=(int)layer;
        if(call!=lastcall||layer!=lastlayer){ lt_call(&lt[layer]); lastcall=call; lastlayer=(int)layer; }
        for(;;){
            while(*p==' ') p++;
            if(*p<'0'||*p>'9') break;
            long e=strtol(p,&q,10); p=q;
            if(*p==':'){ p++; while(*p&&*p!=' '&&*p!='\n') p++; }
            if(e>=0){ lt_push(&lt[layer],(int)e); if((int)e>=nexp) nexp=(int)e+1; }
        }
    }
    fclose(f);
    if(maxlayer<0){ fprintf(stderr,"empty trace\n"); return 1; }

    printf("route_sim: cap=%d pins/layer=%d experts=%d decay_every=%u\n",
           cap,npin,nexp,(unsigned)TIER_DECAY_EVERY);
    printf("%5s %10s %6s | %6s %6s %7s %6s %7s | %6s %6s %7s %6s %7s\n",
           "layer","probes","opt%","lru%","x-opt%","sub%","ret%","ttr",
           "clox%","x-opt%","sub%","ret%","ttr");

    PS TL={0},TC={0},TO={0}; double ksum=0; int nlay=0;
    unsigned long long tprot=0;
    uint64_t *cnt=xrealloc(NULL,nexp*sizeof *cnt);
    uint32_t *seen=xrealloc(NULL,nexp*sizeof *seen);
    unsigned char *pinned=xrealloc(NULL,nexp);
    uint32_t *off=xrealloc(NULL,(nexp+1)*sizeof *off);
    uint32_t *cur=xrealloc(NULL,nexp*sizeof *cur);

    for(int l=0;l<=maxlayer;l++){
        LT *t=&lt[l];
        if(!t->nr) continue;
        /* pins: the npin most-requested experts of this layer */
        memset(cnt,0,nexp*sizeof *cnt);
        for(size_t i=0;i<t->nr;i++) cnt[t->routed[i]]++;
        memset(pinned,0,nexp);
        for(int a=0;a<npin;a++){ int best=-1;
            for(int e=0;e<nexp;e++) if(!pinned[e]&&cnt[e]&&(best<0||cnt[e]>cnt[best])) best=e;
            if(best<0) break;
            pinned[best]=1; }
        /* probe stream: per call, uniques in first-appearance order (pinned flagged) */
        uint32_t *pr=xrealloc(NULL,t->nr*sizeof *pr); size_t np=0;
        size_t *pcs=xrealloc(NULL,(t->nc+1)*sizeof *pcs);
        memset(seen,0xff,nexp*sizeof *seen);
        for(size_t c=0;c<t->nc;c++){
            pcs[c]=np;
            size_t end = c+1<t->nc ? t->cs[c+1] : t->nr;
            for(size_t i=t->cs[c];i<end;i++){
                int e=t->routed[i];
                if(seen[e]==(uint32_t)c) continue;
                seen[e]=(uint32_t)c;
                pr[np++]=(uint32_t)e|(pinned[e]?PIN:0);
            }
        }
        pcs[t->nc]=np;
        if(np>=INF){ fprintf(stderr,"layer %d: trace too long\n",l); return 1; }
        /* occurrence lists (unpinned probes only) for next-use queries */
        memset(off,0,(nexp+1)*sizeof *off);
        for(size_t i=0;i<np;i++) if(!(pr[i]&PIN)) off[pr[i]+1]++;
        for(int e=0;e<nexp;e++) off[e+1]+=off[e];
        uint32_t *occ=xrealloc(NULL,(off[nexp]?off[nexp]:1)*sizeof *occ);
        { uint32_t *fill=xrealloc(NULL,nexp*sizeof *fill);
          memcpy(fill,off,nexp*sizeof *fill);
          for(size_t i=0;i<np;i++) if(!(pr[i]&PIN)) occ[fill[pr[i]]++]=(uint32_t)i;
          free(fill); }

        PS o,lr,cx; TierAdapt ad; unsigned long long prot=0;
        memcpy(cur,off,nexp*sizeof *cur); run_opt(pr,np,nexp,cap,occ,off,cur,&o);
        memcpy(cur,off,nexp*sizeof *cur); run_online(t,pr,pcs,nexp,cap,occ,off,cur,0,&lr,NULL,NULL);
        memcpy(cur,off,nexp*sizeof *cur); run_online(t,pr,pcs,nexp,cap,occ,off,cur,gonce?2:1,&cx,&ad,&prot);

        unsigned long long probes=o.hits+o.miss;
        printf("%5d %10llu %6.1f | %6.1f %6.1f %6.1f%% %5.1f%% %7.0f | %6.1f %6.1f %6.1f%% %5.1f%% %7.0f\n",
               l,probes,pct(o.hits,probes),
               pct(lr.hits,probes),xopt(&lr,&o),pct(lr.subopt,lr.evict),pct(lr.returned,lr.evict),
               lr.returned?(double)lr.ttr/(double)lr.returned:0.0,
               pct(cx.hits,probes),xopt(&cx,&o),pct(cx.subopt,cx.evict),pct(cx.returned,cx.evict),
               cx.returned?(double)cx.ttr/(double)cx.returned:0.0);
        TO.hits+=o.hits; TO.miss+=o.miss;
        TL.hits+=lr.hits; TL.miss+=lr.miss; TL.evict+=lr.evict; TL.subopt+=lr.subopt; TL.returned+=lr.returned; TL.ttr+=lr.ttr;
        TC.hits+=cx.hits; TC.miss+=cx.miss; TC.evict+=cx.evict; TC.subopt+=cx.subopt; TC.returned+=cx.returned; TC.ttr+=cx.ttr;
        tprot+=prot; ksum+=ad.k; nlay++;
        free(pr); free(pcs); free(occ);
    }
    unsigned long long probes=TO.hits+TO.miss;
    printf("%5s %10llu %6.1f | %6.1f %6.1f %6.1f%% %5.1f%% %7.0f | %6.1f %6.1f %6.1f%% %5.1f%% %7.0f\n",
           "TOTAL",probes,pct(TO.hits,probes),
           pct(TL.hits,probes),xopt(&TL,&TO),pct(TL.subopt,TL.evict),pct(TL.returned,TL.evict),
           TL.returned?(double)TL.ttr/(double)TL.returned:0.0,
           pct(TC.hits,probes),xopt(&TC,&TO),pct(TC.subopt,TC.evict),pct(TC.returned,TC.evict),
           TC.returned?(double)TC.ttr/(double)TC.returned:0.0);
    printf("clox: avg final k=%.2f  protected evictions=%.1f%% of %llu\n",
           nlay?ksum/nlay:0.0,pct(tprot,TC.evict),TC.evict);
    printf("sub%% = evictions whose victim returned before another resident (worse than Belady's pick)\n");
    printf("ret%%/ttr = victims re-requested at all / mean probes until they returned\n");
    return 0;
}
