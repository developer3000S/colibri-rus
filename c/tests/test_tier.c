#include <stdio.h>
#include "../tier.h"

static int fail(const char *message){
    fprintf(stderr,"tier test failed: %s\n",message);
    return 1;
}

int main(void){
    uint32_t heat[6]={20,2,8,3,30,1};
    int pinned[2]={0,1}, slot=-1, eid=-1; long gain=0;
    if(!tier_pick_swap(heat,6,pinned,2,&slot,&eid,&gain)) return fail("hot expert not promoted");
    if(slot!=1 || eid!=4 || gain!=28) return fail("wrong promotion candidate");

    uint32_t stable[4]={20,18,24,4}; int resident[2]={0,1};
    if(tier_pick_swap(stable,4,resident,2,&slot,&eid,&gain)) return fail("hysteresis did not block churn");

    tier_decay(heat,6);
    if(heat[0]!=10 || heat[1]!=1 || heat[4]!=15) return fail("heat decay");

    uint32_t freq[5]={10,10,2,18,18}, last[5]={10,90,95,20,99};
    int live[2]={0,1};
    if(!tier_pick_lfru(freq,last,100,5,live,2,&slot,&eid,&gain)) return fail("LFRU promotion");
    if(slot!=0||eid!=4) return fail("LFRU did not prefer recent ties");

    /* adaptive-k SLRU lifecycle: saturating freq, ghost memory, turnover */
    TierAdapt ad; tier_adapt_init(&ad);
    if(ad.k!=TIER_K0) return fail("initial k");
    int8_t f[8]={0}; uint32_t rl[8]={0};
    tier_admit(&f[0]);
    if(f[0]!=1) return fail("fresh admit not at 1");
    for(int i=0;i<40;i++) tier_touch(&ad,&f[0],1);
    if(f[0]!=TIER_FMAX) return fail("freq did not saturate");
    if(ad.graduated!=1) return fail("crossing k under pressure not counted once");
    rl[0]=7; tier_evict(&ad,f,8,rl,0,4);
    if(f[0]!=-TIER_FMAX) return fail("evict did not ghost the freq");
    if(ad.ev_prot!=1) return fail("protected eviction not counted");
    tier_admit(&f[0]);
    if(f[0]!=TIER_FMAX) return fail("ghost did not resume at remembered+1 (capped)");
    f[1]=-3; rl[1]=10; f[2]=-5; rl[2]=4; f[3]=-2; rl[3]=9;   /* three ghosts */
    f[4]=6;  rl[4]=8;
    tier_evict(&ad,f,8,rl,4,3);                              /* fourth ghost, cap 3 */
    if(f[2]!=0) return fail("turnover did not forget the least-recently-routed ghost");
    if(f[4]!=-6) return fail("evicted expert lost its ghost to turnover wrongly");

    /* trim-before-ghost: the victim IS the LRU (near-oldest recency in the
     * layer — the common case). At ghost cap, the oldest EXISTING ghost must
     * be forgotten and the fresh ghost must survive. */
    TierAdapt tb; tier_adapt_init(&tb);
    int8_t f2[5]={0}; uint32_t rl2[5]={0};
    f2[1]=-3; rl2[1]=10; f2[2]=-5; rl2[2]=12;                /* two ghosts at cap */
    f2[3]=4;  rl2[3]=2;                                      /* victim: oldest recency */
    tier_evict(&tb,f2,5,rl2,3,2);
    if(f2[3]!=-4) return fail("fresh ghost was forgotten instead of the oldest existing one");
    if(f2[1]!=0) return fail("oldest existing ghost not trimmed to make room");
    if(f2[2]!=-5) return fail("younger existing ghost lost wrongly");

    /* live-freq decay: every TIER_DECAY_EVERY evictions the live freqs take
     * one TIER_DECAY_OP step (linear -1: rank-preserving), so protection
     * expires unless re-earned; ghosts and freq-1 are exempt */
    TierAdapt dc; tier_adapt_init(&dc);
    int8_t fd[5]={0}; uint32_t rd[5]={0};
    fd[1]=8; fd[2]=TIER_FMAX; fd[3]=-6; fd[4]=1;   /* live, saturated, ghost, floor */
    for(unsigned i=0;i<TIER_DECAY_EVERY;i++){ fd[0]=1; tier_evict(&dc,fd,5,rd,0,5); fd[0]=0; }
    if(fd[1]!=7 || fd[2]!=TIER_FMAX-1) return fail("live freqs did not step down on the decay cadence");
    if(fd[3]!=-6) return fail("ghost freq must not decay");
    if(fd[4]!=1) return fail("freq 1 must not decay");

    /* k adaptation: high graduation rate raises k, zero rate lowers it —
     * but only on a COMPLETED hit-rate window (the change->measure->learn
     * gate), never on eviction pressure alone. */
    TierAdapt hi; tier_adapt_init(&hi);
    int8_t fh[2]={0}; uint32_t rh[2]={0};
    for(unsigned p=0;p<TIER_WIN_OPS;p++) tier_probe(&hi,p&1);  /* 50% window */
    for(unsigned i=0;i<TIER_ADAPT_EVERY;i++){
        hi.graduated++; fh[0]=1;
        tier_evict(&hi,fh,2,rh,0,2); fh[0]=0;
        tier_maybe_adapt(&hi);
    }
    if(hi.k!=TIER_K0+1) return fail("k did not rise on high graduation rate");
    if(hi.last_dir!=1) return fail("k raise did not record its direction");

    /* armed check, incomplete window: k must NOT move again */
    for(unsigned i=0;i<TIER_ADAPT_EVERY;i++){
        hi.graduated++; fh[0]=1;
        tier_evict(&hi,fh,2,rh,0,2); fh[0]=0;
        tier_maybe_adapt(&hi);
    }
    if(hi.k!=TIER_K0+1) return fail("k stepped on an incomplete hit-rate window");

    /* gradient: hit rate improved after the raise -> raising gets easier */
    for(unsigned p=0;p<TIER_WIN_OPS;p++) tier_probe(&hi,1);    /* 100% window */
    for(unsigned i=0;i<TIER_ADAPT_EVERY;i++){
        hi.graduated++; fh[0]=1;
        tier_evict(&hi,fh,2,rh,0,2); fh[0]=0;
        tier_maybe_adapt(&hi);
    }
    if(hi.rate_high!=TIER_RATE_HIGH0-TIER_RATE_STEP)
        return fail("improved hit rate after k-raise did not ease rate_high");
    if(hi.k!=TIER_K0+2) return fail("k did not keep rising on sustained graduation");

    TierAdapt lo; tier_adapt_init(&lo);
    for(unsigned p=0;p<TIER_WIN_OPS;p++) tier_probe(&lo,0);    /* all-miss window */
    for(unsigned i=0;i<TIER_ADAPT_EVERY;i++){
        fh[0]=1; tier_evict(&lo,fh,2,rh,0,2); fh[0]=0;
        tier_maybe_adapt(&lo);
    }
    if(lo.k!=TIER_K0-1) return fail("k did not fall on zero graduation rate");
    puts("tier tests: ok");
    return 0;
}
