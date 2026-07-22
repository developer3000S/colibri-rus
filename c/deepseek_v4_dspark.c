/* Amalgamated deepseek_v4_dspark.c — GLM-style source; compile with -DCOLI_V4_UNIT_* per object */
/* Umbrella API: deepseek_v4_dspark.h (included by units) */

#ifdef COLI_V4_UNIT_CONFIG_DSPARK_COMPAT
/* ######## deepseek_v4_config_dspark_compat.c ######## */
#include "deepseek_v4_dspark.h"
/* ---- begin inlined deepseek_v4_config_dspark_compat.c ---- */
#define coli_v4_config_parse coli_v4_config_parse_strict
#define coli_v4_config_load coli_v4_config_load_strict
/* ---- begin include deepseek_v4_config.c ---- */
#include "deepseek_v4_internal.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static int json_int_value(const jval *value, int *output) {
    if (!value || value->t != J_NUM || !isfinite(value->num) ||
        floor(value->num) != value->num ||
        value->num < (double)INT_MIN || value->num > (double)INT_MAX)
        return -1;
    *output = (int)value->num;
    return 0;
}

static int required_int(jval *root, const char *name, int *output,
                        char *error, size_t error_size) {
    if (json_int_value(json_get(root, name), output) != 0)
        return set_error(error, error_size, "invalid integer config field: %s", name);
    return 0;
}

static int required_float(jval *root, const char *name, float *output,
                          char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value || value->t != J_NUM || !isfinite(value->num) ||
        fabs(value->num) > (double)FLT_MAX)
        return set_error(error, error_size, "invalid numeric config field: %s", name);
    *output = (float)value->num;
    return 0;
}

static int require_string(jval *root, const char *name, const char *expected,
                          char *error, size_t error_size) {
    jval *value = json_get(root, name);
    if (!value || value->t != J_STR || strcmp(value->str, expected))
        return set_error(error, error_size, "unsupported %s (expected %s)",
                         name, expected);
    return 0;
}

int coli_v4_config_parse(ColiDeepSeekV4Config *config, const char *json,
                         char *error, size_t error_size) {
    if (!config || !json)
        return set_error(error, error_size, "invalid DeepSeek-V4 config arguments");
    memset(config, 0, sizeof(*config));
    char *arena = NULL;
    jval *root = json_parse(json, &arena);
    if (!root || root->t != J_OBJ) {
        json_free(root);
        free(arena);
        return set_error(error, error_size, "DeepSeek-V4 config is not an object");
    }
    int failed =
        require_string(root, "model_type", "deepseek_v4", error, error_size) ||
        require_string(root, "expert_dtype", "fp4", error, error_size) ||
        require_string(root, "scoring_func", "sqrtsoftplus", error, error_size) ||
        require_string(root, "topk_method", "noaux_tc", error, error_size) ||
        required_int(root, "hidden_size", &config->hidden_size, error, error_size) ||
        required_int(root, "num_hidden_layers", &config->num_hidden_layers, error, error_size) ||
        required_int(root, "num_attention_heads", &config->num_attention_heads, error, error_size) ||
        required_int(root, "head_dim", &config->head_dim, error, error_size) ||
        required_int(root, "q_lora_rank", &config->q_lora_rank, error, error_size) ||
        required_int(root, "qk_rope_head_dim", &config->qk_rope_head_dim, error, error_size) ||
        required_int(root, "o_groups", &config->o_groups, error, error_size) ||
        required_int(root, "o_lora_rank", &config->o_lora_rank, error, error_size) ||
        required_int(root, "sliding_window", &config->sliding_window, error, error_size) ||
        required_int(root, "index_n_heads", &config->index_n_heads, error, error_size) ||
        required_int(root, "index_head_dim", &config->index_head_dim, error, error_size) ||
        required_int(root, "index_topk", &config->index_topk, error, error_size) ||
        required_int(root, "n_routed_experts", &config->n_routed_experts, error, error_size) ||
        required_int(root, "num_experts_per_tok", &config->num_experts_per_tok, error, error_size) ||
        required_int(root, "n_shared_experts", &config->n_shared_experts, error, error_size) ||
        required_int(root, "moe_intermediate_size", &config->moe_intermediate_size, error, error_size) ||
        required_int(root, "num_hash_layers", &config->num_hash_layers, error, error_size) ||
        required_int(root, "num_nextn_predict_layers", &config->num_nextn_predict_layers, error, error_size) ||
        required_int(root, "hc_mult", &config->hc_mult, error, error_size) ||
        required_int(root, "hc_sinkhorn_iters", &config->hc_sinkhorn_iters, error, error_size) ||
        required_int(root, "vocab_size", &config->vocab_size, error, error_size) ||
        required_int(root, "max_position_embeddings", &config->max_position_embeddings, error, error_size) ||
        required_float(root, "rms_norm_eps", &config->rms_norm_eps, error, error_size) ||
        required_float(root, "hc_eps", &config->hc_eps, error, error_size) ||
        required_float(root, "routed_scaling_factor", &config->routed_scaling_factor, error, error_size) ||
        required_float(root, "swiglu_limit", &config->swiglu_limit, error, error_size) ||
        required_float(root, "rope_theta", &config->rope_theta, error, error_size) ||
        required_float(root, "compress_rope_theta", &config->compress_rope_theta, error, error_size);
    if (failed) {
        json_free(root);
        free(arena);
        return -1;
    }
    jval *rope = json_get(root, "rope_scaling");
    if (!rope || rope->t != J_OBJ ||
        required_int(rope, "original_max_position_embeddings",
                     &config->original_max_position_embeddings, error, error_size) ||
        required_int(rope, "beta_fast", &config->rope_beta_fast, error, error_size) ||
        required_int(rope, "beta_slow", &config->rope_beta_slow, error, error_size) ||
        required_float(rope, "factor", &config->rope_factor, error, error_size)) {
        json_free(root);
        free(arena);
        return -1;
    }
    jval *ratios = json_get(root, "compress_ratios");
    if (!ratios || ratios->t != J_ARR || ratios->len < 1 ||
        ratios->len > COLI_V4_MAX_LAYERS) {
        json_free(root);
        free(arena);
        return set_error(error, error_size, "invalid compress_ratios");
    }
    config->compress_ratio_count = ratios->len;
    for (int index = 0; index < ratios->len; index++) {
        if (json_int_value(ratios->kids[index],
                           &config->compress_ratios[index]) != 0) {
            json_free(root);
            free(arena);
            return set_error(error, error_size, "invalid compress ratio");
        }
    }
    jval *quantization = json_get(root, "quantization_config");
    if (!quantization || quantization->t != J_OBJ ||
        require_string(quantization, "fmt", "e4m3", error, error_size) ||
        require_string(quantization, "scale_fmt", "ue8m0", error, error_size)) {
        json_free(root);
        free(arena);
        return -1;
    }
    if (config->hidden_size < 1 || config->num_hidden_layers < 1 ||
        config->num_attention_heads < 1 || config->n_routed_experts < 1 ||
        config->num_experts_per_tok < 1 ||
        config->num_experts_per_tok > config->n_routed_experts ||
        config->n_shared_experts != 1 || config->hc_mult < 1 ||
        config->compress_ratio_count != config->num_hidden_layers +
                                        config->num_nextn_predict_layers) {
        json_free(root);
        free(arena);
        return set_error(error, error_size, "inconsistent DeepSeek-V4 config dimensions");
    }
    json_free(root);
    free(arena);
    return 0;
}

int coli_v4_config_load(ColiDeepSeekV4Config *config, const char *model_dir,
                        char *error, size_t error_size) {
    if (!config || !model_dir)
        return set_error(error, error_size, "invalid DeepSeek-V4 config path");
    size_t path_length = strlen(model_dir) + sizeof("/config.json");
    char *path = malloc(path_length);
    if (!path) return set_error(error, error_size, "out of memory building config path");
    snprintf(path, path_length, "%s/config.json", model_dir);
    FILE *stream = fopen(path, "rb");
    if (!stream) {
        int result = set_error(error, error_size, "cannot open %s", path);
        free(path);
        return result;
    }
    fseek(stream, 0, SEEK_END);
    long length = ftell(stream);
    rewind(stream);
    if (length < 1) {
        fclose(stream);
        free(path);
        return set_error(error, error_size, "empty config: %s", model_dir);
    }
    char *text = malloc((size_t)length + 1);
    if (!text || fread(text, 1, (size_t)length, stream) != (size_t)length) {
        free(text);
        fclose(stream);
        free(path);
        return set_error(error, error_size, "cannot read config: %s", model_dir);
    }
    text[length] = 0;
    fclose(stream);
    int result = coli_v4_config_parse(config, text, error, error_size);
    free(text);
    free(path);
    return result;
}
/* ---- end include deepseek_v4_config.c ---- */

#undef coli_v4_config_load
#undef coli_v4_config_parse

int coli_v4_config_parse(ColiDeepSeekV4Config *config, const char *json,
                         char *error, size_t error_size) {
    char strict_error[256] = {0};
    int result = coli_v4_config_parse_strict(
        config, json, strict_error, sizeof(strict_error));
    if (!result) {
        if (error && error_size) error[0] = 0;
        return 0;
    }
    if (strcmp(strict_error, "inconsistent DeepSeek-V4 config dimensions") ||
        !config || !json || !strstr(json, "\"dspark_block_size\"") ||
        config->num_hidden_layers < 1 ||
        config->compress_ratio_count <= config->num_hidden_layers)
        return set_error(error, error_size, "%s", strict_error);
    int inferred = config->compress_ratio_count - config->num_hidden_layers;
    if (inferred < 1 || inferred > COLI_V4_DSPARK_MAX_STAGES)
        return set_error(error, error_size, "%s", strict_error);
    config->num_nextn_predict_layers = inferred;
    if (config->hidden_size < 1 || config->num_hidden_layers < 1 ||
        config->num_attention_heads < 1 || config->n_routed_experts < 1 ||
        config->num_experts_per_tok < 1 ||
        config->num_experts_per_tok > config->n_routed_experts ||
        config->n_shared_experts != 1 || config->hc_mult < 1 ||
        config->compress_ratio_count != config->num_hidden_layers +
                                        config->num_nextn_predict_layers)
        return set_error(error, error_size, "%s", strict_error);
    if (error && error_size) error[0] = 0;
    return 0;
}

int coli_v4_config_load(ColiDeepSeekV4Config *config, const char *model_dir,
                        char *error, size_t error_size) {
    if (!config || !model_dir)
        return set_error(error, error_size, "invalid DeepSeek-V4 config path");
    size_t path_length = strlen(model_dir) + sizeof("/config.json");
    char *path = malloc(path_length);
    if (!path) return set_error(error, error_size, "out of memory building config path");
    snprintf(path, path_length, "%s/config.json", model_dir);
    FILE *stream = fopen(path, "rb");
    if (!stream) { free(path); return set_error(error, error_size,
                                                "cannot open config"); }
    fseek(stream, 0, SEEK_END); long length = ftell(stream); rewind(stream);
    char *text = length > 0 ? malloc((size_t)length + 1) : NULL;
    if (!text || fread(text, 1, (size_t)length, stream) != (size_t)length) {
        free(text); fclose(stream); free(path);
        return set_error(error, error_size, "cannot read config");
    }
    text[length] = 0; fclose(stream);
    int result = coli_v4_config_parse(config, text, error, error_size);
    free(text); free(path); return result;
}
/* ---- end inlined deepseek_v4_config_dspark_compat.c ---- */
#endif /* COLI_V4_UNIT_CONFIG_DSPARK_COMPAT */

#ifdef COLI_V4_UNIT_DSPARK
/* ######## deepseek_v4_dspark.c ######## */
#include <string.h>

static char *dspark_second_dot(const char *text, int character) {
    char *first = strchr(text, character);
    return first ? strchr(first + 1, character) : NULL;
}

#define strchr dspark_second_dot
/* ---- begin inlined deepseek_v4_dspark.c ---- */
#include "deepseek_v4_dspark.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "json.h"
#include "safetensors_index.h"

static int dspark_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments; va_start(arguments, format);
        vsnprintf(error, size, format, arguments); va_end(arguments);
    }
    return -1;
}

static char *read_config(const char *model_dir) {
    size_t length = strlen(model_dir) + sizeof("/config.json");
    char *path = malloc(length);
    if (!path) return NULL;
    snprintf(path, length, "%s/config.json", model_dir);
    FILE *stream = fopen(path, "rb"); free(path);
    if (!stream) return NULL;
    fseek(stream, 0, SEEK_END); long bytes = ftell(stream); rewind(stream);
    char *text = bytes > 0 ? malloc((size_t)bytes + 1) : NULL;
    if (!text || fread(text, 1, (size_t)bytes, stream) != (size_t)bytes) {
        free(text); fclose(stream); return NULL;
    }
    text[bytes] = 0; fclose(stream); return text;
}

static int json_integer(jval *root, const char *name, int *output) {
    jval *value = json_get(root, name);
    if (!value || value->t != J_NUM) return -1;
    *output = (int)value->num; return 0;
}

int coli_v4_dspark_layer_plan(ColiDeepSeekV4LayerPlan *plan,
                              const ColiDeepSeekV4Config *config, int stage,
                              char *error, size_t error_size) {
    if (!plan || !config || stage < 0 || stage >= COLI_V4_DSPARK_MAX_STAGES)
        return dspark_error(error, error_size, "invalid DSpark layer plan");
    ColiDeepSeekV4Config common = *config;
    common.num_hash_layers = 0;
    common.compress_ratios[0] = 0;
    if (coli_v4_layer_plan(plan, &common, 0, error, error_size)) return -1;
    plan->layer = stage;
    plan->compression_ratio = 0;
    plan->uses_hash_router = 0;
    plan->has_compressor = 0;
    plan->has_indexer = 0;
    for (size_t i = 0; i < plan->tensor_count; i++) {
        const char *suffix = strchr(plan->tensors[i].name, '.');
        if (!suffix) return dspark_error(error, error_size,
                                         "invalid common layer tensor name");
        char rewritten[COLI_V4_MAX_TENSOR_NAME];
        int written = snprintf(rewritten, sizeof(rewritten), "mtp.%d%s",
                               stage, suffix);
        if (written < 0 || (size_t)written >= sizeof(rewritten))
            return dspark_error(error, error_size, "DSpark tensor name too long");
        strcpy(plan->tensors[i].name, rewritten);
    }
    return 0;
}

static int require_tensor(const ColiSafetensorsIndex *index, const char *name,
                          ColiSafetensorsDType dtype, int rank,
                          const int64_t *shape, uint64_t *bytes,
                          char *error, size_t error_size) {
    const ColiSafetensorsTensor *tensor = coli_st_find(index, name);
    if (!tensor) return dspark_error(error, error_size,
                                     "missing DSpark tensor: %s", name);
    if (tensor->dtype != dtype || tensor->rank != rank)
        return dspark_error(error, error_size,
                            "DSpark dtype/rank mismatch: %s", name);
    for (int i = 0; i < rank; i++)
        if (tensor->shape[i] != shape[i])
            return dspark_error(error, error_size,
                                "DSpark shape mismatch: %s", name);
    *bytes += tensor->nbytes; return 0;
}

int coli_v4_dspark_inspect(const char *model_dir,
                           const ColiDeepSeekV4Config *config,
                           ColiDeepSeekV4DSparkManifest *manifest,
                           char *error, size_t error_size) {
    if (!model_dir || !config || !manifest)
        return dspark_error(error, error_size, "invalid DSpark inspect arguments");
    memset(manifest, 0, sizeof(*manifest));
    char *text = read_config(model_dir), *arena = NULL;
    jval *root = text ? json_parse(text, &arena) : NULL;
    if (!root || root->t != J_OBJ ||
        json_integer(root, "dspark_block_size", &manifest->block_size) ||
        json_integer(root, "dspark_noise_token_id", &manifest->noise_token_id) ||
        json_integer(root, "dspark_markov_rank", &manifest->markov_rank)) {
        free(arena); free(text);
        return dspark_error(error, error_size, "missing DSpark config fields");
    }
    jval *targets = json_get(root, "dspark_target_layer_ids");
    if (!targets || targets->t != J_ARR || targets->len < 1 ||
        targets->len > COLI_V4_DSPARK_MAX_TARGETS) {
        free(arena); free(text);
        return dspark_error(error, error_size, "invalid DSpark target layers");
    }
    manifest->target_count = targets->len;
    for (int i = 0; i < targets->len; i++) {
        if (targets->kids[i]->t != J_NUM) {
            free(arena); free(text);
            return dspark_error(error, error_size, "invalid DSpark target layer");
        }
        manifest->target_layer_ids[i] = (int)targets->kids[i]->num;
    }
    free(arena); free(text);

    ColiSafetensorsIndex *index = NULL;
    if (coli_st_index_open(&index, model_dir, error, error_size)) return -1;
    char name[160];
    for (int stage = 0; stage < COLI_V4_DSPARK_MAX_STAGES; stage++) {
        snprintf(name, sizeof(name), "mtp.%d.attn_norm.weight", stage);
        if (!coli_st_find(index, name)) break;
        manifest->stage_count++;
        ColiDeepSeekV4LayerPlan plan;
        ColiDeepSeekV4LayerStats stats;
        if (coli_v4_dspark_layer_plan(&plan, config, stage, error, error_size) ||
            coli_v4_layer_validate(&plan, index, &stats, error, error_size)) {
            coli_st_index_close(index); return -1;
        }
        manifest->common_stage_bytes[stage] = stats.total_bytes;
    }
    if (manifest->stage_count < 1) {
        coli_st_index_close(index);
        return dspark_error(error, error_size, "checkpoint has no DSpark stages");
    }
    int64_t main_weight[] = {config->hidden_size,
        (int64_t)config->hidden_size * manifest->target_count};
    int64_t main_scale[] = {(config->hidden_size + 127) / 128,
        (main_weight[1] + 127) / 128};
    int64_t hidden[] = {config->hidden_size};
    int last = manifest->stage_count - 1;
    int64_t markov[] = {config->vocab_size, manifest->markov_rank};
    int64_t confidence[] = {1, config->hidden_size + manifest->markov_rank};
    snprintf(name, sizeof(name), "mtp.0.main_proj.weight");
    int failed = require_tensor(index, name, COLI_ST_F8_E4M3, 2, main_weight,
                                &manifest->special_bytes, error, error_size);
    snprintf(name, sizeof(name), "mtp.0.main_proj.scale");
    failed |= require_tensor(index, name, COLI_ST_F8_E8M0, 2, main_scale,
                             &manifest->special_bytes, error, error_size);
    failed |= require_tensor(index, "mtp.0.main_norm.weight", COLI_ST_BF16, 1,
                             hidden, &manifest->special_bytes, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.norm.weight", last);
    failed |= require_tensor(index, name, COLI_ST_BF16, 1, hidden,
                             &manifest->special_bytes, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.markov_head.markov_w1.weight", last);
    failed |= require_tensor(index, name, COLI_ST_BF16, 2, markov,
                             &manifest->special_bytes, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.markov_head.markov_w2.weight", last);
    failed |= require_tensor(index, name, COLI_ST_BF16, 2, markov,
                             &manifest->special_bytes, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.confidence_head.proj.weight", last);
    failed |= require_tensor(index, name, COLI_ST_BF16, 2, confidence,
                             &manifest->special_bytes, error, error_size);
    coli_st_index_close(index);
    return failed ? -1 : 0;
}
/* ---- end inlined deepseek_v4_dspark.c ---- */

#undef strchr
#endif /* COLI_V4_UNIT_DSPARK */

#ifdef COLI_V4_UNIT_DSPARK_MEMORY
/* ######## deepseek_v4_dspark_memory.c ######## */
#define coli_v4_dspark_memory_plan coli_v4_dspark_memory_plan_unmargined
/* ---- begin inlined deepseek_v4_dspark_memory.c ---- */
#include "deepseek_v4_dspark.h"

#include <stdio.h>
#include <string.h>

#include "deepseek_v4_dspark.h"
#include "safetensors_index.h"

#define DSPARK_MIB UINT64_C(1048576)

int coli_v4_dspark_memory_plan(const char *model_dir,
                               const ColiDeepSeekV4Config *config,
                               ColiV4DSparkMemoryPlan *plan,
                               char *error, size_t error_size) {
    if (!model_dir || !config || !plan) return -1;
    memset(plan, 0, sizeof(*plan));
    ColiDeepSeekV4DSparkManifest manifest;
    if (coli_v4_dspark_inspect(model_dir, config, &manifest,
                               error, error_size)) return -1;
    plan->stages = manifest.stage_count;
    plan->expert_slots_per_stage = config->num_experts_per_tok;
    plan->resident_heads_bytes = manifest.special_bytes;
    for (int i = 0; i < manifest.stage_count; i++)
        if (manifest.common_stage_bytes[i] > plan->streamed_stage_bytes)
            plan->streamed_stage_bytes = manifest.common_stage_bytes[i];
    ColiSafetensorsIndex *index = NULL;
    if (coli_st_index_open(&index, model_dir, error, error_size)) return -1;
    static const char *parts[] = {
        "mtp.0.ffn.experts.0.w1.weight", "mtp.0.ffn.experts.0.w1.scale",
        "mtp.0.ffn.experts.0.w2.weight", "mtp.0.ffn.experts.0.w2.scale",
        "mtp.0.ffn.experts.0.w3.weight", "mtp.0.ffn.experts.0.w3.scale",
    };
    for (size_t i = 0; i < sizeof(parts) / sizeof(parts[0]); i++) {
        const ColiSafetensorsTensor *tensor = coli_st_find(index, parts[i]);
        if (!tensor) { coli_st_index_close(index); return -1; }
        plan->expert_record_bytes += tensor->nbytes;
    }
    coli_st_index_close(index);
    plan->minimum_expert_cache_bytes = plan->expert_record_bytes *
        (uint64_t)plan->stages * config->num_experts_per_tok;
    uint64_t capture = (uint64_t)64 * manifest.target_count * config->hc_mult *
                       config->hidden_size * sizeof(float);
    uint64_t query = (uint64_t)64 * config->hc_mult * config->hidden_size *
                     sizeof(float) * 2;
    uint64_t kv = (uint64_t)manifest.stage_count *
        (config->sliding_window + 64) * config->head_dim * sizeof(float);
    plan->working_bytes = 64 * DSPARK_MIB + capture + query + kv;
    plan->incremental_reserve_bytes = plan->resident_heads_bytes +
        plan->streamed_stage_bytes + plan->minimum_expert_cache_bytes +
        plan->working_bytes;
    return 0;
}
/* ---- end inlined deepseek_v4_dspark_memory.c ---- */

#undef coli_v4_dspark_memory_plan

int coli_v4_dspark_memory_plan(const char *model_dir,
                               const ColiDeepSeekV4Config *config,
                               ColiV4DSparkMemoryPlan *plan,
                               char *error, size_t error_size) {
    int result = coli_v4_dspark_memory_plan_unmargined(
        model_dir, config, plan, error, error_size);
    if (result) return result;
    /* Native FP8/FP4 batch kernels and OpenMP allocate short-lived per-thread
       buffers outside the explicit query/KV arrays.  Keep this model-size-
       independent margin in the RAM tier plan so low-memory hosts do not lend
       the same bytes to the target expert cache. */
    uint64_t allocator_margin = 256 * DSPARK_MIB;
    if (UINT64_MAX - plan->working_bytes < allocator_margin ||
        UINT64_MAX - plan->incremental_reserve_bytes < allocator_margin)
        return -1;
    plan->working_bytes += allocator_margin;
    plan->incremental_reserve_bytes += allocator_margin;
    return 0;
}
#endif /* COLI_V4_UNIT_DSPARK_MEMORY */

#ifdef COLI_V4_UNIT_DSPARK_RUNTIME_RESIDENT
/* ######## deepseek_v4_dspark_runtime_resident.c ######## */
#define coli_v4_dspark_layer_load coli_v4_dspark_layer_reference_load
/* ---- begin include deepseek_v4_dspark_runtime.c ---- */
#include "deepseek_v4_dspark.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int runtime_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments; va_start(arguments, format);
        vsnprintf(error, size, format, arguments); va_end(arguments);
    }
    return -1;
}

static void add_stats(ColiDeepSeekV4LayerStats *stats,
                      const ColiSafetensorsTensor *tensor) {
    stats->tensor_count++; stats->total_bytes += tensor->nbytes;
    switch (tensor->dtype) {
        case COLI_ST_BF16: stats->bf16_bytes += tensor->nbytes; break;
        case COLI_ST_F32: stats->f32_bytes += tensor->nbytes; break;
        case COLI_ST_F8_E4M3: stats->fp8_weight_bytes += tensor->nbytes; break;
        case COLI_ST_F8_E8M0: stats->fp8_scale_bytes += tensor->nbytes; break;
        case COLI_ST_I64: stats->i64_bytes += tensor->nbytes; break;
        default: break;
    }
}

int coli_v4_dspark_layer_load(ColiV4Engine *engine,
                              ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              const ColiSafetensorsIndex *index, int stage,
                              char *error, size_t error_size) {
    (void)engine;
    if (!weights || !config || !index)
        return runtime_error(error, error_size, "invalid DSpark layer load");
    memset(weights, 0, sizeof(*weights));
    if (coli_v4_dspark_layer_plan(&weights->plan, config, stage,
                                  error, error_size)) return -1;
    for (size_t i = 0; i < weights->plan.tensor_count; i++) {
        ColiDeepSeekV4TensorSpec *spec = &weights->plan.tensors[i];
        char actual[COLI_V4_MAX_TENSOR_NAME];
        strcpy(actual, spec->name);
        const ColiSafetensorsTensor *tensor = coli_st_find(index, actual);
        if (!tensor || tensor->dtype != spec->dtype ||
            tensor->rank != spec->rank) {
            coli_v4_layer_free(NULL, weights);
            return runtime_error(error, error_size,
                                 "invalid DSpark tensor: %s", actual);
        }
        for (int dimension = 0; dimension < spec->rank; dimension++)
            if (tensor->shape[dimension] != spec->shape[dimension]) {
                coli_v4_layer_free(NULL, weights);
                return runtime_error(error, error_size,
                                     "DSpark shape mismatch: %s", actual);
            }
        weights->data[i] = malloc((size_t)tensor->nbytes);
        if (!weights->data[i] ||
            coli_st_read_tensor(index, tensor, weights->data[i])) {
            coli_v4_layer_free(NULL, weights);
            return runtime_error(error, error_size,
                                 "cannot load DSpark tensor: %s", actual);
        }
        add_stats(&weights->stats, tensor);
        const char *suffix = actual + strlen("mtp.0");
        int prefix = snprintf(spec->name, sizeof(spec->name),
                              "layers.%d", stage);
        size_t suffix_length = strlen(suffix);
        if (prefix < 0 || (size_t)prefix + suffix_length >= sizeof(spec->name)) {
            coli_v4_layer_free(NULL, weights);
            return runtime_error(error, error_size,
                                 "DSpark runtime tensor name is too long");
        }
        memcpy(spec->name + prefix, suffix, suffix_length + 1);
    }
    return 0;
}
/* ---- end include deepseek_v4_dspark_runtime.c ---- */

#undef coli_v4_dspark_layer_load

#include "deepseek_v4_dspark.h"
#include "deepseek_v4_internal.h"

static int dspark_stages_resident(ColiV4Engine *engine) {
    return engine && engine->runtime.dspark_resident;
}

int coli_v4_dspark_layer_load(ColiV4Engine *engine,
                              ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              const ColiSafetensorsIndex *index, int stage,
                              char *error, size_t error_size) {
    if (!dspark_stages_resident(engine))
        return coli_v4_dspark_layer_reference_load(
            engine, weights, config, index, stage, error, error_size);
    if (!engine || !weights || !config || !index || stage < 0 ||
        stage >= COLI_V4_DSPARK_MAX_STAGES) return -1;
    if (engine->dspark_resident.index &&
        engine->dspark_resident.index != index) {
        if (error && error_size)
            snprintf(error, error_size,
                     "resident DSpark stages cannot switch model instances");
        return -1;
    }
    engine->dspark_resident.index = index;
    if (!engine->dspark_resident.ready[stage]) {
        if (coli_v4_dspark_layer_reference_load(
                engine, &engine->dspark_resident.layers[stage], config, index,
                stage, error, error_size)) return -1;
        engine->dspark_resident.ready[stage] = 1;
        engine->dspark_resident.total_bytes +=
            engine->dspark_resident.layers[stage].stats.total_bytes;
        fprintf(stderr, "dspark_stage_resident stage=%d total=%.3fGiB\n",
                stage, engine->dspark_resident.total_bytes / 1073741824.0);
    }
    *weights = engine->dspark_resident.layers[stage]; return 0;
}

void coli_v4_dspark_layer_release(ColiV4Engine *engine,
                                  ColiDeepSeekV4LayerWeights *weights) {
    if (!weights) return;
    int stage = weights->plan.layer;
    if (dspark_stages_resident(engine) && engine && stage >= 0 &&
        stage < COLI_V4_DSPARK_MAX_STAGES &&
        engine->dspark_resident.ready[stage] &&
        weights->data[0] == engine->dspark_resident.layers[stage].data[0]) {
        memset(weights, 0, sizeof(*weights)); return;
    }
    coli_v4_layer_free(engine, weights);
}
#endif /* COLI_V4_UNIT_DSPARK_RUNTIME_RESIDENT */

#ifdef COLI_V4_UNIT_DSPARK_EXPERT_STORE
/* ######## deepseek_v4_dspark_expert_store.c ######## */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "deepseek_v4_dspark.h"

static int dspark_store_snprintf_v2(char *output, size_t size,
                                    const char *format, ...) {
    char rewritten[256];
    /* strlen("layers.%d.ffn.experts.") is 22.  Comparing 23 bytes also
     * compares the literal NUL with the following '%' in the real format,
     * preventing the mtp namespace rewrite and loading target experts. */
    if (!strncmp(format, "layers.%d.ffn.experts.", 22))
        snprintf(rewritten, sizeof(rewritten), "mtp.%s", format + 7);
    else
        snprintf(rewritten, sizeof(rewritten), "%s", format);
    va_list arguments; va_start(arguments, format);
    int result = vsnprintf(output, size, rewritten, arguments);
    va_end(arguments); return result;
}

#define snprintf dspark_store_snprintf_v2
#define coli_deepseek_v4_expert_store_open \
    coli_deepseek_v4_dspark_expert_store_open
/* ---- begin include deepseek_v4_expert_store.c ---- */
#define _GNU_SOURCE
#include "deepseek_v4_internal.h"

#include <assert.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "safetensors_index.h"

#ifdef COLI_V4_EXPERIMENTAL_PREFETCH_BATCH
int coli_st_prefetch_many(
    const ColiSafetensorsIndex *index, const int *shards,
    const uint64_t *offsets, const size_t *lengths, size_t count);
#endif

enum { V4_W1 = 0, V4_W2 = 1, V4_W3 = 2, V4_MATRIX_COUNT = 3 };

typedef struct {
    const ColiSafetensorsTensor *weight[V4_MATRIX_COUNT];
    const ColiSafetensorsTensor *scale[V4_MATRIX_COUNT];
    int shard;
    uint64_t scale_offset;
    uint64_t scale_bytes;
    uint64_t weight_offset;
    uint64_t weight_bytes;
    uint64_t record_bytes;
} V4ExpertRecord;

typedef struct {
    int expert;
    unsigned references;
    uint64_t used;
    unsigned char *slab;
} V4ExpertSlot;

typedef struct {
    ColiSafetensorsIndex *index;
    int layers;
    int experts_per_layer;
    int slots_per_layer;
    uint64_t record_bytes;
    V4ExpertRecord *records;
    V4ExpertSlot *slots;
    uint64_t clock;
    unsigned active_leases;
    ColiExpertStoreStats stats;
    pthread_mutex_t mutex;
} V4ExpertStoreState;

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list args;
        va_start(args, format);
        vsnprintf(error, size, format, args);
        va_end(args);
    }
    return -1;
}

static int compare_tensors(const void *left, const void *right) {
    const ColiSafetensorsTensor *const *a = left;
    const ColiSafetensorsTensor *const *b = right;
    return ((*a)->offset > (*b)->offset) - ((*a)->offset < (*b)->offset);
}

static int contiguous_group(const ColiSafetensorsTensor *const input[3],
                            int *shard, uint64_t *offset, uint64_t *bytes) {
    const ColiSafetensorsTensor *parts[3] = {input[0], input[1], input[2]};
    qsort(parts, 3, sizeof(parts[0]), compare_tensors);
    if (parts[0]->shard != parts[1]->shard || parts[1]->shard != parts[2]->shard ||
        parts[0]->offset + parts[0]->nbytes != parts[1]->offset ||
        parts[1]->offset + parts[1]->nbytes != parts[2]->offset)
        return -1;
    *shard = parts[0]->shard;
    *offset = parts[0]->offset;
    *bytes = parts[2]->offset + parts[2]->nbytes - parts[0]->offset;
    return 0;
}

static int validate_matrix(const ColiSafetensorsTensor *weight,
                           const ColiSafetensorsTensor *scale) {
    if (!weight || !scale || weight->dtype != COLI_ST_I8 ||
        scale->dtype != COLI_ST_F8_E8M0 || weight->rank != 2 || scale->rank != 2 ||
        weight->shape[0] != scale->shape[0] || weight->shape[1] <= 0 ||
        scale->shape[1] <= 0)
        return -1;
    int64_t logical_columns = weight->shape[1] * 2;
    return scale->shape[1] * 32 == logical_columns ? 0 : -1;
}

static int build_record(V4ExpertStoreState *state, int layer, int expert,
                        V4ExpertRecord *record, char *error, size_t error_size) {
    static const char *matrix_names[V4_MATRIX_COUNT] = {"w1", "w2", "w3"};
    char name[160];
    memset(record, 0, sizeof(*record));
    for (int matrix = 0; matrix < V4_MATRIX_COUNT; matrix++) {
        snprintf(name, sizeof(name), "layers.%d.ffn.experts.%d.%s.weight",
                 layer, expert, matrix_names[matrix]);
        record->weight[matrix] = coli_st_find(state->index, name);
        snprintf(name, sizeof(name), "layers.%d.ffn.experts.%d.%s.scale",
                 layer, expert, matrix_names[matrix]);
        record->scale[matrix] = coli_st_find(state->index, name);
        if (validate_matrix(record->weight[matrix], record->scale[matrix]) != 0)
            return set_error(error, error_size,
                             "invalid native FP4 expert matrix: layer=%d expert=%d %s",
                             layer, expert, matrix_names[matrix]);
    }
    int scale_shard = -1, weight_shard = -1;
    if (contiguous_group(record->scale, &scale_shard, &record->scale_offset,
                         &record->scale_bytes) != 0 ||
        contiguous_group(record->weight, &weight_shard, &record->weight_offset,
                         &record->weight_bytes) != 0 || scale_shard != weight_shard)
        return set_error(error, error_size,
                         "expert is not two contiguous ranges: layer=%d expert=%d",
                         layer, expert);
    record->shard = scale_shard;
    record->record_bytes = record->scale_bytes + record->weight_bytes;
    return 0;
}

static V4ExpertRecord *get_record(V4ExpertStoreState *state, ColiExpertKey key) {
    if (key.layer < 0 || key.layer >= state->layers || key.expert < 0 ||
        key.expert >= state->experts_per_layer)
        return NULL;
    return &state->records[(size_t)key.layer * state->experts_per_layer + key.expert];
}

static V4ExpertSlot *layer_slots(V4ExpertStoreState *state, int layer) {
    return state->slots + (size_t)layer * state->slots_per_layer;
}

static void fill_tensor_view(ColiTensorView *view,
                             const V4ExpertRecord *record,
                             const V4ExpertSlot *slot, int matrix) {
    const ColiSafetensorsTensor *weight = record->weight[matrix];
    const ColiSafetensorsTensor *scale = record->scale[matrix];
    memset(view, 0, sizeof(*view));
    view->format = COLI_TENSOR_FP4_NATIVE_BLOCK;
    view->scale_format = COLI_SCALE_UE8M0;
    view->data = slot->slab + record->scale_bytes +
                 (weight->offset - record->weight_offset);
    view->scales = slot->slab + (scale->offset - record->scale_offset);
    view->data_bytes = (size_t)weight->nbytes;
    view->scale_bytes = (size_t)scale->nbytes;
    view->rows = weight->shape[0];
    view->columns = weight->shape[1] * 2;
    view->block_rows = 1;
    view->block_columns = 32;
}

static int lookup(ColiExpertStore *store, ColiExpertKey key,
                  ColiExpertView *view) {
    if (!store || !store->state || !view) {
        if (view) memset(view, 0, sizeof(*view));
        return -1;
    }
    V4ExpertStoreState *state = store->state;
    V4ExpertRecord *record = get_record(state, key);
    if (!record) {
        memset(view, 0, sizeof(*view));
        return -1;
    }
    pthread_mutex_lock(&state->mutex);
    state->stats.requests++;
    V4ExpertSlot *slots = layer_slots(state, key.layer);
    V4ExpertSlot *slot = NULL;
    for (int i = 0; i < state->slots_per_layer; i++) {
        if (slots[i].slab && slots[i].expert == key.expert) {
            slot = &slots[i];
            state->stats.hits++;
            break;
        }
    }
    if (!slot) {
        for (int i = 0; i < state->slots_per_layer; i++) {
            if (!slots[i].references && (!slot || !slots[i].slab ||
                                         (slot->slab && slots[i].used < slot->used)))
                slot = &slots[i];
        }
        if (!slot) {
            pthread_mutex_unlock(&state->mutex);
            memset(view, 0, sizeof(*view));
            return -1;
        }
        if (!slot->slab) {
            slot->slab = malloc((size_t)state->record_bytes);
            if (!slot->slab) {
                pthread_mutex_unlock(&state->mutex);
                memset(view, 0, sizeof(*view));
                return -1;
            }
            state->stats.resident_bytes += state->record_bytes;
        }
        /* A short read must never expose a partially overwritten old slot. */
        slot->expert = -1;
        if (coli_st_read_at(state->index, record->shard, record->scale_offset,
                            (size_t)record->scale_bytes, slot->slab) != 0 ||
            coli_st_read_at(state->index, record->shard, record->weight_offset,
                            (size_t)record->weight_bytes,
                            slot->slab + record->scale_bytes) != 0) {
            pthread_mutex_unlock(&state->mutex);
            memset(view, 0, sizeof(*view));
            return -1;
        }
        slot->expert = key.expert;
        state->stats.misses++;
        state->stats.bytes_read += record->record_bytes;
    }
    slot->references++;
    state->active_leases++;
    slot->used = ++state->clock;
    memset(view, 0, sizeof(*view));
    view->key = key;
    fill_tensor_view(&view->gate, record, slot, V4_W1);
    fill_tensor_view(&view->down, record, slot, V4_W2);
    fill_tensor_view(&view->up, record, slot, V4_W3);
    view->lease = slot;
    pthread_mutex_unlock(&state->mutex);
    return 0;
}

static void release(ColiExpertStore *store, ColiExpertView *view) {
    if (!store || !store->state || !view || !view->lease) {
        if (view) memset(view, 0, sizeof(*view));
        return;
    }
    V4ExpertStoreState *state = store->state;
    V4ExpertSlot *slot = view->lease;
    pthread_mutex_lock(&state->mutex);
    if (slot->references) slot->references--;
    if (state->active_leases) state->active_leases--;
    pthread_mutex_unlock(&state->mutex);
    memset(view, 0, sizeof(*view));
}

static int prefetch(ColiExpertStore *store, const ColiExpertKey *keys,
                    size_t count) {
    if (!store || !store->state || (!keys && count)) return 0;
    V4ExpertStoreState *state = store->state;
    int accepted = 0;
#ifdef COLI_V4_EXPERIMENTAL_PREFETCH_BATCH
    size_t capacity = count * 2, ranges = 0;
    int *shards = malloc(capacity * sizeof(*shards));
    uint64_t *offsets = malloc(capacity * sizeof(*offsets));
    size_t *lengths = malloc(capacity * sizeof(*lengths));
    int candidates = 0;
    if ((!shards || !offsets || !lengths) && capacity) {
        free(lengths); free(offsets); free(shards); return 0;
    }
    pthread_mutex_lock(&state->mutex);
    for (size_t i = 0; i < count; i++) {
        V4ExpertRecord *record = get_record(state, keys[i]);
        if (!record) continue;
        int resident = 0;
        V4ExpertSlot *slots = layer_slots(state, keys[i].layer);
        for (int slot = 0; slot < state->slots_per_layer; slot++)
            if (slots[slot].slab && slots[slot].expert == keys[i].expert) {
                resident = 1; break;
            }
        if (resident) continue;
        shards[ranges] = record->shard;
        offsets[ranges] = record->scale_offset;
        lengths[ranges++] = (size_t)record->scale_bytes;
        shards[ranges] = record->shard;
        offsets[ranges] = record->weight_offset;
        lengths[ranges++] = (size_t)record->weight_bytes;
        candidates++;
    }
    pthread_mutex_unlock(&state->mutex);
    if (candidates && !coli_st_prefetch_many(
            state->index, shards, offsets, lengths, ranges))
        accepted = candidates;
    free(lengths); free(offsets); free(shards);
#else
    for (size_t i = 0; i < count; i++) {
        V4ExpertRecord *record = get_record(state, keys[i]);
        if (!record) continue;
        if (coli_st_prefetch_at(state->index, record->shard, record->scale_offset,
                                (size_t)record->scale_bytes) == 0 &&
            coli_st_prefetch_at(state->index, record->shard, record->weight_offset,
                                (size_t)record->weight_bytes) == 0)
            accepted++;
    }
#endif
    pthread_mutex_lock(&state->mutex);
    state->stats.prefetched += (uint64_t)accepted;
    pthread_mutex_unlock(&state->mutex);
    return accepted;
}

static void stats(const ColiExpertStore *store, ColiExpertStoreStats *output) {
    if (!store || !store->state || !output) return;
    V4ExpertStoreState *state = store->state;
    pthread_mutex_lock(&state->mutex);
    *output = state->stats;
    pthread_mutex_unlock(&state->mutex);
}

static void destroy(ColiExpertStore *store) {
    if (!store) return;
    V4ExpertStoreState *state = store->state;
    if (state) {
        assert(state->active_leases == 0 && "destroy with active expert leases");
        for (int i = 0; i < state->layers * state->slots_per_layer; i++)
            free(state->slots[i].slab);
        pthread_mutex_destroy(&state->mutex);
        coli_st_index_close(state->index);
        free(state->records);
        free(state->slots);
        free(state);
    }
    free(store);
}

int coli_deepseek_v4_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options, ColiExpertStore **output,
    char *error, size_t error_size) {
    static const ColiExpertStoreOps operations = {
        lookup, release, prefetch, stats, destroy
    };
    if (!options || !output || !options->model_dir || options->layers < 1 ||
        options->experts_per_layer < 1 || !options->cache_bytes)
        return set_error(error, error_size, "invalid DeepSeek-V4 ExpertStore options");
    *output = NULL;
    ColiExpertStore *store = calloc(1, sizeof(*store));
    V4ExpertStoreState *state = calloc(1, sizeof(*state));
    if (!store || !state) {
        free(store);
        free(state);
        return set_error(error, error_size, "out of memory creating ExpertStore");
    }
    pthread_mutex_init(&state->mutex, NULL);
    state->layers = options->layers;
    state->experts_per_layer = options->experts_per_layer;
    if (coli_st_index_open(&state->index, options->model_dir, error, error_size) != 0)
        goto fail;
    size_t record_count = (size_t)state->layers * state->experts_per_layer;
    state->records = calloc(record_count, sizeof(*state->records));
    if (!state->records) {
        set_error(error, error_size, "out of memory creating expert manifest");
        goto fail;
    }
    for (int layer = 0; layer < state->layers; layer++) {
        for (int expert = 0; expert < state->experts_per_layer; expert++) {
            V4ExpertRecord *record = &state->records[
                (size_t)layer * state->experts_per_layer + expert];
            if (build_record(state, layer, expert, record, error, error_size) != 0)
                goto fail;
            if (!state->record_bytes) state->record_bytes = record->record_bytes;
            if (record->record_bytes != state->record_bytes) {
                set_error(error, error_size, "non-uniform expert size at layer=%d expert=%d",
                          layer, expert);
                goto fail;
            }
        }
    }
    state->slots_per_layer = (int)(options->cache_bytes /
        ((uint64_t)state->layers * state->record_bytes));
    int minimum_slots = state->experts_per_layer < 6
        ? state->experts_per_layer : 6;
    if (state->slots_per_layer < minimum_slots) {
        set_error(error, error_size,
                  "cache budget cannot hold %d active experts per layer "
                  "(need %llu bytes)", minimum_slots,
                  (unsigned long long)((uint64_t)state->layers * minimum_slots *
                                       state->record_bytes));
        goto fail;
    }
    if (state->slots_per_layer > state->experts_per_layer)
        state->slots_per_layer = state->experts_per_layer;
    state->slots = calloc((size_t)state->layers * state->slots_per_layer,
                          sizeof(*state->slots));
    if (!state->slots) {
        set_error(error, error_size, "out of memory creating expert cache slots");
        goto fail;
    }
    for (int i = 0; i < state->layers * state->slots_per_layer; i++)
        state->slots[i].expert = -1;
    state->stats.capacity_bytes = (uint64_t)state->layers *
                                  state->slots_per_layer * state->record_bytes;
    store->ops = &operations;
    store->state = state;
    *output = store;
    return 0;

fail:
    if (state->slots) free(state->slots);
    free(state->records);
    coli_st_index_close(state->index);
    pthread_mutex_destroy(&state->mutex);
    free(state);
    free(store);
    return -1;
}
/* ---- end include deepseek_v4_expert_store.c ---- */

#undef coli_deepseek_v4_expert_store_open
#undef snprintf
#endif /* COLI_V4_UNIT_DSPARK_EXPERT_STORE */

#ifdef COLI_V4_UNIT_DSPARK_ATTENTION
/* ######## deepseek_v4_dspark_attention.c ######## */
/* ---- begin inlined deepseek_v4_dspark_attention.c ---- */
#include "deepseek_v4_dspark.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "native_quant.h"
#include "native_quant_batch.h"

struct ColiV4DSparkAttentionState {
    int window_size;
    int head_dim;
    int valid;
    float *kv;
    int *positions;
};

static int ds_attn_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments; va_start(arguments, format);
        vsnprintf(error, size, format, arguments); va_end(arguments);
    }
    return -1;
}

static const void *ds_layer_data(const ColiDeepSeekV4LayerWeights *weights,
                                 const char *suffix,
                                 const ColiDeepSeekV4TensorSpec **spec) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, spec);
}

static int ds_fp8_view(ColiTensorView *view,
                       const ColiDeepSeekV4LayerWeights *weights,
                       const char *prefix) {
    char name[128]; const ColiDeepSeekV4TensorSpec *ws = NULL, *ss = NULL;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    const void *data = ds_layer_data(weights, name, &ws);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    const void *scales = ds_layer_data(weights, name, &ss);
    if (!data || !scales || !ws || !ss || ws->rank != 2 || ss->rank != 2)
        return -1;
    memset(view, 0, sizeof(*view));
    view->format = COLI_TENSOR_FP8_E4M3_BLOCK;
    view->scale_format = COLI_SCALE_UE8M0;
    view->data = data; view->scales = scales;
    view->rows = ws->shape[0]; view->columns = ws->shape[1];
    view->data_bytes = (size_t)view->rows * view->columns;
    view->scale_bytes = (size_t)ss->shape[0] * ss->shape[1];
    view->block_rows = 128; view->block_columns = 128;
    return 0;
}

int coli_v4_dspark_attention_create(ColiV4DSparkAttentionState **output,
                                    const ColiDeepSeekV4Config *config) {
    if (!output || !config || config->sliding_window < 1 || config->head_dim < 1)
        return -1;
    *output = NULL;
    ColiV4DSparkAttentionState *state = calloc(1, sizeof(*state));
    if (!state) return -1;
    state->window_size = config->sliding_window;
    state->head_dim = config->head_dim;
    state->kv = calloc((size_t)state->window_size * state->head_dim,
                       sizeof(*state->kv));
    state->positions = malloc((size_t)state->window_size *
                              sizeof(*state->positions));
    if (!state->kv || !state->positions) {
        coli_v4_dspark_attention_destroy(state); return -1;
    }
    coli_v4_dspark_attention_reset(state); *output = state; return 0;
}

void coli_v4_dspark_attention_reset(ColiV4DSparkAttentionState *state) {
    if (!state) return;
    state->valid = 0;
    memset(state->kv, 0, (size_t)state->window_size * state->head_dim *
                         sizeof(*state->kv));
    for (int i = 0; i < state->window_size; i++) state->positions[i] = -1;
}

void coli_v4_dspark_attention_destroy(ColiV4DSparkAttentionState *state) {
    if (!state) return;
    free(state->positions); free(state->kv); free(state);
}

int coli_v4_dspark_attention_context_count(
    const ColiV4DSparkAttentionState *state) { return state ? state->valid : 0; }

int coli_v4_dspark_attention_precompute_context(
    ColiV4DSparkAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const float *main_x, int start_position, int batch,
    char *error, size_t error_size) {
    if (!state || !weights || !config || !main_x || start_position < 0 ||
        batch < 1 || batch > 64 || state->head_dim != config->head_dim)
        return ds_attn_error(error, error_size, "invalid DSpark context KV input");
    ColiTensorView wkv;
    if (ds_fp8_view(&wkv, weights, "attn.wkv"))
        return ds_attn_error(error, error_size, "missing DSpark wkv");
    int d = config->head_dim, rope = config->qk_rope_head_dim;
    float *kv = malloc((size_t)batch * d * sizeof(*kv));
    float *norm = malloc((size_t)d * sizeof(*norm));
    int end = start_position + batch;
    size_t pairs = (size_t)rope / 2;
    float *cosines = malloc((size_t)end * pairs * sizeof(*cosines));
    float *sines = malloc((size_t)end * pairs * sizeof(*sines));
    const uint16_t *raw_norm = ds_layer_data(weights, "attn.kv_norm.weight", NULL);
    if (!kv || !norm || !cosines || !sines || !raw_norm) {
        free(sines); free(cosines); free(norm); free(kv);
        return ds_attn_error(error, error_size, "out of memory in DSpark context KV");
    }
    for (int i = 0; i < d; i++) norm[i] = coli_bf16_decode(raw_norm[i]);
    int result = coli_fp8_matmul_batch_ref(kv, &wkv, main_x, batch);
    if (!result) coli_bf16_round_array(kv, (size_t)batch * d);
    if (!result) result = coli_v4_rope_precompute(
        cosines, sines, rope, end, 0, config->rope_theta,
        config->rope_factor, config->rope_beta_fast, config->rope_beta_slow);
    for (int item = 0; !result && item < batch; item++) {
        float *item_kv = kv + (size_t)item * d;
        result = coli_v4_rmsnorm(item_kv, item_kv, norm, d, config->rms_norm_eps);
        if (result) break;
        coli_bf16_round_array(item_kv, (size_t)d);
        int position = start_position + item;
        coli_v4_rope_apply(item_kv + d - rope, 1, rope,
                           cosines + (size_t)position * pairs,
                           sines + (size_t)position * pairs, 0);
        coli_bf16_round_array(item_kv + d - rope, (size_t)rope);
        size_t nope = (size_t)(d - rope);
        float *qdq = malloc(nope * sizeof(*qdq));
        uint8_t *scales = malloc((nope + 63) / 64);
        if (!qdq || !scales || coli_fp8_activation_qdq_ref(
                qdq, scales, item_kv, nope, 64)) result = -1;
        if (!result) {
            memcpy(item_kv, qdq, nope * sizeof(*item_kv));
            coli_bf16_round_array(item_kv, nope);
            int slot = coli_v4_dspark_position_window_put(
                state->positions, state->window_size, &state->valid, position);
            if (slot < 0)
                result = ds_attn_error(
                    error, error_size,
                    "invalid DSpark context window update position=%d", position);
            else
                memcpy(state->kv + (size_t)slot * d, item_kv,
                       (size_t)d * sizeof(*item_kv));
        }
        free(scales); free(qdq);
    }
    free(sines); free(cosines); free(norm); free(kv);
    return result ? ds_attn_error(error, error_size,
                                  "DSpark context KV precompute failed") : 0;
}
/* ---- end inlined deepseek_v4_dspark_attention.c ---- */


#include "deepseek_v4_dspark.h"
#include "deepseek_v4_internal.h"

int coli_v4_dspark_attention_block(
    float *outputs, ColiV4DSparkAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *inputs,
    int query_start_position, int batch, char *error, size_t error_size) {
    if (!outputs || !state || !weights || !config || !inputs ||
        query_start_position < 0 || batch < 1 || batch > 64)
        return ds_attn_error(error, error_size, "invalid DSpark query block");
    int hidden = config->hidden_size, heads = config->num_attention_heads;
    int d = config->head_dim, rope = config->qk_rope_head_dim;
    int qr = config->q_lora_rank, groups = config->o_groups;
    int orank = config->o_lora_rank;
    size_t qwidth = (size_t)heads * d;
    size_t oawidth = (size_t)groups * orank;
    ColiTensorView wqa, wqb, wkv, woa, wob;
    if (ds_fp8_view(&wqa, weights, "attn.wq_a") ||
        ds_fp8_view(&wqb, weights, "attn.wq_b") ||
        ds_fp8_view(&wkv, weights, "attn.wkv") ||
        ds_fp8_view(&woa, weights, "attn.wo_a") ||
        ds_fp8_view(&wob, weights, "attn.wo_b"))
        return ds_attn_error(error, error_size, "missing DSpark attention matrix");
    float *qa = calloc((size_t)batch * qr, sizeof(float));
    float *q = calloc((size_t)batch * qwidth, sizeof(float));
    float *kv = calloc((size_t)batch * d, sizeof(float));
    float *attended = calloc((size_t)batch * qwidth, sizeof(float));
    float *oa = calloc((size_t)batch * oawidth, sizeof(float));
    float *norm = malloc((size_t)(qr > d ? qr : d) * sizeof(float));
    int end = query_start_position + batch; size_t pairs = (size_t)rope / 2;
    float *cosines = malloc((size_t)end * pairs * sizeof(float));
    float *sines = malloc((size_t)end * pairs * sizeof(float));
    int context = state->valid, kv_count = context + batch;
    float *all_kv = malloc((size_t)kv_count * d * sizeof(float));
    int *indices = malloc((size_t)kv_count * sizeof(int));
    if (!qa || !q || !kv || !attended || !oa || !norm || !cosines ||
        !sines || !all_kv || !indices) {
        free(indices); free(all_kv); free(sines); free(cosines); free(norm);
        free(oa); free(attended); free(kv); free(q); free(qa);
        return ds_attn_error(error, error_size, "out of memory in DSpark block");
    }
    int result = coli_fp8_matmul_batch_ref(qa, &wqa, inputs, batch);
    if (result) ds_attn_error(error, error_size,
                              "DSpark attention q_a matmul failed");
    if (!result) coli_bf16_round_array(qa, (size_t)batch * qr);
    const uint16_t *qnorm = ds_layer_data(weights, "attn.q_norm.weight", NULL);
    if (!qnorm) result = ds_attn_error(
        error, error_size, "missing DSpark attention q_norm");
    for (int i = 0; !result && i < qr; i++) norm[i] = coli_bf16_decode(qnorm[i]);
    for (int item = 0; !result && item < batch; item++) {
        float *row = qa + (size_t)item * qr;
        result = coli_v4_rmsnorm(row, row, norm, qr, config->rms_norm_eps);
        if (!result) coli_bf16_round_array(row, (size_t)qr);
    }
    if (!result) {
        result = coli_fp8_matmul_batch_ref(q, &wqb, qa, batch);
        if (result) ds_attn_error(error, error_size,
                                  "DSpark attention q_b matmul failed");
    }
    if (!result) coli_bf16_round_array(q, (size_t)batch * qwidth);
    for (int item = 0; !result && item < batch; item++)
        for (int head = 0; head < heads; head++) {
            float *row = q + (size_t)item * qwidth + (size_t)head * d;
            float squares = 0.0f;
            for (int i = 0; i < d; i++) squares += row[i] * row[i];
            float scale = 1.0f / sqrtf(squares / d + config->rms_norm_eps);
            for (int i = 0; i < d; i++) row[i] = coli_bf16_round(row[i] * scale);
        }
    if (!result) {
        result = coli_fp8_matmul_batch_ref(kv, &wkv, inputs, batch);
        if (result) ds_attn_error(error, error_size,
                                  "DSpark attention query KV matmul failed");
    }
    if (!result) coli_bf16_round_array(kv, (size_t)batch * d);
    const uint16_t *knorm = ds_layer_data(weights, "attn.kv_norm.weight", NULL);
    if (!knorm) result = ds_attn_error(
        error, error_size, "missing DSpark attention kv_norm");
    for (int i = 0; !result && i < d; i++) norm[i] = coli_bf16_decode(knorm[i]);
    if (!result) {
        result = coli_v4_rope_precompute(
            cosines, sines, rope, end, 0, config->rope_theta,
            config->rope_factor, config->rope_beta_fast, config->rope_beta_slow);
        if (result) ds_attn_error(error, error_size,
                                  "DSpark attention RoPE precompute failed");
    }
    for (int item = 0; !result && item < batch; item++) {
        int position = query_start_position + item;
        float *item_kv = kv + (size_t)item * d;
        result = coli_v4_rmsnorm(item_kv, item_kv, norm, d, config->rms_norm_eps);
        if (result) break;
        coli_bf16_round_array(item_kv, (size_t)d);
        for (int head = 0; head < heads; head++) {
            float *r = q + (size_t)item * qwidth + (size_t)head * d + d - rope;
            coli_v4_rope_apply(r, 1, rope,
                               cosines + (size_t)position * pairs,
                               sines + (size_t)position * pairs, 0);
            coli_bf16_round_array(r, (size_t)rope);
        }
        coli_v4_rope_apply(item_kv + d - rope, 1, rope,
                           cosines + (size_t)position * pairs,
                           sines + (size_t)position * pairs, 0);
        coli_bf16_round_array(item_kv + d - rope, (size_t)rope);
        size_t nope = (size_t)(d - rope);
        float *qdq = malloc(nope * sizeof(float));
        uint8_t *scales = malloc((nope + 63) / 64);
        if (!qdq || !scales || coli_fp8_activation_qdq_ref(
                qdq, scales, item_kv, nope, 64))
            result = ds_attn_error(
                error, error_size,
                "DSpark attention query KV quantization failed item=%d", item);
        if (!result) {
            memcpy(item_kv, qdq, nope * sizeof(float));
            coli_bf16_round_array(item_kv, nope);
        }
        free(scales); free(qdq);
    }
    int copied = 0;
    for (int slot = 0; !result && slot < state->window_size; slot++)
        if (state->positions[slot] >= 0) {
            memcpy(all_kv + (size_t)copied * d,
                   state->kv + (size_t)slot * d, (size_t)d * sizeof(float));
            copied++;
        }
    if (copied != context) result = ds_attn_error(
        error, error_size,
        "DSpark attention context state mismatch valid=%d copied=%d",
        context, copied);
    if (!result) memcpy(all_kv + (size_t)context * d, kv,
                        (size_t)batch * d * sizeof(float));
    for (int i = 0; i < kv_count; i++) indices[i] = i;
    const float *sinks = ds_layer_data(weights, "attn.attn_sink", NULL);
    for (int item = 0; !result && item < batch; item++) {
        result = coli_v4_sparse_attention_ref(
            attended + (size_t)item * qwidth,
            q + (size_t)item * qwidth, all_kv, sinks, indices,
            heads, d, kv_count, kv_count, 1.0f / sqrtf((float)d));
        if (result) ds_attn_error(
            error, error_size,
            "DSpark sparse attention failed item=%d position=%d context=%d",
            item, query_start_position + item, context);
        int position = query_start_position + item;
        for (int head = 0; !result && head < heads; head++) {
            float *r = attended + (size_t)item * qwidth +
                       (size_t)head * d + d - rope;
            coli_v4_rope_apply(r, 1, rope,
                               cosines + (size_t)position * pairs,
                               sines + (size_t)position * pairs, 1);
            coli_bf16_round_array(r, (size_t)rope);
        }
    }
    int heads_per_group = heads / groups;
    int group_width = heads_per_group * d;
    int scale_columns = (hidden + 127) / 128;
    int scale_rows = (orank + 127) / 128;
    float *group_inputs = malloc((size_t)batch * group_width * sizeof(float));
    float *group_outputs = malloc((size_t)batch * orank * sizeof(float));
    if (!group_inputs || !group_outputs) result = ds_attn_error(
        error, error_size, "out of memory in DSpark grouped output");
    for (int group = 0; !result && group < groups; group++) {
        for (int item = 0; item < batch; item++)
            memcpy(group_inputs + (size_t)item * group_width,
                   attended + (size_t)item * qwidth +
                   (size_t)group * group_width,
                   (size_t)group_width * sizeof(float));
        ColiTensorView view = woa;
        view.rows = orank; view.columns = group_width;
        view.data = (const uint8_t *)woa.data +
                    (size_t)group * orank * group_width;
        view.scales = (const uint8_t *)woa.scales +
                      (size_t)group * scale_rows * scale_columns;
        view.data_bytes = (size_t)orank * group_width;
        view.scale_bytes = (size_t)scale_rows * scale_columns;
        result = coli_fp8_matmul_batch_ref(group_outputs, &view,
                                            group_inputs, batch);
        if (result) ds_attn_error(
            error, error_size,
            "DSpark attention o_a matmul failed group=%d rows=%lld columns=%lld "
            "data_bytes=%zu scale_bytes=%zu",
            group, (long long)view.rows, (long long)view.columns,
            view.data_bytes, view.scale_bytes);
        for (int item = 0; !result && item < batch; item++)
            memcpy(oa + (size_t)item * oawidth + (size_t)group * orank,
                   group_outputs + (size_t)item * orank,
                   (size_t)orank * sizeof(float));
    }
    if (!result) coli_bf16_round_array(oa, (size_t)batch * oawidth);
    if (!result) {
        result = coli_fp8_matmul_batch_ref(outputs, &wob, oa, batch);
        if (result) ds_attn_error(error, error_size,
                                  "DSpark attention o_b matmul failed");
    }
    if (!result) coli_bf16_round_array(outputs, (size_t)batch * hidden);
    free(group_outputs); free(group_inputs); free(indices); free(all_kv);
    free(sines); free(cosines); free(norm); free(oa); free(attended);
    free(kv); free(q); free(qa);
    if (!result) return 0;
    if (error && error_size && error[0]) return -1;
    return ds_attn_error(error, error_size,
                         "DSpark non-causal attention failed");
}
#endif /* COLI_V4_UNIT_DSPARK_ATTENTION */

#ifdef COLI_V4_UNIT_DSPARK_BLOCK
/* ######## deepseek_v4_dspark_block.c ######## */
#define coli_v4_block_window_batch_ref coli_v4_dspark_causal_block_unused
/* ---- begin include deepseek_v4_block_batch.c ---- */
#define coli_v4_block_token_ref coli_v4_block_token_batch_serial_ref
#define coli_v4_block_window_token_ref coli_v4_block_window_token_batch_serial_ref
/* ---- begin include deepseek_v4_block.c ---- */
#include "deepseek_v4_internal.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "native_quant.h"

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *value(const ColiDeepSeekV4LayerWeights *weights,
                         const char *suffix,
                         const ColiDeepSeekV4TensorSpec **spec) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, spec);
}

static int fp8_view(ColiTensorView *view,
                    const ColiDeepSeekV4LayerWeights *weights,
                    const char *prefix) {
    char name[128];
    const ColiDeepSeekV4TensorSpec *ws = NULL, *ss = NULL;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    const void *data = value(weights, name, &ws);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    const void *scales = value(weights, name, &ss);
    if (!data || !scales || !ws || !ss || ws->rank != 2) return -1;
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_UE8M0, data, scales,
        (size_t)(ws->shape[0] * ws->shape[1]),
        (size_t)(ss->shape[0] * ss->shape[1]),
        ws->shape[0], ws->shape[1], 128, 128
    };
    return 0;
}

static void decode_bf16(float *output, const uint16_t *input, size_t count) {
    for (size_t i = 0; i < count; i++) output[i] = coli_bf16_decode(input[i]);
}

static int normalized_hc_pre(float *reduced, float *post, float *comb,
                             float *normalized, const float *input_hc,
                             const ColiDeepSeekV4LayerWeights *weights,
                             const ColiDeepSeekV4Config *config,
                             const char *branch, const char *norm_name) {
    char name[64];
    snprintf(name, sizeof(name), "hc_%s_fn", branch);
    const float *function = value(weights, name, NULL);
    snprintf(name, sizeof(name), "hc_%s_scale", branch);
    const float *scale = value(weights, name, NULL);
    snprintf(name, sizeof(name), "hc_%s_base", branch);
    const float *base = value(weights, name, NULL);
    const uint16_t *raw_norm = value(weights, norm_name, NULL);
    int d = config->hidden_size;
    float *norm = malloc((size_t)d * sizeof(*norm));
    if (!function || !scale || !base || !raw_norm || !norm) {
        free(norm);
        return -1;
    }
    decode_bf16(norm, raw_norm, (size_t)d);
    int result = coli_v4_hc_pre(reduced, post, comb, input_hc, function,
                                scale, base, config->hc_mult, d,
                                config->hc_sinkhorn_iters,
                                config->rms_norm_eps, config->hc_eps);
    if (!result) {
        coli_bf16_round_array(reduced, (size_t)d);
        result = coli_v4_rmsnorm(normalized, reduced, norm, d,
                                 config->rms_norm_eps);
        coli_bf16_round_array(normalized, (size_t)d);
    }
    free(norm);
    return result;
}

static int moe_token(float *output,
                     const ColiDeepSeekV4LayerWeights *weights,
                     const ColiDeepSeekV4Config *config,
                     ColiExpertStore *store, const float *input, int token) {
    int d = config->hidden_size;
    int n = config->n_routed_experts;
    int topk = config->num_experts_per_tok;
    size_t gate_count = (size_t)n * d;
    float *gate = malloc(gate_count * sizeof(*gate));
    float *route_weights = malloc((size_t)topk * sizeof(*route_weights));
    int *indices = malloc((size_t)topk * sizeof(*indices));
    float *expert_output = malloc((size_t)d * sizeof(*expert_output));
    float *shared_output = malloc((size_t)d * sizeof(*shared_output));
    if (!gate || !route_weights || !indices || !expert_output || !shared_output) {
        free(shared_output); free(expert_output); free(indices);
        free(route_weights); free(gate);
        return -1;
    }
    decode_bf16(gate, value(weights, "ffn.gate.weight", NULL), gate_count);
    const int64_t *table = value(weights, "ffn.gate.tid2eid", NULL);
    const float *bias = value(weights, "ffn.gate.bias", NULL);
    int result = token < 0 || token >= config->vocab_size;
    if (!result && weights->plan.uses_hash_router) {
        if (!table) result = -1;
    }
    if (!result && weights->plan.uses_hash_router) {
        for (int i = 0; i < topk; i++)
            indices[i] = (int)table[(size_t)token * topk + i];
    }
    if (!result) result = coli_v4_route(
        route_weights, indices, input, gate, bias,
        weights->plan.uses_hash_router ? indices : NULL,
        n, d, topk, config->routed_scaling_factor);

    ColiTensorView w1, w2, w3;
    if (!result && (fp8_view(&w1, weights, "ffn.shared_experts.w1") ||
                    fp8_view(&w2, weights, "ffn.shared_experts.w2") ||
                    fp8_view(&w3, weights, "ffn.shared_experts.w3"))) result = -1;
    if (!result) result = coli_v4_shared_expert_forward_ref(
        shared_output, &w1, &w2, &w3, input, config->swiglu_limit);
    if (!result) memset(output, 0, (size_t)d * sizeof(*output));
    for (int expert_id = 0; !result && expert_id < n; expert_id++) {
        int rank = -1;
        for (int candidate = 0; candidate < topk; candidate++)
            if (indices[candidate] == expert_id) rank = candidate;
        if (rank < 0) continue;
        ColiExpertView expert;
        if (coli_expert_lookup(store,
                               (ColiExpertKey){weights->plan.layer, expert_id},
                               &expert)) {
            result = -1;
            break;
        }
        result = coli_v4_expert_forward_ref(expert_output, &expert, input,
                                             route_weights[rank],
                                             config->swiglu_limit);
        coli_expert_release(store, &expert);
        if (!result)
            for (int i = 0; i < d; i++) output[i] += expert_output[i];
    }
    if (!result)
        for (int i = 0; i < d; i++)
            output[i] = coli_bf16_round(output[i] + shared_output[i]);
    free(shared_output); free(expert_output); free(indices);
    free(route_weights); free(gate);
    return result;
}

static int block_token_impl(float *output_hc,
                            ColiDeepSeekV4WindowAttentionState *attention,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size) {
    if (!output_hc || !weights || !config || !experts || !input_hc)
        return set_error(error, error_size, "invalid block arguments");
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *residual = malloc(hd * sizeof(*residual));
    float *state = malloc(hd * sizeof(*state));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *normalized = malloc((size_t)d * sizeof(*normalized));
    float *branch = malloc((size_t)d * sizeof(*branch));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    if (!residual || !state || !reduced || !normalized || !branch || !post || !comb) {
        free(comb); free(post); free(branch); free(normalized);
        free(reduced); free(state); free(residual);
        return set_error(error, error_size, "out of memory in block");
    }
    memcpy(residual, input_hc, hd * sizeof(*residual));
    int result = normalized_hc_pre(reduced, post, comb, normalized, input_hc,
                                   weights, config, "attn", "attn_norm.weight");
    if (!result) result = attention
        ? coli_v4_attention_window_token_ref(branch, attention, weights, config,
                                             normalized, position, error, error_size)
        : coli_v4_attention_token_ref(branch, weights, config, normalized,
                                      position, error, error_size);
    if (!result) result = coli_v4_hc_post(state, branch, residual, post, comb, hc, d);
    if (!result) coli_bf16_round_array(state, hd);

    if (!result) memcpy(residual, state, hd * sizeof(*residual));
    if (!result) result = normalized_hc_pre(reduced, post, comb, normalized, state,
                                            weights, config, "ffn", "ffn_norm.weight");
    if (!result) result = moe_token(branch, weights, config, experts, normalized, token);
    if (!result) result = coli_v4_hc_post(output_hc, branch, residual, post, comb, hc, d);
    if (!result) coli_bf16_round_array(output_hc, hd);

    free(comb); free(post); free(branch); free(normalized);
    free(reduced); free(state); free(residual);
    return result ? set_error(error, error_size, "block computation failed") : 0;
}

int coli_v4_block_token_ref(float *output_hc,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size) {
    return block_token_impl(output_hc, NULL, weights, config, experts, input_hc,
                            token, position, error, error_size);
}

int coli_v4_block_window_token_ref(
    float *output_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *input_hc, int token, int position,
    char *error, size_t error_size) {
    return block_token_impl(output_hc, attention, weights, config, experts,
                            input_hc, token, position, error, error_size);
}
/* ---- end include deepseek_v4_block.c ---- */

#undef coli_v4_block_token_ref
#undef coli_v4_block_window_token_ref

#include "deepseek_v4_internal.h"
#include "native_quant_batch.h"

static int shared_expert_batch(float *outputs, const ColiTensorView *w1,
                               const ColiTensorView *w2,
                               const ColiTensorView *w3,
                               const float *inputs, int batch,
                               float swiglu_limit) {
    int intermediate = (int)w1->rows, d = (int)w2->rows;
    float *gate = malloc((size_t)batch * intermediate * sizeof(*gate));
    float *up = malloc((size_t)batch * intermediate * sizeof(*up));
    float *activated = malloc((size_t)batch * intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp8_matmul_batch_ref(gate, w1, inputs, batch) ||
                 coli_fp8_matmul_batch_ref(up, w3, inputs, batch);
    for (int item = 0; !result && item < batch; item++) {
        float *item_gate = gate + (size_t)item * intermediate;
        float *item_up = up + (size_t)item * intermediate;
        float *item_activated = activated + (size_t)item * intermediate;
        coli_bf16_round_array(item_gate, (size_t)intermediate);
        coli_bf16_round_array(item_up, (size_t)intermediate);
        result = coli_v4_swiglu(item_activated, item_gate, item_up,
                                intermediate, swiglu_limit);
        if (!result) coli_bf16_round_array(item_activated,
                                           (size_t)intermediate);
    }
    if (!result) result = coli_fp8_matmul_batch_ref(
        outputs, w2, activated, batch);
    if (!result) coli_bf16_round_array(outputs, (size_t)batch * d);
    free(activated); free(up); free(gate);
    return result;
}

static int routed_expert_batch(float *outputs, const ColiExpertView *expert,
                               const float *inputs, const float *route_weights,
                               int batch, float swiglu_limit) {
    int intermediate = (int)expert->gate.rows;
    int d = (int)expert->down.rows;
    float *gate = malloc((size_t)batch * intermediate * sizeof(*gate));
    float *up = malloc((size_t)batch * intermediate * sizeof(*up));
    float *activated = malloc((size_t)batch * intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp4_matmul_batch_ref(gate, &expert->gate, inputs, batch) ||
                 coli_fp4_matmul_batch_ref(up, &expert->up, inputs, batch);
    for (int item = 0; !result && item < batch; item++) {
        float *item_gate = gate + (size_t)item * intermediate;
        float *item_up = up + (size_t)item * intermediate;
        float *item_activated = activated + (size_t)item * intermediate;
        coli_bf16_round_array(item_gate, (size_t)intermediate);
        coli_bf16_round_array(item_up, (size_t)intermediate);
        result = coli_v4_swiglu(item_activated, item_gate, item_up,
                                intermediate, swiglu_limit);
        for (int i = 0; !result && i < intermediate; i++)
            item_activated[i] = coli_bf16_round(
                item_activated[i] * route_weights[item]);
    }
    if (!result) result = coli_fp4_matmul_batch_ref(
        outputs, &expert->down, activated, batch);
    if (!result) coli_bf16_round_array(outputs, (size_t)batch * d);
    free(activated); free(up); free(gate);
    return result;
}

static int moe_batch(float *outputs,
                     const ColiDeepSeekV4LayerWeights *weights,
                     const ColiDeepSeekV4Config *config,
                     ColiExpertStore *store, const float *inputs,
                     const int *tokens, int batch,
                     char *error, size_t error_size) {
    int d = config->hidden_size, n = config->n_routed_experts;
    int topk = config->num_experts_per_tok;
    float *gate = malloc((size_t)n * d * sizeof(*gate));
    float *route_weights = malloc((size_t)batch * topk * sizeof(*route_weights));
    int *indices = malloc((size_t)batch * topk * sizeof(*indices));
    float *shared = malloc((size_t)batch * d * sizeof(*shared));
    float *compact_inputs = malloc((size_t)batch * d * sizeof(*compact_inputs));
    float *compact_outputs = malloc((size_t)batch * d * sizeof(*compact_outputs));
    float *compact_weights = malloc((size_t)batch * sizeof(*compact_weights));
    int *compact_items = malloc((size_t)batch * sizeof(*compact_items));
    if (!gate || !route_weights || !indices || !shared || !compact_inputs ||
        !compact_outputs || !compact_weights || !compact_items) {
        free(compact_items); free(compact_weights); free(compact_outputs);
        free(compact_inputs); free(shared); free(indices); free(route_weights);
        free(gate);
        return set_error(error, error_size,
                         "out of memory in DSpark MoE scratch");
    }
    const uint16_t *raw_gate = value(weights, "ffn.gate.weight", NULL);
    if (!raw_gate) {
        free(compact_items); free(compact_weights); free(compact_outputs);
        free(compact_inputs); free(shared); free(indices); free(route_weights);
        free(gate);
        return set_error(error, error_size, "missing DSpark MoE gate weight");
    }
    decode_bf16(gate, raw_gate, (size_t)n * d);
    const int64_t *table = value(weights, "ffn.gate.tid2eid", NULL);
    const float *bias = value(weights, "ffn.gate.bias", NULL);
    int result = 0;
    for (int item = 0; !result && item < batch; item++) {
        int *item_indices = indices + (size_t)item * topk;
        float *item_weights = route_weights + (size_t)item * topk;
        if (tokens[item] < 0 || tokens[item] >= config->vocab_size)
            result = set_error(
                error, error_size,
                "DSpark MoE invalid token item=%d token=%d vocab=%d",
                item, tokens[item], config->vocab_size);
        if (!result && weights->plan.uses_hash_router) {
            if (!table) result = set_error(
                error, error_size, "missing DSpark MoE hash routing table");
            else for (int i = 0; i < topk; i++)
                item_indices[i] = (int)table[(size_t)tokens[item] * topk + i];
        }
        if (!result && coli_v4_route(
                item_weights, item_indices, inputs + (size_t)item * d,
                gate, bias, weights->plan.uses_hash_router ? item_indices : NULL,
                n, d, topk, config->routed_scaling_factor))
            result = set_error(error, error_size,
                               "DSpark MoE routing failed item=%d token=%d",
                               item, tokens[item]);
    }
    ColiTensorView w1, w2, w3;
    if (!result && fp8_view(&w1, weights, "ffn.shared_experts.w1"))
        result = set_error(error, error_size,
                           "invalid DSpark shared expert w1");
    if (!result && fp8_view(&w2, weights, "ffn.shared_experts.w2"))
        result = set_error(error, error_size,
                           "invalid DSpark shared expert w2");
    if (!result && fp8_view(&w3, weights, "ffn.shared_experts.w3"))
        result = set_error(error, error_size,
                           "invalid DSpark shared expert w3");
    if (!result && shared_expert_batch(
            shared, &w1, &w2, &w3, inputs, batch, config->swiglu_limit))
        result = set_error(error, error_size,
                           "DSpark shared expert batch failed");
    if (!result) memset(outputs, 0, (size_t)batch * d * sizeof(*outputs));

    for (int expert_id = 0; !result && expert_id < n; expert_id++) {
        int count = 0;
        for (int item = 0; item < batch; item++) {
            for (int rank = 0; rank < topk; rank++) {
                if (indices[(size_t)item * topk + rank] == expert_id) {
                    compact_items[count] = item;
                    compact_weights[count] =
                        route_weights[(size_t)item * topk + rank];
                    memcpy(compact_inputs + (size_t)count * d,
                           inputs + (size_t)item * d, (size_t)d * sizeof(float));
                    count++;
                }
            }
        }
        if (!count) continue;
        ColiExpertView expert;
        if (coli_expert_lookup(store,
                               (ColiExpertKey){weights->plan.layer, expert_id},
                               &expert)) {
            result = set_error(
                error, error_size,
                "DSpark expert lookup failed layer=%d expert=%d count=%d",
                weights->plan.layer, expert_id, count);
            break;
        }
        if (routed_expert_batch(compact_outputs, &expert, compact_inputs,
                                compact_weights, count,
                                config->swiglu_limit))
            result = set_error(
                error, error_size,
                "DSpark routed expert batch failed layer=%d expert=%d count=%d",
                weights->plan.layer, expert_id, count);
        coli_expert_release(store, &expert);
        for (int compact = 0; !result && compact < count; compact++) {
            float *destination = outputs + (size_t)compact_items[compact] * d;
            const float *source = compact_outputs + (size_t)compact * d;
            for (int i = 0; i < d; i++) destination[i] += source[i];
        }
    }
    for (int item = 0; !result && item < batch; item++)
        for (int i = 0; i < d; i++)
            outputs[(size_t)item * d + i] = coli_bf16_round(
                outputs[(size_t)item * d + i] + shared[(size_t)item * d + i]);
    free(compact_items); free(compact_weights); free(compact_outputs);
    free(compact_inputs); free(shared); free(indices); free(route_weights);
    free(gate); return result;
}

int coli_v4_block_window_batch_ref(
    float *outputs_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens, int start_position, int batch,
    char *error, size_t error_size) {
    if (!outputs_hc || !attention || !weights || !config || !experts ||
        !inputs_hc || !tokens || batch < 1 || batch > 64) return -1;
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *states = malloc((size_t)batch * hd * sizeof(*states));
    float *normalized_ffn = malloc((size_t)batch * d * sizeof(*normalized_ffn));
    float *branches = malloc((size_t)batch * d * sizeof(*branches));
    float *posts = malloc((size_t)batch * hc * sizeof(*posts));
    float *combs = malloc((size_t)batch * hc * hc * sizeof(*combs));
    float *residual = malloc(hd * sizeof(*residual));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *normalized = malloc((size_t)d * sizeof(*normalized));
    float *branch = malloc((size_t)d * sizeof(*branch));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    if (!states || !normalized_ffn || !branches || !posts || !combs ||
        !residual || !reduced || !normalized || !branch || !post || !comb) {
        free(comb); free(post); free(branch); free(normalized); free(reduced);
        free(residual); free(combs); free(posts); free(branches);
        free(normalized_ffn); free(states); return -1;
    }
    int result = 0;
    for (int item = 0; !result && item < batch; item++) {
        const float *input = inputs_hc + (size_t)item * hd;
        float *state = states + (size_t)item * hd;
        memcpy(residual, input, hd * sizeof(float));
        result = normalized_hc_pre(reduced, post, comb, normalized, input,
                                   weights, config, "attn", "attn_norm.weight");
        if (!result) result = coli_v4_attention_window_token_ref(
            branch, attention, weights, config, normalized,
            start_position + item, error, error_size);
        if (!result) result = coli_v4_hc_post(state, branch, residual,
                                              post, comb, hc, d);
        if (!result) coli_bf16_round_array(state, hd);
        if (!result) result = normalized_hc_pre(
            reduced, posts + (size_t)item * hc,
            combs + (size_t)item * hc * hc,
            normalized_ffn + (size_t)item * d, state,
            weights, config, "ffn", "ffn_norm.weight");
    }
    if (!result) result = moe_batch(branches, weights, config, experts,
                                    normalized_ffn, tokens, batch,
                                    error, error_size);
    for (int item = 0; !result && item < batch; item++) {
        result = coli_v4_hc_post(
            outputs_hc + (size_t)item * hd,
            branches + (size_t)item * d,
            states + (size_t)item * hd,
            posts + (size_t)item * hc,
            combs + (size_t)item * hc * hc, hc, d);
        if (!result) coli_bf16_round_array(
            outputs_hc + (size_t)item * hd, hd);
    }
    free(comb); free(post); free(branch); free(normalized); free(reduced);
    free(residual); free(combs); free(posts); free(branches);
    free(normalized_ffn); free(states);
    return result ? set_error(error, error_size, "batched block failed") : 0;
}
/* ---- end include deepseek_v4_block_batch.c ---- */

#undef coli_v4_block_window_batch_ref

#include "deepseek_v4_dspark.h"
#include "deepseek_v4_dspark.h"

int coli_v4_dspark_block(
    float *outputs_hc, ColiV4DSparkAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens,
    int query_start_position, int batch,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!outputs_hc || !attention || !weights || !config || !experts ||
        !inputs_hc || !tokens || batch < 1 || batch > 64)
        return set_error(error, error_size, "invalid DSpark block arguments");
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *states = malloc((size_t)batch * hd * sizeof(float));
    float *attn_norm = malloc((size_t)batch * d * sizeof(float));
    float *attn_branch = malloc((size_t)batch * d * sizeof(float));
    float *attn_post = malloc((size_t)batch * hc * sizeof(float));
    float *attn_comb = malloc((size_t)batch * hc * hc * sizeof(float));
    float *ffn_norm = malloc((size_t)batch * d * sizeof(float));
    float *ffn_branch = malloc((size_t)batch * d * sizeof(float));
    float *ffn_post = malloc((size_t)batch * hc * sizeof(float));
    float *ffn_comb = malloc((size_t)batch * hc * hc * sizeof(float));
    float *reduced = malloc((size_t)d * sizeof(float));
    if (!states || !attn_norm || !attn_branch || !attn_post || !attn_comb ||
        !ffn_norm || !ffn_branch || !ffn_post || !ffn_comb || !reduced) {
        free(reduced); free(ffn_comb); free(ffn_post); free(ffn_branch);
        free(ffn_norm); free(attn_comb); free(attn_post); free(attn_branch);
        free(attn_norm); free(states);
        return set_error(error, error_size,
                         "out of memory in DSpark block scratch");
    }
    int result = 0;
    const char *failed_step = NULL;
    for (int item = 0; !result && item < batch; item++) {
        result = normalized_hc_pre(
            reduced, attn_post + (size_t)item * hc,
            attn_comb + (size_t)item * hc * hc,
            attn_norm + (size_t)item * d,
            inputs_hc + (size_t)item * hd,
            weights, config, "attn", "attn_norm.weight");
        if (result) failed_step = "attention HC pre";
    }
    if (!result) {
        result = coli_v4_dspark_attention_block(
            attn_branch, attention, weights, config, attn_norm,
            query_start_position, batch, error, error_size);
        if (result) failed_step = "attention";
    }
    for (int item = 0; !result && item < batch; item++) {
        result = coli_v4_hc_post(
            states + (size_t)item * hd,
            attn_branch + (size_t)item * d,
            inputs_hc + (size_t)item * hd,
            attn_post + (size_t)item * hc,
            attn_comb + (size_t)item * hc * hc, hc, d);
        if (result) failed_step = "attention HC post";
        if (!result) coli_bf16_round_array(states + (size_t)item * hd, hd);
    }
    for (int item = 0; !result && item < batch; item++) {
        result = normalized_hc_pre(
            reduced, ffn_post + (size_t)item * hc,
            ffn_comb + (size_t)item * hc * hc,
            ffn_norm + (size_t)item * d,
            states + (size_t)item * hd,
            weights, config, "ffn", "ffn_norm.weight");
        if (result) failed_step = "FFN HC pre";
    }
    if (!result) {
        result = moe_batch(ffn_branch, weights, config, experts, ffn_norm,
                           tokens, batch, error, error_size);
        if (result) failed_step = "MoE";
    }
    for (int item = 0; !result && item < batch; item++) {
        result = coli_v4_hc_post(
            outputs_hc + (size_t)item * hd,
            ffn_branch + (size_t)item * d,
            states + (size_t)item * hd,
            ffn_post + (size_t)item * hc,
            ffn_comb + (size_t)item * hc * hc, hc, d);
        if (result) failed_step = "FFN HC post";
        if (!result) coli_bf16_round_array(outputs_hc + (size_t)item * hd, hd);
    }
    free(reduced); free(ffn_comb); free(ffn_post); free(ffn_branch);
    free(ffn_norm); free(attn_comb); free(attn_post); free(attn_branch);
    free(attn_norm); free(states);
    if (!result) return 0;
    char detail[256] = {0};
    if (error && error_size && error[0])
        snprintf(detail, sizeof(detail), "%s", error);
    return set_error(error, error_size,
                     "DSpark block failed layer=%d step=%s%s%s",
                     weights->plan.layer, failed_step ? failed_step : "unknown",
                     detail[0] ? ": " : "", detail);
}
#endif /* COLI_V4_UNIT_DSPARK_BLOCK */

#ifdef COLI_V4_UNIT_DSPARK_FINAL
/* ######## deepseek_v4_dspark_final.c ######## */
#include "native_quant.h"
/* ---- begin inlined deepseek_v4_dspark_final.c ---- */
#include "deepseek_v4_dspark.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "deepseek_v4_internal.h"
#include "safetensors_index.h"
#include "tensor_io.h"

struct ColiV4DSparkFinal {
    int hidden, hc;
    float eps, hc_eps;
    ColiFloatTensor function, base, scale, norm;
};

void coli_v4_dspark_final_close(ColiV4DSparkFinal *final) {
    if (!final) return;
    coli_float_tensor_free(&final->norm);
    coli_float_tensor_free(&final->scale);
    coli_float_tensor_free(&final->base);
    coli_float_tensor_free(&final->function);
    free(final);
}

int coli_v4_dspark_final_open(ColiV4DSparkFinal **output,
                              const char *model_dir,
                              const ColiDeepSeekV4Config *config,
                              const ColiDeepSeekV4DSparkManifest *manifest,
                              char *error, size_t error_size) {
    if (!output || !model_dir || !config || !manifest ||
        manifest->stage_count < 1) return -1;
    *output = NULL; ColiSafetensorsIndex *index = NULL;
    if (coli_st_index_open(&index, model_dir, error, error_size)) return -1;
    ColiV4DSparkFinal *final = calloc(1, sizeof(*final));
    if (!final) { coli_st_index_close(index); return -1; }
    final->hidden = config->hidden_size; final->hc = config->hc_mult;
    final->eps = config->rms_norm_eps; final->hc_eps = config->hc_eps;
    char name[160]; int last = manifest->stage_count - 1, failed = 0;
    snprintf(name, sizeof(name), "mtp.%d.hc_head_fn", last);
    failed |= coli_tensor_load_f32(&final->function, index, name,
                                   error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.hc_head_base", last);
    failed |= coli_tensor_load_f32(&final->base, index, name, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.hc_head_scale", last);
    failed |= coli_tensor_load_f32(&final->scale, index, name, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.norm.weight", last);
    failed |= coli_tensor_load_f32(&final->norm, index, name, error, error_size);
    coli_st_index_close(index);
    if (failed) { coli_v4_dspark_final_close(final); return -1; }
    *output = final; return 0;
}

int coli_v4_dspark_final_hidden(ColiV4DSparkFinal *final,
                                float *outputs, const float *states_hc,
                                int batch) {
    if (!final || !outputs || !states_hc || batch < 1 || batch > 64) return -1;
    int d = final->hidden, hc = final->hc, flattened = d * hc;
    if (hc > 16) return -1;
    for (int item = 0; item < batch; item++) {
        const float *state = states_hc + (size_t)item * flattened;
        float *output = outputs + (size_t)item * d;
        float square = 0.0f, pre[16];
        for (int i = 0; i < flattened; i++) square += state[i] * state[i];
        float inverse = 1.0f / sqrtf(square / flattened + final->eps);
        for (int copy = 0; copy < hc; copy++) {
            float mix = 0.0f;
            for (int i = 0; i < flattened; i++)
                mix += final->function.data[(size_t)copy * flattened + i] *
                       state[i];
            float z = mix * inverse * final->scale.data[0] +
                      final->base.data[copy];
            float sigmoid = z >= 0.0f ? 1.0f / (1.0f + expf(-z))
                                      : expf(z) / (1.0f + expf(z));
            pre[copy] = sigmoid + final->hc_eps;
        }
        for (int i = 0; i < d; i++) {
            float value = 0.0f;
            for (int copy = 0; copy < hc; copy++)
                value += pre[copy] * state[(size_t)copy * d + i];
            output[i] = coli_bf16_round(value);
        }
        if (coli_v4_rmsnorm(output, output, final->norm.data, d, final->eps))
            return -1;
        coli_bf16_round_array(output, (size_t)d);
    }
    return 0;
}
/* ---- end inlined deepseek_v4_dspark_final.c ---- */
#endif /* COLI_V4_UNIT_DSPARK_FINAL */

#ifdef COLI_V4_UNIT_DSPARK_RUNNER
/* ######## deepseek_v4_dspark_runner.c ######## */
#include "native_quant.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_dspark.h"

#define coli_v4_dspark_runner_open coli_v4_dspark_runner_owned_open
#define coli_v4_dspark_runner_close coli_v4_dspark_runner_owned_close
/* ---- begin inlined deepseek_v4_dspark_runner.c ---- */
#include "deepseek_v4_dspark.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_dspark.h"
#include "deepseek_v4_dspark.h"
#include "deepseek_v4_dspark.h"
#include "deepseek_v4_dspark.h"
#include "deepseek_v4_dspark.h"
#include "deepseek_v4_dspark.h"
#include "deepseek_v4_internal.h"
#include "safetensors_index.h"

struct ColiV4DSparkRunner {
    ColiDeepSeekV4Config config;
    ColiDeepSeekV4DSparkManifest manifest;
    char *model_dir;
    ColiSafetensorsIndex *dspark_index;
    ColiSafetensorsIndex *target_index;
    ColiV4DSparkHeads *heads;
    ColiV4DSparkFinal *final;
    ColiExpertStore *experts;
    ColiV4DSparkAttentionState *attention[COLI_V4_DSPARK_MAX_STAGES];
    uint64_t loaded_stage_peak;
    ColiV4Engine *engine;
};

static int runner_error(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
    return -1;
}

static int runner_embedding(ColiV4DSparkRunner *runner, float *state,
                            int token) {
    const ColiSafetensorsTensor *embed =
        coli_st_find(runner->dspark_index, "embed.weight");
    int d = runner->config.hidden_size, hc = runner->config.hc_mult;
    uint16_t *raw = malloc((size_t)d * sizeof(*raw));
    if (!embed || embed->dtype != COLI_ST_BF16 || !raw || token < 0 ||
        token >= runner->config.vocab_size ||
        coli_st_read_at(runner->dspark_index, embed->shard,
                        embed->offset + (uint64_t)token * d * sizeof(*raw),
                        (size_t)d * sizeof(*raw), raw)) {
        free(raw); return -1;
    }
    for (int copy = 0; copy < hc; copy++)
        for (int i = 0; i < d; i++)
            state[(size_t)copy * d + i] = coli_bf16_decode(raw[i]);
    free(raw); return 0;
}

void coli_v4_dspark_runner_close(ColiV4DSparkRunner *runner) {
    if (!runner) return;
    for (int stage = 0; stage < runner->manifest.stage_count; stage++)
        coli_v4_dspark_attention_destroy(runner->attention[stage]);
    if (runner->experts) runner->experts->ops->destroy(runner->experts);
    coli_v4_dspark_final_close(runner->final);
    coli_v4_dspark_heads_close(runner->heads);
    coli_st_index_close(runner->target_index);
    coli_st_index_close(runner->dspark_index);
    free(runner->model_dir); free(runner);
}

int coli_v4_dspark_runner_open(ColiV4DSparkRunner **output,
                               const char *dspark_model_dir,
                               const char *target_model_dir,
                               const ColiDeepSeekV4Config *config,
                               uint64_t expert_cache_bytes,
                               char *error, size_t error_size) {
    if (!output || !dspark_model_dir || !target_model_dir || !config) return -1;
    *output = NULL; ColiV4DSparkRunner *runner = calloc(1, sizeof(*runner));
    if (!runner) return -1;
    runner->config = *config;
    runner->model_dir = malloc(strlen(dspark_model_dir) + 1);
    if (!runner->model_dir) { coli_v4_dspark_runner_close(runner); return -1; }
    strcpy(runner->model_dir, dspark_model_dir);
    if (coli_v4_dspark_inspect(dspark_model_dir, config, &runner->manifest,
                               error, error_size) ||
        coli_st_index_open(&runner->dspark_index, dspark_model_dir,
                           error, error_size) ||
        coli_st_index_open(&runner->target_index, target_model_dir,
                           error, error_size) ||
        coli_v4_dspark_heads_open(&runner->heads, dspark_model_dir, config,
                                  &runner->manifest, error, error_size) ||
        coli_v4_dspark_final_open(&runner->final, dspark_model_dir, config,
                                  &runner->manifest, error, error_size)) {
        coli_v4_dspark_runner_close(runner); return -1;
    }
    for (int stage = 0; stage < runner->manifest.stage_count; stage++) {
        if (runner->manifest.common_stage_bytes[stage] > runner->loaded_stage_peak)
            runner->loaded_stage_peak = runner->manifest.common_stage_bytes[stage];
        if (coli_v4_dspark_attention_create(&runner->attention[stage], config)) {
            coli_v4_dspark_runner_close(runner); return -1;
        }
    }
    ColiDeepSeekV4ExpertStoreOptions options = {
        dspark_model_dir, runner->manifest.stage_count,
        config->n_routed_experts, expert_cache_bytes, -1, 0};
    if (coli_deepseek_v4_dspark_expert_store_open(
            &options, &runner->experts, error, error_size)) {
        coli_v4_dspark_runner_close(runner); return -1;
    }
    *output = runner; return 0;
}

int coli_v4_dspark_runner_prefill(ColiV4DSparkRunner *runner,
                                  const float *main_x,
                                  int start_position, int batch,
                                  char *error, size_t error_size) {
    if (!runner || !main_x || start_position < 0 || batch < 1 || batch > 64)
        return runner_error(error, error_size, "invalid DSpark prefill");
    for (int stage = 0; stage < runner->manifest.stage_count; stage++) {
        ColiDeepSeekV4LayerWeights weights;
        if (coli_v4_dspark_layer_load(runner->engine, &weights, &runner->config,
                                      runner->dspark_index, stage,
                                      error, error_size)) return -1;
        int result = coli_v4_dspark_attention_precompute_context(
            runner->attention[stage], &weights, &runner->config,
            main_x, start_position, batch, error, error_size);
        coli_v4_dspark_layer_release(runner->engine, &weights);
        if (result) return -1;
    }
    return 0;
}

int coli_v4_dspark_runner_draft(ColiV4DSparkRunner *runner,
                                const float *main_x, int anchor_token,
                                int position, int *draft_tokens,
                                float *draft_logits,
                                char *error, size_t error_size) {
    if (!runner || !main_x || !draft_tokens || position < 0 ||
        anchor_token < 0 || anchor_token >= runner->config.vocab_size)
        return runner_error(error, error_size, "invalid DSpark draft");
    int batch = runner->manifest.block_size;
    int d = runner->config.hidden_size, hc = runner->config.hc_mult;
    size_t hd = (size_t)hc * d;
    float *state = calloc((size_t)batch * hd, sizeof(*state));
    float *next = calloc((size_t)batch * hd, sizeof(*next));
    float *hidden = malloc((size_t)batch * d * sizeof(*hidden));
    int *input_tokens = malloc((size_t)batch * sizeof(*input_tokens));
    if (!state || !next || !hidden || !input_tokens) {
        free(input_tokens); free(hidden); free(next); free(state);
        return runner_error(error, error_size, "out of memory in DSpark draft");
    }
    for (int item = 0; item < batch; item++) {
        input_tokens[item] = item ? runner->manifest.noise_token_id : anchor_token;
        if (runner_embedding(runner, state + (size_t)item * hd,
                             input_tokens[item])) {
            free(input_tokens); free(hidden); free(next); free(state);
            return runner_error(error, error_size, "DSpark embedding read failed");
        }
    }
    for (int stage = 0; stage < runner->manifest.stage_count; stage++) {
        ColiDeepSeekV4LayerWeights weights;
        if (coli_v4_dspark_layer_load(runner->engine, &weights, &runner->config,
                                      runner->dspark_index, stage,
                                      error, error_size)) {
            free(input_tokens); free(hidden); free(next); free(state); return -1;
        }
        int result = coli_v4_dspark_attention_precompute_context(
            runner->attention[stage], &weights, &runner->config,
            main_x, position, 1, error, error_size);
        if (!result) result = coli_v4_dspark_block(
            next, runner->attention[stage], &weights, &runner->config,
            runner->experts, state, input_tokens, position + 1, batch,
            error, error_size);
        coli_v4_dspark_layer_release(runner->engine, &weights);
        if (result) {
            free(input_tokens); free(hidden); free(next); free(state); return -1;
        }
        float *swap = state; state = next; next = swap;
    }
    int result = coli_v4_dspark_final_hidden(runner->final, hidden, state, batch);
    ColiV4Engine *engine = runner->engine;
    int limit = batch;
    if (engine) {
        int requested = engine->runtime.verify_drafts;
        if (requested >= 1 && requested < limit) limit = requested;
        engine->dspark_verify.available = batch;
        engine->dspark_verify.limit = limit;
        engine->dspark_verify.head_slot = 0;
    }
    if (!result && limit > 0) {
        int *tokens_out = engine ? engine->dspark_verify.tokens : draft_tokens;
        float *logits_out = engine ? engine->dspark_verify.logits : draft_logits;
        float local_logits[64];
        if (!logits_out) logits_out = local_logits;
        result = coli_v4_dspark_biased_argmax_batch(
            engine, runner->heads, runner->target_index, hidden, anchor_token,
            tokens_out, logits_out, limit);
        for (int item = 0; !result && item < limit; item++) {
            draft_tokens[item] = tokens_out[item];
            if (draft_logits) draft_logits[item] = logits_out[item];
        }
    }
    int previous = limit > 0 ? draft_tokens[limit - 1] : anchor_token;
    for (int item = limit; !result && item < batch; item++) {
        draft_tokens[item] = previous;
        if (draft_logits) draft_logits[item] = 0.0f;
    }
    free(input_tokens); free(hidden); free(next); free(state);
    return result ? runner_error(error, error_size, "DSpark draft head failed") : 0;
}

int coli_v4_dspark_runner_block_size(const ColiV4DSparkRunner *runner) {
    return runner ? runner->manifest.block_size : 0;
}

int coli_v4_dspark_runner_verify_block_size(const ColiV4DSparkRunner *runner) {
    if (!runner) return 0;
    int available = runner->manifest.block_size;
    ColiV4Engine *engine = runner->engine;
    int requested = engine ? engine->runtime.verify_drafts : 0;
    if (requested < 1) return available;
    return requested < available ? requested : available;
}

uint64_t coli_v4_dspark_runner_loaded_stage_peak(
    const ColiV4DSparkRunner *runner) {
    return runner ? runner->loaded_stage_peak : 0;
}
/* ---- end inlined deepseek_v4_dspark_runner.c ---- */

#undef coli_v4_dspark_runner_close
#undef coli_v4_dspark_runner_open

#include "deepseek_v4_dspark.h"

int coli_v4_dspark_runner_open(ColiV4DSparkRunner **output,
                               ColiV4Engine *engine,
                               const char *dspark_model_dir,
                               const char *target_model_dir,
                               const ColiDeepSeekV4Config *config,
                               uint64_t expert_cache_bytes,
                               char *error, size_t error_size) {
    uint64_t cache = (engine && engine->runtime.dspark_expert_cache_bytes)
        ? engine->runtime.dspark_expert_cache_bytes : expert_cache_bytes;
    int rc = coli_v4_dspark_runner_owned_open(
        output, dspark_model_dir, target_model_dir, config,
        cache, error, error_size);
    if (!rc && output && *output) (*output)->engine = engine;
    return rc;
}

int coli_v4_dspark_runner_use_shared_heads(ColiV4DSparkRunner *runner,
                                           ColiV4DSparkHeads *heads) {
    if (!runner || !heads) return -1;
    coli_v4_dspark_heads_close(runner->heads);
    runner->heads = heads;
    return 0;
}

void coli_v4_dspark_runner_close(ColiV4DSparkRunner *runner) {
    if (runner && runner->engine &&
        runner->heads == runner->engine->dspark_capture.heads)
        runner->heads = NULL;
    coli_v4_dspark_runner_owned_close(runner);
}
#endif /* COLI_V4_UNIT_DSPARK_RUNNER */

#ifdef COLI_V4_UNIT_TARGET_HEAD_BATCH
/* ######## deepseek_v4_target_head_batch.c ######## */
#include "deepseek_v4_dspark.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "native_quant.h"
#include "tensor_io.h"

int coli_v4_target_load_embeddings(float *states_hc,
                                    const ColiSafetensorsIndex *index,
                                    const ColiDeepSeekV4Config *config,
                                    const int *tokens, int batch) {
    const ColiSafetensorsTensor *embed = coli_st_find(index, "embed.weight");
    if (!states_hc || !index || !config || !tokens || batch < 1 || batch > 64 ||
        !embed || embed->dtype != COLI_ST_BF16) return -1;
    int d = config->hidden_size, hc = config->hc_mult;
    uint16_t *raw = malloc((size_t)d * sizeof(*raw));
    if (!raw) return -1;
    for (int item = 0; item < batch; item++) {
        if (tokens[item] < 0 || tokens[item] >= config->vocab_size ||
            coli_st_read_at(index, embed->shard,
                            embed->offset + (uint64_t)tokens[item] * d * 2,
                            (size_t)d * sizeof(*raw), raw)) {
            free(raw); return -1;
        }
        float *state = states_hc + (size_t)item * hc * d;
        for (int copy = 0; copy < hc; copy++)
            for (int i = 0; i < d; i++)
                state[(size_t)copy * d + i] = coli_bf16_decode(raw[i]);
    }
    free(raw); return 0;
}

int coli_v4_target_head_argmax_batch(
    const float *states_hc, const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, int batch,
    int *tokens, float *logits, char *error, size_t error_size) {
    if (!states_hc || !index || !config || batch < 1 || batch > 64 || !tokens)
        return -1;
    ColiFloatTensor function = {0}, base = {0}, scale = {0}, norm = {0};
    if (coli_tensor_load_f32(&function, index, "hc_head_fn", error, error_size) ||
        coli_tensor_load_f32(&base, index, "hc_head_base", error, error_size) ||
        coli_tensor_load_f32(&scale, index, "hc_head_scale", error, error_size) ||
        coli_tensor_load_f32(&norm, index, "norm.weight", error, error_size))
        return -1;
    int d = config->hidden_size, hc = config->hc_mult, flat = hc * d;
    float *hidden = malloc((size_t)batch * d * sizeof(*hidden));
    if (!hidden || hc > 16) { free(hidden); return -1; }
    for (int item = 0; item < batch; item++) {
        const float *state = states_hc + (size_t)item * flat;
        float *out = hidden + (size_t)item * d;
        float square = 0.0f, pre[16];
        for (int i = 0; i < flat; i++) square += state[i] * state[i];
        float inverse = 1.0f / sqrtf(square / flat + config->rms_norm_eps);
        for (int copy = 0; copy < hc; copy++) {
            float mix = 0.0f;
            for (int i = 0; i < flat; i++)
                mix += function.data[(size_t)copy * flat + i] * state[i];
            float z = mix * inverse * scale.data[0] + base.data[copy];
            float sigmoid = z >= 0.0f ? 1.0f / (1.0f + expf(-z))
                                      : expf(z) / (1.0f + expf(z));
            pre[copy] = sigmoid + config->hc_eps;
        }
        for (int i = 0; i < d; i++) {
            float value = 0.0f;
            for (int copy = 0; copy < hc; copy++)
                value += pre[copy] * state[(size_t)copy * d + i];
            out[i] = coli_bf16_round(value);
        }
        if (coli_v4_rmsnorm(out, out, norm.data, d, config->rms_norm_eps))
            return -1;
        coli_bf16_round_array(out, (size_t)d);
        tokens[item] = -1; if (logits) logits[item] = -FLT_MAX;
    }
    const ColiSafetensorsTensor *head = coli_st_find(index, "head.weight");
#ifndef COLI_V4_DISABLE_ZERO_COPY_HEAD
    const uint16_t *resident = head ? coli_v4_head_cache_data(
        NULL, head->shard, head->offset, (size_t)head->nbytes) : NULL;
    if (resident) {
        float *resident_scores = malloc(
            (size_t)batch * config->vocab_size * sizeof(*resident_scores));
        if (!resident_scores) return -1;
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < config->vocab_size; row++) {
            float sums[64] = {0};
            const uint16_t *weight = resident + (size_t)row * d;
            for (int i = 0; i < d; i++) {
                float decoded = coli_bf16_decode(weight[i]);
                for (int item = 0; item < batch; item++)
                    sums[item] += decoded * hidden[(size_t)item * d + i];
            }
            for (int item = 0; item < batch; item++)
                resident_scores[(size_t)item * config->vocab_size + row] =
                    sums[item];
        }
        for (int item = 0; item < batch; item++)
            for (int row = 0; row < config->vocab_size; row++) {
                float score = resident_scores[
                    (size_t)item * config->vocab_size + row];
                float current = logits ? logits[item] :
                    (tokens[item] < 0 ? -FLT_MAX : 0.0f);
                if (tokens[item] < 0 || score > current) {
                    tokens[item] = row;
                    if (logits) logits[item] = score;
                }
            }
        free(resident_scores);
        goto head_complete;
    }
#endif
    enum { ROWS = 32 };
    uint16_t *raw = malloc((size_t)ROWS * d * sizeof(*raw));
    float *scores = malloc((size_t)batch * ROWS * sizeof(*scores));
    if (!head || head->dtype != COLI_ST_BF16 || !raw || !scores) return -1;
    for (int start = 0; start < config->vocab_size; start += ROWS) {
        int rows = config->vocab_size - start < ROWS
            ? config->vocab_size - start : ROWS;
        if (coli_st_read_at(index, head->shard,
                            head->offset + (uint64_t)start * d * 2,
                            (size_t)rows * d * sizeof(*raw), raw)) return -1;
        #pragma omp parallel for collapse(2) schedule(static)
        for (int item = 0; item < batch; item++) for (int row = 0; row < rows; row++) {
            float sum = 0.0f; const uint16_t *weight = raw + (size_t)row * d;
            const float *input = hidden + (size_t)item * d;
            for (int i = 0; i < d; i++) sum += coli_bf16_decode(weight[i]) * input[i];
            scores[(size_t)item * ROWS + row] = sum;
        }
        for (int item = 0; item < batch; item++) for (int row = 0; row < rows; row++) {
            float score = scores[(size_t)item * ROWS + row];
            float current = logits ? logits[item] :
                (tokens[item] < 0 ? -FLT_MAX : 0.0f);
            if (tokens[item] < 0 || score > current) {
                tokens[item] = start + row;
                if (logits) logits[item] = score;
            }
        }
    }
    free(scores); free(raw);
#ifndef COLI_V4_DISABLE_ZERO_COPY_HEAD
head_complete:
#endif
    free(hidden);
    coli_float_tensor_free(&norm); coli_float_tensor_free(&scale);
    coli_float_tensor_free(&base); coli_float_tensor_free(&function);
    return 0;
}
#endif /* COLI_V4_UNIT_TARGET_HEAD_BATCH */

#ifdef COLI_V4_UNIT_SPECULATIVE
/* ######## deepseek_v4_speculative.c ######## */
#include "deepseek_v4_dspark.h"

#include <string.h>

void coli_v4_speculative_controller_init(ColiV4SpeculativeController *state,
                                         uint64_t minimum_proposals,
                                         float disable_threshold) {
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->minimum_proposals = minimum_proposals ? minimum_proposals : 32;
    state->disable_threshold = disable_threshold > 0.0f &&
                               disable_threshold <= 1.0f
        ? disable_threshold : 0.35f;
    state->enabled = 1;
}

float coli_v4_speculative_acceptance(const ColiV4SpeculativeController *state) {
    return state && state->proposed
        ? (float)state->accepted / (float)state->proposed : 0.0f;
}

void coli_v4_speculative_record(ColiV4SpeculativeController *state,
                                int proposed, int accepted) {
    if (!state || !state->enabled || proposed < 1 || accepted < 0 ||
        accepted > proposed) return;
    state->rounds++;
    state->proposed += (uint64_t)proposed;
    state->accepted += (uint64_t)accepted;
    if (state->proposed >= state->minimum_proposals &&
        coli_v4_speculative_acceptance(state) < state->disable_threshold)
        state->enabled = 0;
}

int coli_v4_verify_greedy(ColiV4VerificationResult *result,
                          int *output_tokens, int output_capacity,
                          const int *draft_tokens, int draft_count,
                          const int *target_tokens, int target_count) {
    if (!result || !output_tokens || !draft_tokens || !target_tokens ||
        draft_count < 1 || target_count != draft_count + 1 ||
        output_capacity < draft_count + 1) return -1;
    int accepted = 0;
    while (accepted < draft_count &&
           draft_tokens[accepted] == target_tokens[accepted]) {
        output_tokens[accepted] = draft_tokens[accepted];
        accepted++;
    }
    output_tokens[accepted] = target_tokens[accepted];
    result->accepted_draft_tokens = accepted;
    result->output_count = accepted + 1;
    result->mismatch_index = accepted < draft_count ? accepted : -1;
    return 0;
}
#endif /* COLI_V4_UNIT_SPECULATIVE */

#ifdef COLI_V4_UNIT_DSPARK_HEADS
/* ######## deepseek_v4_dspark_heads.c ######## */
/* ---- begin inlined deepseek_v4_dspark_heads_v2.c ---- */
/* ---- begin inlined deepseek_v4_dspark_heads.c ---- */
#include "deepseek_v4_dspark.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "native_quant.h"
#include "safetensors_index.h"

struct ColiV4DSparkHeads {
    int hidden, hc, targets, vocab, rank;
    float rms_eps;
    ColiTensorView main_proj;
    unsigned char *main_weight;
    unsigned char *main_scale;
    float *main_norm;
    uint16_t *markov_w1;
    uint16_t *markov_w2;
};

static int heads_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments; va_start(arguments, format);
        vsnprintf(error, size, format, arguments); va_end(arguments);
    }
    return -1;
}

static void *read_named(ColiSafetensorsIndex *index, const char *name,
                        ColiSafetensorsDType dtype, char *error, size_t size) {
    const ColiSafetensorsTensor *tensor = coli_st_find(index, name);
    if (!tensor || tensor->dtype != dtype) {
        heads_error(error, size, "invalid DSpark head tensor: %s", name);
        return NULL;
    }
    void *data = malloc((size_t)tensor->nbytes);
    if (!data || coli_st_read_tensor(index, tensor, data)) {
        free(data); heads_error(error, size, "cannot read DSpark head: %s", name);
        return NULL;
    }
    return data;
}

void coli_v4_dspark_heads_close(ColiV4DSparkHeads *heads) {
    if (!heads) return;
    free(heads->markov_w2); free(heads->markov_w1); free(heads->main_norm);
    free(heads->main_scale); free(heads->main_weight); free(heads);
}

int coli_v4_dspark_heads_open(ColiV4DSparkHeads **output,
                              const char *model_dir,
                              const ColiDeepSeekV4Config *config,
                              const ColiDeepSeekV4DSparkManifest *manifest,
                              char *error, size_t error_size) {
    if (!output || !model_dir || !config || !manifest ||
        manifest->stage_count < 1) return -1;
    *output = NULL; ColiSafetensorsIndex *index = NULL;
    if (coli_st_index_open(&index, model_dir, error, error_size)) return -1;
    ColiV4DSparkHeads *heads = calloc(1, sizeof(*heads));
    if (!heads) { coli_st_index_close(index); return -1; }
    heads->hidden = config->hidden_size; heads->hc = config->hc_mult;
    heads->targets = manifest->target_count; heads->vocab = config->vocab_size;
    heads->rank = manifest->markov_rank; heads->rms_eps = config->rms_norm_eps;
    heads->main_weight = read_named(index, "mtp.0.main_proj.weight",
                                    COLI_ST_F8_E4M3, error, error_size);
    heads->main_scale = read_named(index, "mtp.0.main_proj.scale",
                                   COLI_ST_F8_E8M0, error, error_size);
    uint16_t *norm_bf16 = read_named(index, "mtp.0.main_norm.weight",
                                     COLI_ST_BF16, error, error_size);
    char name[160]; int last = manifest->stage_count - 1;
    snprintf(name, sizeof(name), "mtp.%d.markov_head.markov_w1.weight", last);
    heads->markov_w1 = read_named(index, name, COLI_ST_BF16, error, error_size);
    snprintf(name, sizeof(name), "mtp.%d.markov_head.markov_w2.weight", last);
    heads->markov_w2 = read_named(index, name, COLI_ST_BF16, error, error_size);
    coli_st_index_close(index);
    heads->main_norm = malloc((size_t)heads->hidden * sizeof(float));
    if (!heads->main_weight || !heads->main_scale || !norm_bf16 ||
        !heads->markov_w1 || !heads->markov_w2 || !heads->main_norm) {
        free(norm_bf16); coli_v4_dspark_heads_close(heads); return -1;
    }
    for (int i = 0; i < heads->hidden; i++)
        heads->main_norm[i] = coli_bf16_decode(norm_bf16[i]);
    free(norm_bf16);
    size_t columns = (size_t)heads->hidden * heads->targets;
    heads->main_proj = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_UE8M0,
        heads->main_weight, heads->main_scale,
        (size_t)heads->hidden * columns,
        (size_t)((heads->hidden + 127) / 128) * ((columns + 127) / 128),
        heads->hidden, (int64_t)columns, 128, 128
    };
    *output = heads; return 0;
}

int coli_v4_dspark_combine_hidden(ColiV4DSparkHeads *heads, float *output,
                                  const float *target_hidden_hc) {
    if (!heads || !output || !target_hidden_hc) return -1;
    size_t columns = (size_t)heads->hidden * heads->targets;
    float *combined = malloc(columns * sizeof(*combined));
    float *projected = malloc((size_t)heads->hidden * sizeof(*projected));
    if (!combined || !projected) { free(projected); free(combined); return -1; }
    size_t hc_width = (size_t)heads->hc * heads->hidden;
    for (int target = 0; target < heads->targets; target++)
        for (int column = 0; column < heads->hidden; column++) {
            float sum = 0.0f;
            for (int copy = 0; copy < heads->hc; copy++)
                sum += target_hidden_hc[(size_t)target * hc_width +
                                        (size_t)copy * heads->hidden + column];
            combined[(size_t)target * heads->hidden + column] =
                sum / (float)heads->hc;
        }
    int result = coli_fp8_matvec_ref(projected, &heads->main_proj, combined);
    if (!result) {
        coli_bf16_round_array(projected, (size_t)heads->hidden);
        result = coli_v4_rmsnorm(output, projected, heads->main_norm,
                                 heads->hidden, heads->rms_eps);
        if (!result) coli_bf16_round_array(output, (size_t)heads->hidden);
    }
    free(projected); free(combined); return result;
}

int coli_v4_dspark_markov_bias(ColiV4DSparkHeads *heads, float *bias,
                               int previous_token) {
    if (!heads || !bias || previous_token < 0 || previous_token >= heads->vocab)
        return -1;
    float *embedding = malloc((size_t)heads->rank * sizeof(*embedding));
    if (!embedding) return -1;
    const uint16_t *row = heads->markov_w1 + (size_t)previous_token * heads->rank;
    for (int i = 0; i < heads->rank; i++) embedding[i] = coli_bf16_decode(row[i]);
    #pragma omp parallel for schedule(static)
    for (int64_t token = 0; token < heads->vocab; token++) {
        const uint16_t *weight = heads->markov_w2 + (size_t)token * heads->rank;
        float sum = 0.0f;
        for (int i = 0; i < heads->rank; i++)
            sum += embedding[i] * coli_bf16_decode(weight[i]);
        bias[token] = sum;
    }
    free(embedding); return 0;
}
/* ---- end inlined deepseek_v4_dspark_heads.c ---- */


#include <float.h>



int coli_v4_dspark_biased_argmax(
    ColiV4Engine *engine, ColiV4DSparkHeads *heads,
    const ColiSafetensorsIndex *target_index, const float *hidden,
    int previous_token, int *best_token, float *best_logit) {
    if (!heads || !target_index || !hidden || !best_token || !best_logit ||
        previous_token < 0 || previous_token >= heads->vocab) return -1;
    const ColiSafetensorsTensor *head = coli_st_find(target_index, "head.weight");
    if (!head || head->dtype != COLI_ST_BF16 || head->rank != 2 ||
        head->shape[0] != heads->vocab || head->shape[1] != heads->hidden)
        return -1;
    enum { ROWS = 32 };
    uint16_t *raw = malloc((size_t)ROWS * heads->hidden * sizeof(*raw));
    float *scores = malloc((size_t)ROWS * sizeof(*scores));
    float *markov = malloc((size_t)heads->rank * sizeof(*markov));
    if (!raw || !scores || !markov) {
        free(markov); free(scores); free(raw); return -1;
    }
    const uint16_t *m1 = heads->markov_w1 +
                         (size_t)previous_token * heads->rank;
    for (int i = 0; i < heads->rank; i++) markov[i] = coli_bf16_decode(m1[i]);
    int winner = -1; float maximum = -FLT_MAX;
    for (int start = 0; start < heads->vocab; start += ROWS) {
        int rows = heads->vocab - start < ROWS ? heads->vocab - start : ROWS;
        size_t bytes = (size_t)rows * heads->hidden * sizeof(*raw);
        if (coli_st_read_at_engine(
                engine, target_index, head->shard,
                head->offset + (uint64_t)start * heads->hidden * 2,
                bytes, raw)) {
            free(markov); free(scores); free(raw); return -1;
        }
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < rows; row++) {
            float sum = 0.0f;
            const uint16_t *weight = raw + (size_t)row * heads->hidden;
            const uint16_t *m2 = heads->markov_w2 +
                                 (size_t)(start + row) * heads->rank;
            for (int i = 0; i < heads->hidden; i++)
                sum += coli_bf16_decode(weight[i]) * hidden[i];
            for (int i = 0; i < heads->rank; i++)
                sum += coli_bf16_decode(m2[i]) * markov[i];
            scores[row] = sum;
        }
        for (int row = 0; row < rows; row++)
            if (scores[row] > maximum) {
                maximum = scores[row]; winner = start + row;
            }
    }
    free(markov); free(scores); free(raw);
    *best_token = winner; *best_logit = maximum;
    return winner < 0 ? -1 : 0;
}
/* ---- end inlined deepseek_v4_dspark_heads_v2.c ---- */




int coli_v4_dspark_biased_argmax_batch(
    ColiV4Engine *engine, ColiV4DSparkHeads *heads,
    const ColiSafetensorsIndex *target_index, const float *hidden_batch,
    int previous_token, int *best_tokens, float *best_logits, int batch) {
    if (!heads || !target_index || !hidden_batch || !best_tokens ||
        !best_logits || batch < 1 || batch > 64 || previous_token < 0 ||
        previous_token >= heads->vocab) return -1;
    const ColiSafetensorsTensor *head = coli_st_find(target_index, "head.weight");
    if (!head || head->dtype != COLI_ST_BF16 || head->rank != 2 ||
        head->shape[0] != heads->vocab || head->shape[1] != heads->hidden)
        return -1;

    enum { ROWS = 32 };
    uint16_t *raw = malloc((size_t)ROWS * heads->hidden * sizeof(*raw));
    float *base = malloc((size_t)batch * heads->vocab * sizeof(*base));
    float *markov = malloc((size_t)heads->rank * sizeof(*markov));
    if (!raw || !base || !markov) {
        free(markov); free(base); free(raw); return -1;
    }

    /* Each sum still visits columns in scalar order; BF16 decode is shared. */
    for (int start = 0; start < heads->vocab; start += ROWS) {
        int rows = heads->vocab - start < ROWS ? heads->vocab - start : ROWS;
        size_t bytes = (size_t)rows * heads->hidden * sizeof(*raw);
        if (coli_st_read_at_engine(
                engine, target_index, head->shard,
                head->offset + (uint64_t)start * heads->hidden * 2,
                bytes, raw)) {
            free(markov); free(base); free(raw); return -1;
        }
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < rows; row++) {
            const uint16_t *weight = raw + (size_t)row * heads->hidden;
            float sums[64] = {0};
            for (int i = 0; i < heads->hidden; i++) {
                float decoded = coli_bf16_decode(weight[i]);
                for (int item = 0; item < batch; item++)
                    sums[item] += decoded *
                        hidden_batch[(size_t)item * heads->hidden + i];
            }
            for (int item = 0; item < batch; item++)
                base[(size_t)item * heads->vocab + start + row] = sums[item];
        }
    }

    for (int item = 0; item < batch; item++) {
        const uint16_t *m1 = heads->markov_w1 +
                             (size_t)previous_token * heads->rank;
        for (int i = 0; i < heads->rank; i++)
            markov[i] = coli_bf16_decode(m1[i]);
        float *scores = base + (size_t)item * heads->vocab;
        #pragma omp parallel for schedule(static)
        for (int token = 0; token < heads->vocab; token++) {
            const uint16_t *m2 = heads->markov_w2 +
                                 (size_t)token * heads->rank;
            float sum = scores[token];
            for (int i = 0; i < heads->rank; i++)
                sum += coli_bf16_decode(m2[i]) * markov[i];
            scores[token] = sum;
        }
        int winner = 0;
        for (int token = 1; token < heads->vocab; token++)
            if (scores[token] > scores[winner]) winner = token;
        best_tokens[item] = winner;
        best_logits[item] = scores[winner];
        previous_token = winner;
    }
    free(markov); free(base); free(raw);
    return 0;
}
#endif /* COLI_V4_UNIT_DSPARK_HEADS */

#ifdef COLI_V4_UNIT_DSPARK_CAPTURE
/* ######## deepseek_v4_dspark_capture.c ######## */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_dspark.h"

static int capture_init(ColiV4Engine *engine, const ColiDeepSeekV4Config *config) {
    if (!engine) return -1;
    if (engine->dspark_capture.heads) return 0;
    ColiDeepSeekV4RuntimeOptions *runtime = &engine->runtime;
    const char *model = runtime->dspark_model_dir;
    char error[512];
    if (!model ||
        coli_v4_dspark_inspect(model, config, &engine->dspark_capture.manifest,
                               error, sizeof(error)) ||
        coli_v4_dspark_heads_open(&engine->dspark_capture.heads, model, config,
                                  &engine->dspark_capture.manifest, error,
                                  sizeof(error))) {
        fprintf(stderr, "dspark capture disabled: %s\n", model ? error :
                "DSpark model path is unset");
        return -1;
    }
    return 0;
}

static int target_ordinal(const ColiV4Engine *engine, int layer) {
    for (int i = 0; i < engine->dspark_capture.manifest.target_count; i++)
        if (engine->dspark_capture.manifest.target_layer_ids[i] == layer)
            return i;
    return -1;
}

static void capture_publish(ColiV4Engine *engine,
                            const ColiDeepSeekV4Config *config, int batch) {
    size_t token_size = (size_t)config->hc_mult * config->hidden_size;
    int targets = engine->dspark_capture.manifest.target_count;
    float *joined = malloc((size_t)targets * token_size * sizeof(float));
    float *main_x = malloc((size_t)config->hidden_size * sizeof(float));
    if (!joined || !main_x) { free(main_x); free(joined); return; }
    for (int item = 0; item < batch; item++) {
        for (int target = 0; target < targets; target++)
            memcpy(joined + (size_t)target * token_size,
                   engine->dspark_capture.states[target] +
                       (size_t)item * token_size,
                   token_size * sizeof(float));
        if (coli_v4_dspark_combine_hidden(engine->dspark_capture.heads, main_x,
                                          joined))
            break;
        double checksum = 0.0;
        for (int i = 0; i < config->hidden_size; i++)
            checksum += fabs((double)main_x[i]);
        if (item == batch - 1)
            fprintf(stderr, "dspark_main_x batch=%d first=%.9g l1=%.9g\n",
                    batch, main_x[0], checksum);
    }
    free(main_x); free(joined);
}

void coli_v4_dspark_capture_after_block(
    ColiV4Engine *engine, const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *outputs, int batch) {
    if (!engine || !weights || !config || !outputs || batch < 1) return;
    if (capture_init(engine, config)) return;
    int ordinal = target_ordinal(engine, weights->plan.layer);
    if (ordinal < 0) return;
    size_t count = (size_t)batch * config->hc_mult * config->hidden_size;
    if (count > engine->dspark_capture.capacity) {
        for (int i = 0; i < engine->dspark_capture.manifest.target_count; i++) {
            float *allocation = realloc(engine->dspark_capture.states[i],
                                        count * sizeof(float));
            if (!allocation) return;
            engine->dspark_capture.states[i] = allocation;
        }
        engine->dspark_capture.capacity = count;
    }
    memcpy(engine->dspark_capture.states[ordinal], outputs,
           count * sizeof(float));
    if (ordinal == engine->dspark_capture.manifest.target_count - 1)
        capture_publish(engine, config, batch);
}

static int capture_main_x_from_states(ColiV4Engine *engine, float *outputs,
                                      int batch,
                                      const ColiDeepSeekV4Config *config) {
    if (!outputs || !config || batch < 1 || !engine ||
        capture_init(engine, config) ||
        engine->dspark_capture.capacity <
            (size_t)batch * config->hc_mult * config->hidden_size)
        return -1;
    size_t token_size = (size_t)config->hc_mult * config->hidden_size;
    int targets = engine->dspark_capture.manifest.target_count;
    float *joined = malloc((size_t)targets * token_size * sizeof(*joined));
    if (!joined) return -1;
    int result = 0;
    for (int item = 0; !result && item < batch; item++) {
        for (int target = 0; target < targets; target++)
            memcpy(joined + (size_t)target * token_size,
                   engine->dspark_capture.states[target] +
                       (size_t)item * token_size,
                   token_size * sizeof(*joined));
        result = coli_v4_dspark_combine_hidden(
            engine->dspark_capture.heads,
            outputs + (size_t)item * config->hidden_size, joined);
    }
    free(joined); return result;
}

ColiV4DSparkHeads *coli_v4_dspark_capture_heads(ColiV4Engine *engine) {
    return engine ? engine->dspark_capture.heads : NULL;
}

int coli_v4_dspark_capture_stage_main_x(ColiV4Engine *engine,
                                        const float *values, int batch,
                                        int hidden_size) {
    if (!engine || !values || batch < 1 || batch > 64 || hidden_size < 1)
        return -1;
    size_t count = (size_t)batch * hidden_size;
    float *copy = realloc(engine->dspark_capture.staged_main_x,
                          count * sizeof(*copy));
    if (!copy) return -1;
    engine->dspark_capture.staged_main_x = copy;
    memcpy(engine->dspark_capture.staged_main_x, values,
           count * sizeof(*copy));
    engine->dspark_capture.staged_batch = batch;
    engine->dspark_capture.staged_hidden = hidden_size;
    return 0;
}

int coli_v4_dspark_capture_main_x(ColiV4Engine *engine, float *outputs,
                                  int batch,
                                  const ColiDeepSeekV4Config *config) {
    if (engine && engine->dspark_capture.staged_batch) {
        if (!outputs || !config || batch < 1 ||
            batch > engine->dspark_capture.staged_batch ||
            config->hidden_size != engine->dspark_capture.staged_hidden)
            return -1;
        memcpy(outputs, engine->dspark_capture.staged_main_x,
               (size_t)batch * engine->dspark_capture.staged_hidden *
                   sizeof(*outputs));
        engine->dspark_capture.staged_batch = 0;
        return 0;
    }
    return capture_main_x_from_states(engine, outputs, batch, config);
}
#endif /* COLI_V4_UNIT_DSPARK_CAPTURE */


#ifdef COLI_V4_UNIT_TARGET_VERIFY
/* ######## deepseek_v4_target_verify.c ######## */
#define coli_v4_target_verify_greedy_batch \
    coli_v4_target_verify_greedy_batch_v2_fallback
/* ---- begin inlined deepseek_v4_target_verify_v2.c ---- */
#include "deepseek_v4_dspark.h"

#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_dspark.h"
#include "deepseek_v4_dspark.h"

static int run_target_block_record(
    ColiV4Engine *engine, float **state_ptr, float **next_ptr,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const int *tokens, int start, int batch, float *layer_inputs,
    char *error, size_t error_size) {
    float *state = *state_ptr, *next = *next_ptr;
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    for (int layer_id = 0; layer_id < config->num_hidden_layers; layer_id++) {
        if (layer_inputs)
            memcpy(layer_inputs + (size_t)layer_id * batch * hd, state,
                   (size_t)batch * hd * sizeof(float));
        ColiDeepSeekV4LayerWeights layer;
        if (coli_v4_layer_load(engine, &layer, config, index, layer_id,
                               error, error_size)) return -1;
        int result = coli_v4_block_window_batch_ref(
            next, attention[layer_id], &layer, config, experts,
            state, tokens, start, batch, error, error_size);
        if (!result)
            coli_v4_dspark_capture_after_block(engine, &layer, config, next,
                                               batch);
        coli_v4_layer_free(engine, &layer);
        if (result) return -1;
        float *swap = state; state = next; next = swap;
    }
    *state_ptr = state; *next_ptr = next; return 0;
}

static int commit_target_attention(
    ColiV4Engine *engine,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config,
    const float *layer_inputs, int recorded_batch,
    int start, int commit_batch, char *error, size_t error_size) {
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    for (int layer_id = 0; layer_id < config->num_hidden_layers; layer_id++) {
        ColiDeepSeekV4LayerWeights layer;
        if (coli_v4_layer_load(engine, &layer, config, index, layer_id,
                               error, error_size)) return -1;
        int result = coli_v4_target_attention_commit_batch(
            attention[layer_id], &layer, config,
            layer_inputs + (size_t)layer_id * recorded_batch * hd,
            start, commit_batch, error, error_size);
        coli_v4_layer_free(engine, &layer);
        if (result) return -1;
    }
    return 0;
}

int coli_v4_target_verify_greedy_batch(
    ColiV4VerificationResult *verification,
    int *output_tokens, int output_capacity,
    ColiV4Engine *engine,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    int anchor_token, const int *draft_tokens, int draft_count,
    int start_position, char *error, size_t error_size) {
    if (!verification || !output_tokens || !attention || !index || !config ||
        !experts || !draft_tokens || draft_count < 1 || draft_count > 63 ||
        output_capacity < draft_count + 1) return -1;
    int batch = draft_count + 1;
    int *inputs = malloc((size_t)batch * sizeof(*inputs));
    int *targets = malloc((size_t)batch * sizeof(*targets));
    float *logits = malloc((size_t)batch * sizeof(*logits));
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    float *state = malloc((size_t)batch * hd * sizeof(*state));
    float *next = malloc((size_t)batch * hd * sizeof(*next));
    float *layer_inputs = malloc((size_t)config->num_hidden_layers *
                                 batch * hd * sizeof(*layer_inputs));
    ColiV4AttentionSnapshot **snapshots = calloc(
        (size_t)config->num_hidden_layers, sizeof(*snapshots));
    if (!inputs || !targets || !logits || !state || !next || !layer_inputs ||
        !snapshots) return -1;
    inputs[0] = anchor_token;
    for (int i = 0; i < draft_count; i++) inputs[i + 1] = draft_tokens[i];
    for (int layer = 0; layer < config->num_hidden_layers; layer++)
        if (coli_v4_attention_snapshot_create(attention[layer],
                                               &snapshots[layer])) return -1;
    if (coli_v4_target_load_embeddings(state, index, config, inputs, batch) ||
        run_target_block_record(engine, &state, &next, attention, index, config,
                                experts, inputs, start_position, batch,
                                layer_inputs, error, error_size) ||
        coli_v4_target_head_argmax_batch(state, index, config, batch,
                                         targets, logits, error, error_size) ||
        coli_v4_verify_greedy(verification, output_tokens, output_capacity,
                              draft_tokens, draft_count, targets, batch))
        return -1;
    if (verification->accepted_draft_tokens < draft_count) {
        for (int layer = 0; layer < config->num_hidden_layers; layer++)
            if (coli_v4_attention_snapshot_restore(attention[layer],
                                                   snapshots[layer])) return -1;
        int commit = verification->accepted_draft_tokens + 1;
        if (commit_target_attention(engine, attention, index, config,
                                    layer_inputs, batch, start_position, commit,
                                    error, error_size)) return -1;
    }
    for (int layer = 0; layer < config->num_hidden_layers; layer++)
        coli_v4_attention_snapshot_destroy(snapshots[layer]);
    free(snapshots); free(layer_inputs); free(next); free(state);
    free(logits); free(targets); free(inputs); return 0;
}
/* ---- end inlined deepseek_v4_target_verify_v2.c ---- */

#undef coli_v4_target_verify_greedy_batch

#include "deepseek_v4_dspark.h"


static void destroy_snapshots(ColiV4AttentionSnapshot **snapshots, int layers) {
    if (!snapshots) return;
    for (int layer = 0; layer < layers; layer++)
        coli_v4_attention_snapshot_destroy(snapshots[layer]);
    free(snapshots);
}

static int create_snapshots(
    ColiV4AttentionSnapshot ***output,
    ColiDeepSeekV4WindowAttentionState **attention, int layers) {
    ColiV4AttentionSnapshot **snapshots = calloc(
        (size_t)layers, sizeof(*snapshots));
    if (!snapshots) return -1;
    for (int layer = 0; layer < layers; layer++)
        if (coli_v4_attention_snapshot_create(attention[layer],
                                               &snapshots[layer])) {
            destroy_snapshots(snapshots, layers); return -1;
        }
    *output = snapshots; return 0;
}

static int restore_snapshots(
    ColiDeepSeekV4WindowAttentionState **attention,
    ColiV4AttentionSnapshot **snapshots, int layers) {
    for (int layer = 0; layer < layers; layer++)
        if (coli_v4_attention_snapshot_restore(attention[layer],
                                               snapshots[layer])) return -1;
    return 0;
}

int coli_v4_target_verify_greedy_batch(
    ColiV4VerificationResult *verification,
    int *output_tokens, int output_capacity,
    ColiV4Engine *engine,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    int anchor_token, const int *draft_tokens, int draft_count,
    int start_position, char *error, size_t error_size) {
    if (draft_count != 4)
        return coli_v4_target_verify_greedy_batch_v2_fallback(
            verification, output_tokens, output_capacity, engine, attention,
            index, config, experts, anchor_token, draft_tokens, draft_count,
            start_position, error, error_size);
    if (!verification || !output_tokens || output_capacity < 5 || !attention ||
        !index || !config || !experts || !draft_tokens) return -1;

    enum { FIRST_BATCH = 3, TAIL_BATCH = 2 };
    int first_inputs[FIRST_BATCH] = {
        anchor_token, draft_tokens[0], draft_tokens[1]
    };
    int first_targets[FIRST_BATCH]; float first_logits[FIRST_BATCH];
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)d * hc;
    size_t layer_stride_first = (size_t)FIRST_BATCH * hd;
    float *state = malloc((size_t)FIRST_BATCH * hd * sizeof(*state));
    float *next = malloc((size_t)FIRST_BATCH * hd * sizeof(*next));
    float *layer_inputs = malloc((size_t)config->num_hidden_layers *
                                 layer_stride_first * sizeof(*layer_inputs));
    float *combined_main_x = malloc((FIRST_BATCH + TAIL_BATCH) *
                                    (size_t)d * sizeof(*combined_main_x));
    ColiV4AttentionSnapshot **snapshots = NULL;
    if (!state || !next || !layer_inputs || !combined_main_x ||
        create_snapshots(&snapshots, attention, config->num_hidden_layers))
        return -1;
    int result = coli_v4_target_load_embeddings(
                     state, index, config, first_inputs, FIRST_BATCH) ||
                 run_target_block_record(
                     engine, &state, &next, attention, index, config, experts,
                     first_inputs, start_position, FIRST_BATCH, layer_inputs,
                     error, error_size) ||
                 coli_v4_target_head_argmax_batch(
                     state, index, config, FIRST_BATCH,
                     first_targets, first_logits, error, error_size) ||
                 coli_v4_dspark_capture_main_x(engine, combined_main_x, FIRST_BATCH, config);
    if (result) goto cleanup;
    int accepted = 0;
    while (accepted < FIRST_BATCH &&
           draft_tokens[accepted] == first_targets[accepted]) accepted++;
    if (accepted < FIRST_BATCH) {
        for (int i = 0; i < accepted; i++) output_tokens[i] = draft_tokens[i];
        output_tokens[accepted] = first_targets[accepted];
        verification->accepted_draft_tokens = accepted;
        verification->output_count = accepted + 1;
        verification->mismatch_index = accepted;
        int commit = accepted + 1;
        if (commit < FIRST_BATCH &&
            (restore_snapshots(attention, snapshots,
                               config->num_hidden_layers) ||
             commit_target_attention(
                 engine, attention, index, config, layer_inputs, FIRST_BATCH,
                 start_position, commit, error, error_size))) result = -1;
        goto cleanup;
    }
    destroy_snapshots(snapshots, config->num_hidden_layers); snapshots = NULL;
    free(layer_inputs); layer_inputs = NULL;
    free(next); free(state); next = NULL; state = NULL;

    int tail_inputs[TAIL_BATCH] = {draft_tokens[2], draft_tokens[3]};
    int tail_targets[TAIL_BATCH]; float tail_logits[TAIL_BATCH];
    state = malloc((size_t)TAIL_BATCH * hd * sizeof(*state));
    next = malloc((size_t)TAIL_BATCH * hd * sizeof(*next));
    layer_inputs = malloc((size_t)config->num_hidden_layers * TAIL_BATCH *
                          hd * sizeof(*layer_inputs));
    if (!state || !next || !layer_inputs ||
        create_snapshots(&snapshots, attention, config->num_hidden_layers)) {
        result = -1; goto cleanup;
    }
    result = coli_v4_target_load_embeddings(
                 state, index, config, tail_inputs, TAIL_BATCH) ||
             run_target_block_record(
                 engine, &state, &next, attention, index, config, experts,
                 tail_inputs, start_position + FIRST_BATCH, TAIL_BATCH,
                 layer_inputs, error, error_size) ||
             coli_v4_target_head_argmax_batch(
                 state, index, config, TAIL_BATCH,
                 tail_targets, tail_logits, error, error_size) ||
             coli_v4_dspark_capture_main_x(engine, combined_main_x + (size_t)FIRST_BATCH * d,
                 TAIL_BATCH, config);
    if (result) goto cleanup;
    for (int i = 0; i < FIRST_BATCH; i++) output_tokens[i] = draft_tokens[i];
    if (draft_tokens[3] == tail_targets[0]) {
        output_tokens[3] = draft_tokens[3];
        output_tokens[4] = tail_targets[1];
        verification->accepted_draft_tokens = 4;
        verification->output_count = 5;
        verification->mismatch_index = -1;
        result = coli_v4_dspark_capture_stage_main_x(engine, combined_main_x, 5, d);
    } else {
        output_tokens[3] = tail_targets[0];
        verification->accepted_draft_tokens = 3;
        verification->output_count = 4;
        verification->mismatch_index = 3;
        if (restore_snapshots(attention, snapshots,
                              config->num_hidden_layers) ||
            commit_target_attention(
                engine, attention, index, config, layer_inputs, TAIL_BATCH,
                start_position + FIRST_BATCH, 1, error, error_size) ||
            coli_v4_dspark_capture_stage_main_x(engine, combined_main_x, 4, d))
            result = -1;
    }

cleanup:
    destroy_snapshots(snapshots, config->num_hidden_layers);
    free(combined_main_x); free(layer_inputs); free(next); free(state);
    return result;
}
#endif /* COLI_V4_UNIT_TARGET_VERIFY */

#ifdef COLI_V4_UNIT_TARGET_ATTENTION_COMMIT
/* ######## deepseek_v4_target_attention_commit.c ######## */
#include "deepseek_v4_internal.h"
#define coli_v4_block_token_batch_serial_ref \
    coli_v4_commit_block_token_serial_ref
#define coli_v4_block_window_token_batch_serial_ref \
    coli_v4_commit_block_window_token_serial_ref
/* ---- begin inlined deepseek_v4_target_attention_commit.c ---- */
#define coli_v4_block_window_batch_ref coli_v4_commit_unused_full_block
/* ---- begin include deepseek_v4_block_batch.c ---- */
#define coli_v4_block_token_ref coli_v4_block_token_batch_serial_ref
#define coli_v4_block_window_token_ref coli_v4_block_window_token_batch_serial_ref
/* ---- begin include deepseek_v4_block.c ---- */
#include "deepseek_v4_internal.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "native_quant.h"

static int set_error(char *error, size_t size, const char *format, ...) {
    if (error && size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(error, size, format, arguments);
        va_end(arguments);
    }
    return -1;
}

static const void *value(const ColiDeepSeekV4LayerWeights *weights,
                         const char *suffix,
                         const ColiDeepSeekV4TensorSpec **spec) {
    char name[COLI_V4_MAX_TENSOR_NAME];
    snprintf(name, sizeof(name), "layers.%d.%s", weights->plan.layer, suffix);
    return coli_v4_layer_data(weights, name, spec);
}

static int fp8_view(ColiTensorView *view,
                    const ColiDeepSeekV4LayerWeights *weights,
                    const char *prefix) {
    char name[128];
    const ColiDeepSeekV4TensorSpec *ws = NULL, *ss = NULL;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    const void *data = value(weights, name, &ws);
    snprintf(name, sizeof(name), "%s.scale", prefix);
    const void *scales = value(weights, name, &ss);
    if (!data || !scales || !ws || !ss || ws->rank != 2) return -1;
    *view = (ColiTensorView){
        COLI_TENSOR_FP8_E4M3_BLOCK, COLI_SCALE_UE8M0, data, scales,
        (size_t)(ws->shape[0] * ws->shape[1]),
        (size_t)(ss->shape[0] * ss->shape[1]),
        ws->shape[0], ws->shape[1], 128, 128
    };
    return 0;
}

static void decode_bf16(float *output, const uint16_t *input, size_t count) {
    for (size_t i = 0; i < count; i++) output[i] = coli_bf16_decode(input[i]);
}

static int normalized_hc_pre(float *reduced, float *post, float *comb,
                             float *normalized, const float *input_hc,
                             const ColiDeepSeekV4LayerWeights *weights,
                             const ColiDeepSeekV4Config *config,
                             const char *branch, const char *norm_name) {
    char name[64];
    snprintf(name, sizeof(name), "hc_%s_fn", branch);
    const float *function = value(weights, name, NULL);
    snprintf(name, sizeof(name), "hc_%s_scale", branch);
    const float *scale = value(weights, name, NULL);
    snprintf(name, sizeof(name), "hc_%s_base", branch);
    const float *base = value(weights, name, NULL);
    const uint16_t *raw_norm = value(weights, norm_name, NULL);
    int d = config->hidden_size;
    float *norm = malloc((size_t)d * sizeof(*norm));
    if (!function || !scale || !base || !raw_norm || !norm) {
        free(norm);
        return -1;
    }
    decode_bf16(norm, raw_norm, (size_t)d);
    int result = coli_v4_hc_pre(reduced, post, comb, input_hc, function,
                                scale, base, config->hc_mult, d,
                                config->hc_sinkhorn_iters,
                                config->rms_norm_eps, config->hc_eps);
    if (!result) {
        coli_bf16_round_array(reduced, (size_t)d);
        result = coli_v4_rmsnorm(normalized, reduced, norm, d,
                                 config->rms_norm_eps);
        coli_bf16_round_array(normalized, (size_t)d);
    }
    free(norm);
    return result;
}

static int moe_token(float *output,
                     const ColiDeepSeekV4LayerWeights *weights,
                     const ColiDeepSeekV4Config *config,
                     ColiExpertStore *store, const float *input, int token) {
    int d = config->hidden_size;
    int n = config->n_routed_experts;
    int topk = config->num_experts_per_tok;
    size_t gate_count = (size_t)n * d;
    float *gate = malloc(gate_count * sizeof(*gate));
    float *route_weights = malloc((size_t)topk * sizeof(*route_weights));
    int *indices = malloc((size_t)topk * sizeof(*indices));
    float *expert_output = malloc((size_t)d * sizeof(*expert_output));
    float *shared_output = malloc((size_t)d * sizeof(*shared_output));
    if (!gate || !route_weights || !indices || !expert_output || !shared_output) {
        free(shared_output); free(expert_output); free(indices);
        free(route_weights); free(gate);
        return -1;
    }
    decode_bf16(gate, value(weights, "ffn.gate.weight", NULL), gate_count);
    const int64_t *table = value(weights, "ffn.gate.tid2eid", NULL);
    const float *bias = value(weights, "ffn.gate.bias", NULL);
    int result = token < 0 || token >= config->vocab_size;
    if (!result && weights->plan.uses_hash_router) {
        if (!table) result = -1;
    }
    if (!result && weights->plan.uses_hash_router) {
        for (int i = 0; i < topk; i++)
            indices[i] = (int)table[(size_t)token * topk + i];
    }
    if (!result) result = coli_v4_route(
        route_weights, indices, input, gate, bias,
        weights->plan.uses_hash_router ? indices : NULL,
        n, d, topk, config->routed_scaling_factor);

    ColiTensorView w1, w2, w3;
    if (!result && (fp8_view(&w1, weights, "ffn.shared_experts.w1") ||
                    fp8_view(&w2, weights, "ffn.shared_experts.w2") ||
                    fp8_view(&w3, weights, "ffn.shared_experts.w3"))) result = -1;
    if (!result) result = coli_v4_shared_expert_forward_ref(
        shared_output, &w1, &w2, &w3, input, config->swiglu_limit);
    if (!result) memset(output, 0, (size_t)d * sizeof(*output));
    for (int expert_id = 0; !result && expert_id < n; expert_id++) {
        int rank = -1;
        for (int candidate = 0; candidate < topk; candidate++)
            if (indices[candidate] == expert_id) rank = candidate;
        if (rank < 0) continue;
        ColiExpertView expert;
        if (coli_expert_lookup(store,
                               (ColiExpertKey){weights->plan.layer, expert_id},
                               &expert)) {
            result = -1;
            break;
        }
        result = coli_v4_expert_forward_ref(expert_output, &expert, input,
                                             route_weights[rank],
                                             config->swiglu_limit);
        coli_expert_release(store, &expert);
        if (!result)
            for (int i = 0; i < d; i++) output[i] += expert_output[i];
    }
    if (!result)
        for (int i = 0; i < d; i++)
            output[i] = coli_bf16_round(output[i] + shared_output[i]);
    free(shared_output); free(expert_output); free(indices);
    free(route_weights); free(gate);
    return result;
}

static int block_token_impl(float *output_hc,
                            ColiDeepSeekV4WindowAttentionState *attention,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size) {
    if (!output_hc || !weights || !config || !experts || !input_hc)
        return set_error(error, error_size, "invalid block arguments");
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *residual = malloc(hd * sizeof(*residual));
    float *state = malloc(hd * sizeof(*state));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *normalized = malloc((size_t)d * sizeof(*normalized));
    float *branch = malloc((size_t)d * sizeof(*branch));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    if (!residual || !state || !reduced || !normalized || !branch || !post || !comb) {
        free(comb); free(post); free(branch); free(normalized);
        free(reduced); free(state); free(residual);
        return set_error(error, error_size, "out of memory in block");
    }
    memcpy(residual, input_hc, hd * sizeof(*residual));
    int result = normalized_hc_pre(reduced, post, comb, normalized, input_hc,
                                   weights, config, "attn", "attn_norm.weight");
    if (!result) result = attention
        ? coli_v4_attention_window_token_ref(branch, attention, weights, config,
                                             normalized, position, error, error_size)
        : coli_v4_attention_token_ref(branch, weights, config, normalized,
                                      position, error, error_size);
    if (!result) result = coli_v4_hc_post(state, branch, residual, post, comb, hc, d);
    if (!result) coli_bf16_round_array(state, hd);

    if (!result) memcpy(residual, state, hd * sizeof(*residual));
    if (!result) result = normalized_hc_pre(reduced, post, comb, normalized, state,
                                            weights, config, "ffn", "ffn_norm.weight");
    if (!result) result = moe_token(branch, weights, config, experts, normalized, token);
    if (!result) result = coli_v4_hc_post(output_hc, branch, residual, post, comb, hc, d);
    if (!result) coli_bf16_round_array(output_hc, hd);

    free(comb); free(post); free(branch); free(normalized);
    free(reduced); free(state); free(residual);
    return result ? set_error(error, error_size, "block computation failed") : 0;
}

int coli_v4_block_token_ref(float *output_hc,
                            const ColiDeepSeekV4LayerWeights *weights,
                            const ColiDeepSeekV4Config *config,
                            ColiExpertStore *experts,
                            const float *input_hc, int token, int position,
                            char *error, size_t error_size) {
    return block_token_impl(output_hc, NULL, weights, config, experts, input_hc,
                            token, position, error, error_size);
}

int coli_v4_block_window_token_ref(
    float *output_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *input_hc, int token, int position,
    char *error, size_t error_size) {
    return block_token_impl(output_hc, attention, weights, config, experts,
                            input_hc, token, position, error, error_size);
}
/* ---- end include deepseek_v4_block.c ---- */

#undef coli_v4_block_token_ref
#undef coli_v4_block_window_token_ref

#include "deepseek_v4_internal.h"
#include "native_quant_batch.h"

static int shared_expert_batch(float *outputs, const ColiTensorView *w1,
                               const ColiTensorView *w2,
                               const ColiTensorView *w3,
                               const float *inputs, int batch,
                               float swiglu_limit) {
    int intermediate = (int)w1->rows, d = (int)w2->rows;
    float *gate = malloc((size_t)batch * intermediate * sizeof(*gate));
    float *up = malloc((size_t)batch * intermediate * sizeof(*up));
    float *activated = malloc((size_t)batch * intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp8_matmul_batch_ref(gate, w1, inputs, batch) ||
                 coli_fp8_matmul_batch_ref(up, w3, inputs, batch);
    for (int item = 0; !result && item < batch; item++) {
        float *item_gate = gate + (size_t)item * intermediate;
        float *item_up = up + (size_t)item * intermediate;
        float *item_activated = activated + (size_t)item * intermediate;
        coli_bf16_round_array(item_gate, (size_t)intermediate);
        coli_bf16_round_array(item_up, (size_t)intermediate);
        result = coli_v4_swiglu(item_activated, item_gate, item_up,
                                intermediate, swiglu_limit);
        if (!result) coli_bf16_round_array(item_activated,
                                           (size_t)intermediate);
    }
    if (!result) result = coli_fp8_matmul_batch_ref(
        outputs, w2, activated, batch);
    if (!result) coli_bf16_round_array(outputs, (size_t)batch * d);
    free(activated); free(up); free(gate);
    return result;
}

static int routed_expert_batch(float *outputs, const ColiExpertView *expert,
                               const float *inputs, const float *route_weights,
                               int batch, float swiglu_limit) {
    int intermediate = (int)expert->gate.rows;
    int d = (int)expert->down.rows;
    float *gate = malloc((size_t)batch * intermediate * sizeof(*gate));
    float *up = malloc((size_t)batch * intermediate * sizeof(*up));
    float *activated = malloc((size_t)batch * intermediate * sizeof(*activated));
    if (!gate || !up || !activated) {
        free(activated); free(up); free(gate); return -1;
    }
    int result = coli_fp4_matmul_batch_ref(gate, &expert->gate, inputs, batch) ||
                 coli_fp4_matmul_batch_ref(up, &expert->up, inputs, batch);
    for (int item = 0; !result && item < batch; item++) {
        float *item_gate = gate + (size_t)item * intermediate;
        float *item_up = up + (size_t)item * intermediate;
        float *item_activated = activated + (size_t)item * intermediate;
        coli_bf16_round_array(item_gate, (size_t)intermediate);
        coli_bf16_round_array(item_up, (size_t)intermediate);
        result = coli_v4_swiglu(item_activated, item_gate, item_up,
                                intermediate, swiglu_limit);
        for (int i = 0; !result && i < intermediate; i++)
            item_activated[i] = coli_bf16_round(
                item_activated[i] * route_weights[item]);
    }
    if (!result) result = coli_fp4_matmul_batch_ref(
        outputs, &expert->down, activated, batch);
    if (!result) coli_bf16_round_array(outputs, (size_t)batch * d);
    free(activated); free(up); free(gate);
    return result;
}

static int moe_batch(float *outputs,
                     const ColiDeepSeekV4LayerWeights *weights,
                     const ColiDeepSeekV4Config *config,
                     ColiExpertStore *store, const float *inputs,
                     const int *tokens, int batch) {
    int d = config->hidden_size, n = config->n_routed_experts;
    int topk = config->num_experts_per_tok;
    float *gate = malloc((size_t)n * d * sizeof(*gate));
    float *route_weights = malloc((size_t)batch * topk * sizeof(*route_weights));
    int *indices = malloc((size_t)batch * topk * sizeof(*indices));
    float *shared = malloc((size_t)batch * d * sizeof(*shared));
    float *compact_inputs = malloc((size_t)batch * d * sizeof(*compact_inputs));
    float *compact_outputs = malloc((size_t)batch * d * sizeof(*compact_outputs));
    float *compact_weights = malloc((size_t)batch * sizeof(*compact_weights));
    int *compact_items = malloc((size_t)batch * sizeof(*compact_items));
    if (!gate || !route_weights || !indices || !shared || !compact_inputs ||
        !compact_outputs || !compact_weights || !compact_items) {
        free(compact_items); free(compact_weights); free(compact_outputs);
        free(compact_inputs); free(shared); free(indices); free(route_weights);
        free(gate); return -1;
    }
    decode_bf16(gate, value(weights, "ffn.gate.weight", NULL), (size_t)n * d);
    const int64_t *table = value(weights, "ffn.gate.tid2eid", NULL);
    const float *bias = value(weights, "ffn.gate.bias", NULL);
    int result = 0;
    for (int item = 0; !result && item < batch; item++) {
        int *item_indices = indices + (size_t)item * topk;
        float *item_weights = route_weights + (size_t)item * topk;
        if (tokens[item] < 0 || tokens[item] >= config->vocab_size) result = -1;
        if (!result && weights->plan.uses_hash_router) {
            if (!table) result = -1;
            else for (int i = 0; i < topk; i++)
                item_indices[i] = (int)table[(size_t)tokens[item] * topk + i];
        }
        if (!result) result = coli_v4_route(
            item_weights, item_indices, inputs + (size_t)item * d,
            gate, bias, weights->plan.uses_hash_router ? item_indices : NULL,
            n, d, topk, config->routed_scaling_factor);
    }
    ColiTensorView w1, w2, w3;
    if (!result && (fp8_view(&w1, weights, "ffn.shared_experts.w1") ||
                    fp8_view(&w2, weights, "ffn.shared_experts.w2") ||
                    fp8_view(&w3, weights, "ffn.shared_experts.w3"))) result = -1;
    if (!result) result = shared_expert_batch(
        shared, &w1, &w2, &w3, inputs, batch, config->swiglu_limit);
    if (!result) memset(outputs, 0, (size_t)batch * d * sizeof(*outputs));

    for (int expert_id = 0; !result && expert_id < n; expert_id++) {
        int count = 0;
        for (int item = 0; item < batch; item++) {
            for (int rank = 0; rank < topk; rank++) {
                if (indices[(size_t)item * topk + rank] == expert_id) {
                    compact_items[count] = item;
                    compact_weights[count] =
                        route_weights[(size_t)item * topk + rank];
                    memcpy(compact_inputs + (size_t)count * d,
                           inputs + (size_t)item * d, (size_t)d * sizeof(float));
                    count++;
                }
            }
        }
        if (!count) continue;
        ColiExpertView expert;
        if (coli_expert_lookup(store,
                               (ColiExpertKey){weights->plan.layer, expert_id},
                               &expert)) { result = -1; break; }
        result = routed_expert_batch(compact_outputs, &expert, compact_inputs,
                                     compact_weights, count,
                                     config->swiglu_limit);
        coli_expert_release(store, &expert);
        for (int compact = 0; !result && compact < count; compact++) {
            float *destination = outputs + (size_t)compact_items[compact] * d;
            const float *source = compact_outputs + (size_t)compact * d;
            for (int i = 0; i < d; i++) destination[i] += source[i];
        }
    }
    for (int item = 0; !result && item < batch; item++)
        for (int i = 0; i < d; i++)
            outputs[(size_t)item * d + i] = coli_bf16_round(
                outputs[(size_t)item * d + i] + shared[(size_t)item * d + i]);
    free(compact_items); free(compact_weights); free(compact_outputs);
    free(compact_inputs); free(shared); free(indices); free(route_weights);
    free(gate); return result;
}

int coli_v4_block_window_batch_ref(
    float *outputs_hc, ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens, int start_position, int batch,
    char *error, size_t error_size) {
    if (!outputs_hc || !attention || !weights || !config || !experts ||
        !inputs_hc || !tokens || batch < 1 || batch > 64) return -1;
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *states = malloc((size_t)batch * hd * sizeof(*states));
    float *normalized_ffn = malloc((size_t)batch * d * sizeof(*normalized_ffn));
    float *branches = malloc((size_t)batch * d * sizeof(*branches));
    float *posts = malloc((size_t)batch * hc * sizeof(*posts));
    float *combs = malloc((size_t)batch * hc * hc * sizeof(*combs));
    float *residual = malloc(hd * sizeof(*residual));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *normalized = malloc((size_t)d * sizeof(*normalized));
    float *branch = malloc((size_t)d * sizeof(*branch));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    if (!states || !normalized_ffn || !branches || !posts || !combs ||
        !residual || !reduced || !normalized || !branch || !post || !comb) {
        free(comb); free(post); free(branch); free(normalized); free(reduced);
        free(residual); free(combs); free(posts); free(branches);
        free(normalized_ffn); free(states); return -1;
    }
    int result = 0;
    for (int item = 0; !result && item < batch; item++) {
        const float *input = inputs_hc + (size_t)item * hd;
        float *state = states + (size_t)item * hd;
        memcpy(residual, input, hd * sizeof(float));
        result = normalized_hc_pre(reduced, post, comb, normalized, input,
                                   weights, config, "attn", "attn_norm.weight");
        if (!result) result = coli_v4_attention_window_token_ref(
            branch, attention, weights, config, normalized,
            start_position + item, error, error_size);
        if (!result) result = coli_v4_hc_post(state, branch, residual,
                                              post, comb, hc, d);
        if (!result) coli_bf16_round_array(state, hd);
        if (!result) result = normalized_hc_pre(
            reduced, posts + (size_t)item * hc,
            combs + (size_t)item * hc * hc,
            normalized_ffn + (size_t)item * d, state,
            weights, config, "ffn", "ffn_norm.weight");
    }
    if (!result) result = moe_batch(branches, weights, config, experts,
                                    normalized_ffn, tokens, batch);
    for (int item = 0; !result && item < batch; item++) {
        result = coli_v4_hc_post(
            outputs_hc + (size_t)item * hd,
            branches + (size_t)item * d,
            states + (size_t)item * hd,
            posts + (size_t)item * hc,
            combs + (size_t)item * hc * hc, hc, d);
        if (!result) coli_bf16_round_array(
            outputs_hc + (size_t)item * hd, hd);
    }
    free(comb); free(post); free(branch); free(normalized); free(reduced);
    free(residual); free(combs); free(posts); free(branches);
    free(normalized_ffn); free(states);
    return result ? set_error(error, error_size, "batched block failed") : 0;
}
/* ---- end include deepseek_v4_block_batch.c ---- */

#undef coli_v4_block_window_batch_ref

#include "deepseek_v4_dspark.h"

int coli_v4_target_attention_commit_batch(
    ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const float *inputs_hc, int start_position, int batch,
    char *error, size_t error_size) {
    if (!attention || !weights || !config || !inputs_hc ||
        start_position < 0 || batch < 1 || batch > 64) return -1;
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)hc * d;
    float *normalized = malloc((size_t)batch * d * sizeof(*normalized));
    float *discarded = malloc((size_t)batch * d * sizeof(*discarded));
    float *reduced = malloc((size_t)d * sizeof(*reduced));
    float *post = malloc((size_t)hc * sizeof(*post));
    float *comb = malloc((size_t)hc * hc * sizeof(*comb));
    if (!normalized || !discarded || !reduced || !post || !comb) {
        free(comb); free(post); free(reduced); free(discarded); free(normalized);
        return -1;
    }
    int result = 0;
    for (int item = 0; !result && item < batch; item++)
        result = normalized_hc_pre(
            reduced, post, comb, normalized + (size_t)item * d,
            inputs_hc + (size_t)item * hd, weights, config,
            "attn", "attn_norm.weight");
    if (!result) result = coli_v4_attention_window_batch_ref(
        discarded, attention, weights, config, normalized,
        start_position, batch, error, error_size);
    free(comb); free(post); free(reduced); free(discarded); free(normalized);
    return result;
}
/* ---- end inlined deepseek_v4_target_attention_commit.c ---- */

#undef coli_v4_block_window_token_batch_serial_ref
#undef coli_v4_block_token_batch_serial_ref
#endif /* COLI_V4_UNIT_TARGET_ATTENTION_COMMIT */

#ifdef COLI_V4_UNIT_TARGET_VERIFY_PREFIX
/* ######## deepseek_v4_target_verify_prefix.c ######## */
#include "deepseek_v4_dspark.h"

#include <stdlib.h>
#include <string.h>

#include "deepseek_v4_internal.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_dspark.h"
#include "deepseek_v4_internal.h"
#include "deepseek_v4_dspark.h"
#include "deepseek_v4_dspark.h"

static void prefix_destroy_snapshots(ColiV4AttentionSnapshot **snapshots,
                                     int layers) {
    if (!snapshots) return;
    for (int layer = 0; layer < layers; layer++)
        coli_v4_attention_snapshot_destroy(snapshots[layer]);
    free(snapshots);
}

static int prefix_create_snapshots(
    ColiV4AttentionSnapshot ***output,
    ColiDeepSeekV4WindowAttentionState **attention, int layers) {
    ColiV4AttentionSnapshot **snapshots = calloc(
        (size_t)layers, sizeof(*snapshots));
    if (!snapshots) return -1;
    for (int layer = 0; layer < layers; layer++)
        if (coli_v4_attention_snapshot_create(attention[layer],
                                               &snapshots[layer])) {
            prefix_destroy_snapshots(snapshots, layers); return -1;
        }
    *output = snapshots;
    return 0;
}

static int prefix_restore_snapshots(
    ColiDeepSeekV4WindowAttentionState **attention,
    ColiV4AttentionSnapshot **snapshots, int layers) {
    for (int layer = 0; layer < layers; layer++)
        if (coli_v4_attention_snapshot_restore(attention[layer],
                                               snapshots[layer])) return -1;
    return 0;
}

static int prefix_run_target_batch(
    ColiV4Engine *engine, float **state_ptr, float **next_ptr,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const int *tokens, int start, int batch, float *layer_inputs,
    char *error, size_t error_size) {
    float *state = *state_ptr, *next = *next_ptr;
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    for (int layer_id = 0; layer_id < config->num_hidden_layers; layer_id++) {
        memcpy(layer_inputs + (size_t)layer_id * batch * hd, state,
               (size_t)batch * hd * sizeof(*state));
        ColiDeepSeekV4LayerWeights layer;
        if (coli_v4_layer_load(engine, &layer, config, index, layer_id,
                               error, error_size)) return -1;
        int result = coli_v4_block_window_batch_ref(
            next, attention[layer_id], &layer, config, experts,
            state, tokens, start, batch, error, error_size);
        if (!result)
            coli_v4_dspark_capture_after_block(engine, &layer, config, next,
                                               batch);
        coli_v4_layer_free(engine, &layer);
        if (result) return -1;
        float *swap = state; state = next; next = swap;
    }
    *state_ptr = state;
    *next_ptr = next;
    return 0;
}

static int prefix_commit_attention(
    ColiV4Engine *engine,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config,
    const float *layer_inputs, int recorded_batch,
    int start, int commit_batch, char *error, size_t error_size) {
    size_t hd = (size_t)config->hc_mult * config->hidden_size;
    for (int layer_id = 0; layer_id < config->num_hidden_layers; layer_id++) {
        ColiDeepSeekV4LayerWeights layer;
        if (coli_v4_layer_load(engine, &layer, config, index, layer_id,
                               error, error_size)) return -1;
        int result = coli_v4_target_attention_commit_batch(
            attention[layer_id], &layer, config,
            layer_inputs + (size_t)layer_id * recorded_batch * hd,
            start, commit_batch, error, error_size);
        coli_v4_layer_free(engine, &layer);
        if (result) return -1;
    }
    return 0;
}

int coli_v4_target_verify_after_prefix_v69(
    ColiV4VerificationResult *verification,
    int *output_tokens, int output_capacity,
    ColiV4Engine *engine,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const int *draft_tokens, int draft_count, int prefix_position,
    const float *prefix_main_x, char *error, size_t error_size) {
    enum { FIRST_BATCH = 2, TAIL_BATCH = 2, TOTAL_ROWS = 5 };
    if (!verification || !output_tokens || output_capacity < TOTAL_ROWS ||
        !attention || !index || !config || !experts || !draft_tokens ||
        draft_count != 4 || prefix_position < 0 || !prefix_main_x) return -1;

    int result = 0;
    int d = config->hidden_size, hc = config->hc_mult;
    size_t hd = (size_t)d * hc;
    float *combined_main_x = malloc((size_t)TOTAL_ROWS * d * sizeof(float));
    float *state = malloc((size_t)FIRST_BATCH * hd * sizeof(float));
    float *next = malloc((size_t)FIRST_BATCH * hd * sizeof(float));
    float *layer_inputs = malloc((size_t)config->num_hidden_layers *
                                 FIRST_BATCH * hd * sizeof(float));
    ColiV4AttentionSnapshot **snapshots = NULL;
    if (!combined_main_x || !state || !next || !layer_inputs ||
        prefix_create_snapshots(&snapshots, attention,
                                config->num_hidden_layers)) {
        result = -1; goto cleanup;
    }
    memcpy(combined_main_x, prefix_main_x, (size_t)d * sizeof(float));

    int first_inputs[FIRST_BATCH] = {draft_tokens[0], draft_tokens[1]};
    int first_targets[FIRST_BATCH];
    float first_logits[FIRST_BATCH];
    result = coli_v4_target_load_embeddings(
                 state, index, config, first_inputs, FIRST_BATCH) ||
             prefix_run_target_batch(
                 engine, &state, &next, attention, index, config, experts,
                 first_inputs, prefix_position + 1, FIRST_BATCH,
                 layer_inputs, error, error_size) ||
             coli_v4_target_head_argmax_batch(
                 state, index, config, FIRST_BATCH,
                 first_targets, first_logits, error, error_size) ||
             coli_v4_dspark_capture_main_x(engine, combined_main_x + d, FIRST_BATCH, config);
    if (result) goto cleanup;

    output_tokens[0] = draft_tokens[0];
    int accepted = 0;
    while (accepted < FIRST_BATCH &&
           draft_tokens[accepted + 1] == first_targets[accepted]) {
        output_tokens[accepted + 1] = draft_tokens[accepted + 1];
        accepted++;
    }
    if (accepted < FIRST_BATCH) {
        int commit = accepted + 1;
        output_tokens[accepted + 1] = first_targets[accepted];
        verification->accepted_draft_tokens = accepted + 1;
        verification->output_count = accepted + 2;
        verification->mismatch_index = accepted + 1;
        if (prefix_restore_snapshots(attention, snapshots,
                                     config->num_hidden_layers) ||
            prefix_commit_attention(
                engine, attention, index, config, layer_inputs, FIRST_BATCH,
                prefix_position + 1, commit, error, error_size) ||
            coli_v4_dspark_capture_stage_main_x(engine, combined_main_x, commit + 1, d)) result = -1;
        goto cleanup;
    }

    prefix_destroy_snapshots(snapshots, config->num_hidden_layers);
    snapshots = NULL;
    free(layer_inputs); free(next); free(state);
    layer_inputs = NULL; next = NULL; state = NULL;

    state = malloc((size_t)TAIL_BATCH * hd * sizeof(float));
    next = malloc((size_t)TAIL_BATCH * hd * sizeof(float));
    layer_inputs = malloc((size_t)config->num_hidden_layers *
                          TAIL_BATCH * hd * sizeof(float));
    if (!state || !next || !layer_inputs ||
        prefix_create_snapshots(&snapshots, attention,
                                config->num_hidden_layers)) {
        result = -1; goto cleanup;
    }
    int tail_inputs[TAIL_BATCH] = {draft_tokens[2], draft_tokens[3]};
    int tail_targets[TAIL_BATCH];
    float tail_logits[TAIL_BATCH];
    result = coli_v4_target_load_embeddings(
                 state, index, config, tail_inputs, TAIL_BATCH) ||
             prefix_run_target_batch(
                 engine, &state, &next, attention, index, config, experts,
                 tail_inputs, prefix_position + 3, TAIL_BATCH,
                 layer_inputs, error, error_size) ||
             coli_v4_target_head_argmax_batch(
                 state, index, config, TAIL_BATCH,
                 tail_targets, tail_logits, error, error_size) ||
             coli_v4_dspark_capture_main_x(engine, combined_main_x + (size_t)3 * d, TAIL_BATCH, config);
    if (result) goto cleanup;

    output_tokens[1] = draft_tokens[1];
    output_tokens[2] = draft_tokens[2];
    if (draft_tokens[3] == tail_targets[0]) {
        output_tokens[3] = draft_tokens[3];
        output_tokens[4] = tail_targets[1];
        verification->accepted_draft_tokens = 4;
        verification->output_count = 5;
        verification->mismatch_index = -1;
        result = coli_v4_dspark_capture_stage_main_x(engine, combined_main_x, TOTAL_ROWS, d);
    } else {
        output_tokens[3] = tail_targets[0];
        verification->accepted_draft_tokens = 3;
        verification->output_count = 4;
        verification->mismatch_index = 3;
        if (prefix_restore_snapshots(attention, snapshots,
                                     config->num_hidden_layers) ||
            prefix_commit_attention(
                engine, attention, index, config, layer_inputs, TAIL_BATCH,
                prefix_position + 3, 1, error, error_size) ||
            coli_v4_dspark_capture_stage_main_x(engine, combined_main_x, 4, d)) result = -1;
    }

cleanup:
    prefix_destroy_snapshots(snapshots, config->num_hidden_layers);
    free(layer_inputs); free(next); free(state); free(combined_main_x);
    return result ? -1 : 0;
}
#endif /* COLI_V4_UNIT_TARGET_VERIFY_PREFIX */

#ifdef COLI_V4_UNIT_DSPARK_VERIFY_WINDOW
/* ######## deepseek_v4_dspark_verify_window.c ######## */
/* Verify-window batching now lives in coli_v4_dspark_runner_draft /
 * coli_v4_dspark_runner_verify_block_size (no ld --wrap). */
#endif /* COLI_V4_UNIT_DSPARK_VERIFY_WINDOW */
