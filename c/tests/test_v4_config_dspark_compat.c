/* Focused regression test for the config parser linked by the production V4
 * binary. This intentionally links COLI_V4_UNIT_CONFIG_DSPARK_COMPAT, not the
 * standalone COLI_V4_UNIT_CONFIG test object. */
#include "../deepseek_v4.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char target_config[] =
    "{\"model_type\":\"deepseek_v4\",\"expert_dtype\":\"fp4\","
    "\"scoring_func\":\"sqrtsoftplus\",\"topk_method\":\"noaux_tc\","
    "\"hidden_size\":128,\"num_hidden_layers\":3,"
    "\"num_attention_heads\":4,\"head_dim\":32,\"q_lora_rank\":64,"
    "\"qk_rope_head_dim\":8,\"o_groups\":2,\"o_lora_rank\":64,"
    "\"sliding_window\":16,\"index_n_heads\":4,\"index_head_dim\":16,"
    "\"index_topk\":8,\"n_routed_experts\":8,\"num_experts_per_tok\":2,"
    "\"n_shared_experts\":1,\"moe_intermediate_size\":32,"
    "\"num_hash_layers\":1,\"num_nextn_predict_layers\":1,"
    "\"hc_mult\":4,\"hc_sinkhorn_iters\":5,\"vocab_size\":256,"
    "\"max_position_embeddings\":4096,\"rms_norm_eps\":1e-6,"
    "\"hc_eps\":1e-6,\"routed_scaling_factor\":1.5,\"swiglu_limit\":10,"
    "\"rope_theta\":10000,\"compress_rope_theta\":40000,"
    "\"compress_ratios\":[0,4,128,0],"
    "\"rope_scaling\":{\"original_max_position_embeddings\":1024,"
    "\"beta_fast\":32,\"beta_slow\":1,\"factor\":4},"
    "\"quantization_config\":{\"fmt\":\"e4m3\",\"scale_fmt\":\"ue8m0\"}}";

static const char dspark_config[] =
    "{\"model_type\":\"deepseek_v4\",\"expert_dtype\":\"fp4\","
    "\"scoring_func\":\"sqrtsoftplus\",\"topk_method\":\"noaux_tc\","
    "\"hidden_size\":128,\"num_hidden_layers\":3,"
    "\"num_attention_heads\":4,\"head_dim\":32,\"q_lora_rank\":64,"
    "\"qk_rope_head_dim\":8,\"o_groups\":2,\"o_lora_rank\":64,"
    "\"sliding_window\":16,\"index_n_heads\":4,\"index_head_dim\":16,"
    "\"index_topk\":8,\"n_routed_experts\":8,\"num_experts_per_tok\":2,"
    "\"n_shared_experts\":1,\"moe_intermediate_size\":32,"
    "\"num_hash_layers\":1,\"num_nextn_predict_layers\":0,"
    "\"hc_mult\":4,\"hc_sinkhorn_iters\":5,\"vocab_size\":256,"
    "\"max_position_embeddings\":4096,\"rms_norm_eps\":1e-6,"
    "\"hc_eps\":1e-6,\"routed_scaling_factor\":1.5,\"swiglu_limit\":10,"
    "\"rope_theta\":10000,\"compress_rope_theta\":40000,"
    "\"compress_ratios\":[0,4,128,0],\"dspark_block_size\":5,"
    "\"rope_scaling\":{\"original_max_position_embeddings\":1024,"
    "\"beta_fast\":32,\"beta_slow\":1,\"factor\":4},"
    "\"quantization_config\":{\"fmt\":\"e4m3\",\"scale_fmt\":\"ue8m0\"}}";

static char *replace_once(const char *source, const char *needle,
                          const char *replacement) {
    const char *match = strstr(source, needle);
    if (!match) return NULL;
    size_t prefix = (size_t)(match - source);
    size_t length = prefix + strlen(replacement) +
                    strlen(match + strlen(needle)) + 1;
    char *result = malloc(length);
    if (!result) return NULL;
    memcpy(result, source, prefix);
    strcpy(result + prefix, replacement);
    strcpy(result + prefix + strlen(replacement), match + strlen(needle));
    return result;
}

static int parse_ok(const char *json, int expected_nextn, const char *label) {
    ColiDeepSeekV4Config config;
    char error[256] = {0};
    if (coli_v4_config_parse(&config, json, error, sizeof(error))) {
        fprintf(stderr, "%s rejected: %s\n", label, error);
        return 1;
    }
    if (config.hidden_size != 128 || config.num_hidden_layers != 3 ||
        config.num_nextn_predict_layers != expected_nextn ||
        config.compress_ratio_count != 4 ||
        config.compress_ratios[1] != 4 || config.compress_ratios[2] != 128) {
        fprintf(stderr, "%s produced incorrect config values\n", label);
        return 1;
    }
    return 0;
}

static int parse_rejected(const char *json, const char *label) {
    ColiDeepSeekV4Config config;
    char error[256] = {0};
    if (!coli_v4_config_parse(&config, json, error, sizeof(error))) {
        fprintf(stderr, "%s was accepted\n", label);
        return 1;
    }
    return 0;
}

int main(void) {
    static const struct {
        const char *label;
        const char *needle;
        const char *replacement;
    } invalid[] = {
        {"fractional integer", "\"hidden_size\":128", "\"hidden_size\":1.5"},
        {"integer NaN", "\"hidden_size\":128", "\"hidden_size\":NaN"},
        {"integer Infinity", "\"hidden_size\":128", "\"hidden_size\":1e309"},
        {"floating NaN", "\"rms_norm_eps\":1e-6", "\"rms_norm_eps\":NaN"},
        {"floating Infinity", "\"rms_norm_eps\":1e-6",
         "\"rms_norm_eps\":1e309"},
        {"integer overflow", "\"hidden_size\":128",
         "\"hidden_size\":2147483648"},
        {"invalid compress ratio", "\"compress_ratios\":[0,4,128,0]",
         "\"compress_ratios\":[0,1.5,128,0]"},
        {"non-finite compress ratio", "\"compress_ratios\":[0,4,128,0]",
         "\"compress_ratios\":[0,NaN,128,0]"},
        {"overflowing compress ratio", "\"compress_ratios\":[0,4,128,0]",
         "\"compress_ratios\":[0,2147483648,128,0]"},
    };
    enum { REPEAT_COUNT = 1000 };
    char *invalid_json[sizeof(invalid) / sizeof(invalid[0])] = {0};
    for (size_t item = 0; item < sizeof(invalid) / sizeof(invalid[0]); item++) {
        invalid_json[item] = replace_once(
            dspark_config, invalid[item].needle, invalid[item].replacement);
        if (!invalid_json[item]) {
            fprintf(stderr, "cannot build %s fixture\n", invalid[item].label);
            return 1;
        }
    }

    int failed = 0;
    for (int iteration = 0; iteration < REPEAT_COUNT && !failed; iteration++) {
        failed = parse_ok(target_config, 1, "target config") ||
                 parse_ok(dspark_config, 1, "DSpark compatibility config");
        for (size_t item = 0;
             item < sizeof(invalid) / sizeof(invalid[0]) && !failed; item++)
            failed = parse_rejected(invalid_json[item], invalid[item].label);
    }
    for (size_t item = 0; item < sizeof(invalid) / sizeof(invalid[0]); item++)
        free(invalid_json[item]);
    if (failed) return 1;
    puts("DeepSeek-V4 production DSpark config parser tests: ok");
    return 0;
}
