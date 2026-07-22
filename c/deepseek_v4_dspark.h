#ifndef COLIBRI_DEEPSEEK_V4_DSPARK_H
#define COLIBRI_DEEPSEEK_V4_DSPARK_H

#include "deepseek_v4_internal.h"

#include "tensor.h"
#include "expert_store.h"
#include "native_quant.h"
#include "native_quant_batch.h"
#include "native_quant_dual.h"
#include "native_quant_fp4_rows16.h"
#include "safetensors_index.h"
#include "tensor_io.h"

/* ==== begin deepseek_v4_dspark.h ==== */

#include <stddef.h>
#include <stdint.h>

/* via deepseek_v4.h: deepseek_v4_config.h */
/* via deepseek_v4.h: deepseek_v4_layer.h */

/* COLI_V4_DSPARK_MAX_* and ColiDeepSeekV4DSparkManifest: deepseek_v4_internal.h */

int coli_v4_dspark_inspect(const char *model_dir,
                           const ColiDeepSeekV4Config *config,
                           ColiDeepSeekV4DSparkManifest *manifest,
                           char *error, size_t error_size);
int coli_v4_dspark_layer_plan(ColiDeepSeekV4LayerPlan *plan,
                              const ColiDeepSeekV4Config *config, int stage,
                              char *error, size_t error_size);
/* ==== end deepseek_v4_dspark.h ==== */

/* ==== begin deepseek_v4_dspark_memory.h ==== */

#include <stddef.h>
#include <stdint.h>

/* via deepseek_v4.h: deepseek_v4_config.h */

typedef struct {
    uint64_t resident_heads_bytes;
    uint64_t streamed_stage_bytes;
    uint64_t minimum_expert_cache_bytes;
    uint64_t working_bytes;
    uint64_t incremental_reserve_bytes;
    uint64_t expert_record_bytes;
    int stages;
    int expert_slots_per_stage;
} ColiV4DSparkMemoryPlan;

int coli_v4_dspark_memory_plan(const char *model_dir,
                               const ColiDeepSeekV4Config *config,
                               ColiV4DSparkMemoryPlan *plan,
                               char *error, size_t error_size);
/* ==== end deepseek_v4_dspark_memory.h ==== */

/* ==== begin deepseek_v4_dspark_runtime.h ==== */

/* amalgamated: deepseek_v4_dspark.h */
/* via deepseek_v4.h: deepseek_v4_expert_store.h */

int coli_v4_dspark_layer_load(ColiV4Engine *engine,
                              ColiDeepSeekV4LayerWeights *weights,
                              const ColiDeepSeekV4Config *config,
                              const ColiSafetensorsIndex *index, int stage,
                              char *error, size_t error_size);
void coli_v4_dspark_layer_release(ColiV4Engine *engine,
                                  ColiDeepSeekV4LayerWeights *weights);
int coli_deepseek_v4_dspark_expert_store_open(
    const ColiDeepSeekV4ExpertStoreOptions *options,
    ColiExpertStore **store, char *error, size_t error_size);
/* ==== end deepseek_v4_dspark_runtime.h ==== */

/* ==== begin deepseek_v4_dspark_runtime_resident.h ==== */

/* via deepseek_v4.h: deepseek_v4_layer.h */
/* coli_v4_dspark_layer_release declared above with engine parameter */

/* ==== end deepseek_v4_dspark_runtime_resident.h ==== */

/* ==== begin deepseek_v4_dspark_attention.h ==== */

#include <stddef.h>

/* via deepseek_v4.h: deepseek_v4_config.h */
/* via deepseek_v4.h: deepseek_v4_layer.h */

typedef struct ColiV4DSparkAttentionState ColiV4DSparkAttentionState;

/*
 * Record an absolute token position in a sparse fixed-size sliding window.
 * Positions may jump when one speculative verification commits several
 * tokens, so occupancy cannot be inferred from the number of writes.
 */
static inline int coli_v4_dspark_position_window_put(
    int *positions, int window_size, int *valid, int position) {
    if (!positions || !valid || window_size < 1 || position < 0) return -1;
    int64_t minimum = (int64_t)position - window_size + 1;
    for (int slot = 0; slot < window_size; slot++)
        if (positions[slot] >= 0 && (int64_t)positions[slot] < minimum)
            positions[slot] = -1;
    int slot = position % window_size;
    positions[slot] = position;
    int occupied = 0;
    for (int index = 0; index < window_size; index++)
        if (positions[index] >= 0) occupied++;
    *valid = occupied;
    return slot;
}

int coli_v4_dspark_attention_create(ColiV4DSparkAttentionState **state,
                                    const ColiDeepSeekV4Config *config);
void coli_v4_dspark_attention_reset(ColiV4DSparkAttentionState *state);
void coli_v4_dspark_attention_destroy(ColiV4DSparkAttentionState *state);
int coli_v4_dspark_attention_context_count(
    const ColiV4DSparkAttentionState *state);
int coli_v4_dspark_attention_precompute_context(
    ColiV4DSparkAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const float *main_x, int start_position, int batch,
    char *error, size_t error_size);
/* ==== end deepseek_v4_dspark_attention.h ==== */

/* ==== begin deepseek_v4_dspark_attention_block.h ==== */

/* amalgamated: deepseek_v4_dspark_attention.h */

int coli_v4_dspark_attention_block(
    float *outputs, ColiV4DSparkAttentionState *state,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *inputs,
    int query_start_position, int batch, char *error, size_t error_size);
/* ==== end deepseek_v4_dspark_attention_block.h ==== */

/* ==== begin deepseek_v4_dspark_block.h ==== */

/* amalgamated: deepseek_v4_dspark_attention.h */
/* via deepseek_v4.h: deepseek_v4_expert_store.h */

int coli_v4_dspark_block(
    float *outputs_hc, ColiV4DSparkAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const float *inputs_hc, const int *tokens,
    int query_start_position, int batch,
    char *error, size_t error_size);
/* ==== end deepseek_v4_dspark_block.h ==== */

/* ==== begin deepseek_v4_dspark_final.h ==== */

/* via deepseek_v4.h: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_dspark.h */

typedef struct ColiV4DSparkFinal ColiV4DSparkFinal;

int coli_v4_dspark_final_open(ColiV4DSparkFinal **output,
                              const char *model_dir,
                              const ColiDeepSeekV4Config *config,
                              const ColiDeepSeekV4DSparkManifest *manifest,
                              char *error, size_t error_size);
void coli_v4_dspark_final_close(ColiV4DSparkFinal *final);
int coli_v4_dspark_final_hidden(ColiV4DSparkFinal *final,
                                float *outputs, const float *states_hc,
                                int batch);
/* ==== end deepseek_v4_dspark_final.h ==== */

/* ==== begin deepseek_v4_dspark_heads.h ==== */

#include <stddef.h>

/* via deepseek_v4.h: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_dspark.h */
#include "safetensors_index.h"

typedef struct ColiV4DSparkHeads ColiV4DSparkHeads;

int coli_v4_dspark_heads_open(ColiV4DSparkHeads **output,
                              const char *model_dir,
                              const ColiDeepSeekV4Config *config,
                              const ColiDeepSeekV4DSparkManifest *manifest,
                              char *error, size_t error_size);
void coli_v4_dspark_heads_close(ColiV4DSparkHeads *heads);
int coli_v4_dspark_combine_hidden(ColiV4DSparkHeads *heads, float *output,
                                  const float *target_hidden_hc);
int coli_v4_dspark_markov_bias(ColiV4DSparkHeads *heads, float *bias,
                               int previous_token);
int coli_v4_dspark_biased_argmax(
    ColiV4Engine *engine, ColiV4DSparkHeads *heads,
    const ColiSafetensorsIndex *target_index, const float *hidden,
    int previous_token, int *best_token, float *best_logit);
int coli_v4_dspark_biased_argmax_batch(
    ColiV4Engine *engine, ColiV4DSparkHeads *heads,
    const ColiSafetensorsIndex *target_index, const float *hidden_batch,
    int previous_token, int *best_tokens, float *best_logits, int batch);
/* ==== end deepseek_v4_dspark_heads.h ==== */

/* ==== begin deepseek_v4_dspark_runner.h ==== */

#include <stddef.h>
#include <stdint.h>

/* via deepseek_v4.h: deepseek_v4_config.h */

typedef struct ColiV4DSparkRunner ColiV4DSparkRunner;

int coli_v4_dspark_runner_open(ColiV4DSparkRunner **output,
                               ColiV4Engine *engine,
                               const char *dspark_model_dir,
                               const char *target_model_dir,
                               const ColiDeepSeekV4Config *config,
                               uint64_t expert_cache_bytes,
                               char *error, size_t error_size);
void coli_v4_dspark_runner_close(ColiV4DSparkRunner *runner);
int coli_v4_dspark_runner_prefill(ColiV4DSparkRunner *runner,
                                  const float *main_x,
                                  int start_position, int batch,
                                  char *error, size_t error_size);
int coli_v4_dspark_runner_draft(ColiV4DSparkRunner *runner,
                                const float *main_x, int anchor_token,
                                int position, int *draft_tokens,
                                float *draft_logits,
                                char *error, size_t error_size);
int coli_v4_dspark_runner_block_size(const ColiV4DSparkRunner *runner);
/* Verify-window draft count (min of manifest.block_size and runtime.verify_drafts). */
int coli_v4_dspark_runner_verify_block_size(const ColiV4DSparkRunner *runner);
uint64_t coli_v4_dspark_runner_loaded_stage_peak(
    const ColiV4DSparkRunner *runner);
/* ==== end deepseek_v4_dspark_runner.h ==== */

/* ==== begin deepseek_v4_dspark_runner_shared.h ==== */

/* amalgamated: deepseek_v4_dspark_heads.h */
/* amalgamated: deepseek_v4_dspark_runner.h */

int coli_v4_dspark_runner_use_shared_heads(ColiV4DSparkRunner *runner,
                                           ColiV4DSparkHeads *heads);
/* ==== end deepseek_v4_dspark_runner_shared.h ==== */

/* ==== begin deepseek_v4_dspark_capture.h ==== */

/* via deepseek_v4.h: deepseek_v4_config.h */
/* amalgamated: deepseek_v4_dspark_heads.h */

int coli_v4_dspark_capture_main_x(ColiV4Engine *engine, float *outputs,
                                  int batch,
                                  const ColiDeepSeekV4Config *config);
ColiV4DSparkHeads *coli_v4_dspark_capture_heads(ColiV4Engine *engine);
int coli_v4_dspark_capture_stage_main_x(ColiV4Engine *engine,
                                        const float *values, int batch,
                                        int hidden_size);
void coli_v4_dspark_capture_after_block(
    ColiV4Engine *engine, const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config, const float *outputs, int batch);
/* ==== end deepseek_v4_dspark_capture.h ==== */

/* ==== begin deepseek_v4_speculative.h ==== */

#include <stdint.h>

typedef struct {
    uint64_t rounds;
    uint64_t proposed;
    uint64_t accepted;
    uint64_t minimum_proposals;
    float disable_threshold;
    int enabled;
} ColiV4SpeculativeController;

typedef struct {
    int accepted_draft_tokens;
    int output_count;
    int mismatch_index;
} ColiV4VerificationResult;

void coli_v4_speculative_controller_init(ColiV4SpeculativeController *state,
                                         uint64_t minimum_proposals,
                                         float disable_threshold);
void coli_v4_speculative_record(ColiV4SpeculativeController *state,
                                int proposed, int accepted);
float coli_v4_speculative_acceptance(const ColiV4SpeculativeController *state);

int coli_v4_verify_greedy(ColiV4VerificationResult *result,
                          int *output_tokens, int output_capacity,
                          const int *draft_tokens, int draft_count,
                          const int *target_tokens, int target_count);
/* ==== end deepseek_v4_speculative.h ==== */

/* ==== begin deepseek_v4_target_verify.h ==== */

/* via deepseek_v4.h: deepseek_v4_attention.h */
/* via deepseek_v4.h: deepseek_v4_config.h */
/* via deepseek_v4.h: deepseek_v4_expert_store.h */
/* amalgamated: deepseek_v4_speculative.h */

int coli_v4_target_verify_greedy_batch(
    ColiV4VerificationResult *verification,
    int *output_tokens, int output_capacity,
    ColiV4Engine *engine,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    int anchor_token, const int *draft_tokens, int draft_count,
    int start_position, char *error, size_t error_size);
/* ==== end deepseek_v4_target_verify.h ==== */

/* ==== begin deepseek_v4_target_verify_prefix.h ==== */

/* amalgamated: deepseek_v4_target_verify.h */

/* The target has already processed the anchor and draft_tokens[0] matched its
 * prediction. Verify drafts 1..3 with the same 2+2 schedule as the prior
 * verifier, while preserving the prefix main_x row for the next DSpark round. */
int coli_v4_target_verify_after_prefix_v69(
    ColiV4VerificationResult *verification,
    int *output_tokens, int output_capacity,
    ColiV4Engine *engine,
    ColiDeepSeekV4WindowAttentionState **attention,
    const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, ColiExpertStore *experts,
    const int *draft_tokens, int draft_count, int prefix_position,
    const float *prefix_main_x, char *error, size_t error_size);
/* ==== end deepseek_v4_target_verify_prefix.h ==== */

/* ==== begin deepseek_v4_target_attention_commit.h ==== */

/* via deepseek_v4.h: deepseek_v4_attention.h */
/* via deepseek_v4.h: deepseek_v4_config.h */
/* via deepseek_v4.h: deepseek_v4_layer.h */

/* Advance only a target layer's causal attention/cache for already-computed
 * layer inputs. Used after speculative rollback to avoid replaying HC post and
 * the complete MoE branch. */
int coli_v4_target_attention_commit_batch(
    ColiDeepSeekV4WindowAttentionState *attention,
    const ColiDeepSeekV4LayerWeights *weights,
    const ColiDeepSeekV4Config *config,
    const float *inputs_hc, int start_position, int batch,
    char *error, size_t error_size);
/* ==== end deepseek_v4_target_attention_commit.h ==== */

/* ==== begin deepseek_v4_target_head_batch.h ==== */

/* via deepseek_v4.h: deepseek_v4_config.h */
#include "safetensors_index.h"

int coli_v4_target_load_embeddings(float *states_hc,
                                    const ColiSafetensorsIndex *index,
                                    const ColiDeepSeekV4Config *config,
                                    const int *tokens, int batch);
int coli_v4_target_head_argmax_batch(
    const float *states_hc, const ColiSafetensorsIndex *index,
    const ColiDeepSeekV4Config *config, int batch,
    int *tokens, float *logits, char *error, size_t error_size);
/* ==== end deepseek_v4_target_head_batch.h ==== */

#endif /* COLIBRI_DEEPSEEK_V4_DSPARK_H */
