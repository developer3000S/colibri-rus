# DeepSeek V4 (colibri CPU)

English · [简体中文](deepseek-v4.zh-CN.md)

Local CPU inference for DeepSeek V4 Flash + DSpark on colibri: weight loading,
expert caching, sparse attention, speculative decode, and automatic RAM
tiering—end-to-end generation that passes smoke tests.

## Done

- **Target engine**: config / layer / attention / block / expert store / math /
  runtime / prompt; FP8 dense + native FP4 expert kernels; AVX-512 batch
  verify and rows16 hot-expert layout when available.
- **DSpark speculative**: draft runner, heads, verify window, prefix/commit;
  acceptance reaches 100% on the smoke prompt.
- **Automatic RAM policy**: plans resident tensors, DSpark stages, expert
  cache, and the output head from OS-available memory or `--ram`.
- **SSD expert streaming**: two coalesced reads per expert (scales + weights);
  no reordered 160 GB container required.
- **Source layout**: GLM-style amalgams `deepseek_v4.c` /
  `deepseek_v4_dspark.c`; public API in `deepseek_v4.h`
  (`ColiV4Engine` / `ColiV4Session` / config / prompt); implementation details in
  `deepseek_v4_internal.h` (not a stability commitment).
  `make deepseek-v4` builds `c/deepseek_v4.exe`.
- **CLI**: `c/v4` (Python launcher, stdlib only) wraps `run` / `chat`; inference
  itself is engine + session (`coli_v4_engine_open` → `coli_v4_session_create` /
  `generate` → destroy sessions, then engine). The engine/session surface in
  `deepseek_v4.h` is an **experimental** public API and may change. Engine copies
  model directory strings; destroy every session before `coli_v4_engine_destroy`.
  Index and ExpertStore accessors stay internal (`deepseek_v4_internal.h`).

## Model snapshot

Typical DeepSeek-V4-Flash-DSpark shape (from checkpoint metadata):

| Item | Value |
|------|------:|
| Transformer layers | 43 |
| Hidden size | 4096 |
| Routed experts / layer | 256 (top-k 6) |
| Bytes per expert record | ~12.75 MiB (FP4 E2M1 + UE8M0 scales) |
| Sliding window | 128 |
| Dense resident (layers) | ~6.27 GiB |
| Routed expert payload | ~137 GiB (streamed; not all resident) |

Experts are packed FP4 (`float4_e2m1fn_x2` in I8 safetensors), not INT8.
Dense weights use E4M3 with UE8M0 block scales. Activations are FP8×FP4 with
FP32 accumulate on the reference path.

## Automatic RAM

Budget order: system reserve → runtime working set → minimum expert slots
(top-k × sparse layers) → optional resident BF16 lm_head (~1.06 GiB) → grow
per-layer expert-cache slots with leftover budget.

- Without `--ram`, the planner uses **OS-available** memory (not total RAM) and
  keeps an adaptive system reserve.
- With `--ram GiB`, that value is a **planner budget**, not an OS-enforced hard
  limit. It governs projected resident weights, caches, and runtime workspace.
- If the minimum expert working set does not fit, startup fails with a clear
  shortfall (no silent swap overcommit).
- Head residency is optional: low-memory modes stream the BF16 head; surplus
  keeps it resident for speed.
- Hot experts may be pinned and rearranged in place to a 16-row AVX-512 layout;
  cold experts stay official row-major FP4. Capacity does not grow from the
  rearrange.

Example residency under an explicit 32 GiB planner budget:

```text
dense ≈ 6.27 GiB
dspark ≈ 10.45 GiB
target expert cache ≈ 9.1 GiB (more slots when RAM allows)
head = resident BF16
```

At 8 GiB the planner can still produce a process plan when the **host** has
enough free memory for the minimum working set; decode is slower because the
expert cache is tighter. That is not the same as shipping on a physical 8 GiB
PC—see Bench.

## Bench

Hardware used for the numbers below:

| Item | Value |
|------|-------|
| CPU | AMD Ryzen AI MAX+ 395 (16 cores / 32 threads, Radeon 8060S) |
| System RAM | 128 GiB |
| OS | Windows 11 |
| Build | `make deepseek-v4` (MSYS2 UCRT64, `-march=x86-64-v3`) |

Model: `DeepSeek-V4-Flash-DSpark`

Prompt: `--stop-sentence "What is the capital of France?"`
(output: *The capital of France is Paris.*)

`--ram` here is a **planner budget** on this high-RAM machine, not a hard RSS
limit or a physical 8 GiB/32 GiB PC. The **8 GiB row is a planning setting**
(`--ram 8`): the host still has ample OS RAM/page cache for SSD expert I/O.
A real ~8 GiB system may fail to start or run much slower once OS reserve,
runtime scratch, and disk cache compete for the same budget—do not treat that
row as a guarantee for 8 GiB hardware.

| `--ram` | TTFT | prefill | decode | DSpark acceptance |
|---------|------|---------|--------|-------------------|
| 32 GiB | 9.57s | 1.403 tok/s | 1.236 tok/s | 100% |
| 8 GiB (planner budget) | 10.02s | 1.338 tok/s | 0.712 tok/s | 100% |

```text
# --ram 32
[stats] RAM 31.98/32.00 GiB | TTFT 9.57s | prefill 1.403 tok/s | decode 1.236 tok/s | DSpark acceptance 100.0%

# --ram 8  (planner budget on ~127 GiB host — not a physical 8 GiB machine)
[stats] RAM 7.55/8.00 GiB | TTFT 10.02s | prefill 1.338 tok/s | decode 0.712 tok/s | DSpark acceptance 100.0%
```

## Build

Supported platforms for the V4 engine and its amalgam unit tests:

- x86-64 Linux (gcc; AVX-512 paths when available)
- Windows / MSYS2 UCRT64 (same)

macOS, PowerPC, and other hosts keep validating the main colibri engine via
`make check` and do
**not** build or link DeepSeek V4. On unsupported platforms `make deepseek-v4`
exits with a clear error.

```bash
# MSYS2 UCRT64
export PATH=/ucrt64/bin:/usr/bin
cd /d/ai/colibri/c
make deepseek-v4
```

Or from the repo root: `make deepseek-v4` → `c/deepseek_v4.exe`.

Amalgamated sources are compiled per former top-level unit via
`-DCOLI_V4_UNIT_*` so onion `#include` + macro wraps stay correct. Shared
kernels (`native_quant*`, `safetensors_index`, `tensor_io`) remain separate
translation units.

## Committed tiny independent oracle

`make check` on x86-64 Linux and Windows/MSYS2 builds the V4 engine and runs
`make deepseek-v4-tiny-check`.  The committed `c/deepseek_v4_tiny` fixture is a
valid reduced checkpoint, including one DSpark stage, and is about 1.2 MiB.
It requires no network, PyTorch, Transformers, or model download at test time.
Unsupported platforms skip the V4 execution while retaining the ordinary GLM
build and platform-gating checks.

The reference in `deepseek_v4_tiny/ref.json` is generated by the official
Transformers `DeepseekV4ForCausalLM`, after the dense FP8, routed-expert FP4,
and BF16 weight round trips used by the C checkpoint.  The dependency-free
test compares integer token IDs, never decoded text, and covers:

- target teacher forcing and target-only greedy decoding;
- target session generation with drafting disabled;
- DSpark proposal plus target verification, with exact target identity;
- sliding, compressed sparse, and heavily compressed attention;
- a 72-token prompt crossing the internal 64-token prefill boundary; and
- repeated engine/session open, generate, and destroy lifecycles.

Regeneration is intentionally separate from CI.  The checked-in fixture was
created with Python 3.12, PyTorch 2.13.0, Transformers 5.14.1, and
safetensors 0.8.0 (installed as a Transformers dependency):

```bash
python -m pip install torch==2.13.0 transformers==5.14.1 safetensors==0.8.0
python c/tools/make_deepseek_v4_tiny.py --force
make -C c deepseek-v4-tiny-check
```

The generator has no C-engine fallback: it fails if official DeepSeek V4
support is unavailable, prints every tensor name and shape, and records its
schema/generator, PyTorch, and Transformers versions in `ref.json`.

These checks prove top-1 token identity for this reduced, quantized model and
exercise the real target/DSpark/session paths.  They do not prove logit-level
identity, production-checkpoint performance, every expert/cache residency
policy, or full-model quality.

To exercise the normal runtime without drafting:

```text
./deepseek_v4 deepseek_v4_tiny '<t005><t007><t009>' \
  --raw-prompt --draft-model deepseek_v4_tiny/dspark --no-dspark
```

Source distinction:

- tiny committed fixture: independent `source=transformers` oracle;
- full checkpoint smoke test below: `source=coli-self` consistency oracle.

## Full-checkpoint oracle validation (not in light CI)

Upstream expects targeted, token-level checks. V4 unit tests cover config /
stores / math; full-model correctness is a separate heavy path:

```bash
cd c
make deepseek-v4-oracle MODEL=/path/to/DeepSeek-V4-Flash-DSpark MEMORY_GB=32
# or:
python tools/make_deepseek_v4_oracle.py \
  --model /path/to/DeepSeek-V4-Flash-DSpark \
  --binary ./deepseek_v4.exe \
  --output tests/deepseek_v4_oracle.json \
  --validate --teacher-forcing 32 --greedy 20 --check-dspark
```

```text
./deepseek_v4 MODEL --oracle tests/deepseek_v4_oracle.json \
  --teacher-forcing 32 --greedy 20 --memory-gb 32
```

### Comparison contract

| Check | Criterion |
|-------|-----------|
| Teacher-forcing | top-1 token exact: N/N positions vs `tf_pred` |
| Greedy | top-1 token exact: N/N continuation tokens vs `full_ids` |
| DSpark on/off | greedy token sequences identical (`--check-dspark`) |
| Logits / top-k | reserved for `source=transformers` fixtures |

Fixture `source` field:

- `coli-self` — recorded by the C engine with `--no-dspark` (default when
  Hugging Face DeepSeek V4 is unavailable). Proves reproducibility and that
  speculative decode matches the target greedy path; **not** HF bit-exact.
- `transformers` — official implementation when
  `DeepseekV4ForCausalLM` loads (`--prefer-transformers`).

The full-checkpoint oracle JSON is local/generated and is not required for
`make check`; the committed tiny independent oracle is required there.

## Run

```powershell
cd D:\ai\colibri
python ./c/v4 run --model D:/ai/DeepSeek-V4-Flash-DSpark --ram 32 `
  --stop-sentence "What is the capital of France?"

```

Or call the engine directly:
`.\c\deepseek_v4.exe <model> <prompt> --max-tokens N --memory-gb G …`.

### Options

- `--model PATH` (required) DeepSeek V4 checkpoint directory
- `--ram GiB` planner budget; omit for adaptive OS-available planning
- `--ngen N` max generation tokens (default 128)
- `--stop-sentence` (`run`) stop at the first sentence terminator
- `--system` / `--thinking` (`chat` only)

Chat re-prefills history each turn; cross-turn KV reuse is not implemented yet.
Default non-thinking encoding:

```text
<｜begin▁of▁sentence｜>[system]<｜User｜>[user]<｜Assistant｜></think>
```

## Todo

- [ ] **session parallel-prefix**: wire `COLI_V4_EXPERIMENTAL_PARALLEL_PREFIX_VERIFY`
      into `coli_v4_session_generate` (match legacy CLI verify throughput)
- [ ] **16 GiB performance**: close the decode gap vs 32 GiB (cache, pinning, I/O overlap)
- [ ] **Model quantization**: more aggressive weight/activation paths for footprint and bandwidth
- [ ] **Server**: HTTP / OpenAI-compatible API for multi-client use
- [ ] **CUDA**: optional GPU backend alongside the CPU path
- [ ] **Dynamic DSpark for long outputs**: adapt the verify window between K=2–4
      from recent prefix survival / confidence instead of keeping a fixed K
- [ ] Chat cross-turn KV cache reuse (avoid full re-prefill every turn)
