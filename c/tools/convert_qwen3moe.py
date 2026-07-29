#!/usr/bin/env python3
"""Convert a Qwen3-MoE (e.g. Qwen/Qwen3-30B-A3B) HuggingFace checkpoint to
colibri's merged int4/int8 format, readable by qwen3moe.c.

DISK-SAFE strategy (same discipline as convert_fp8_to_int4.py's GLM converter):
with --repo, download ONE source shard at a time, extract whatever it holds,
delete it, move to the next. Peak extra disk usage is one source shard (a few
GB) plus the growing quantized output -- never the full ~60GB bf16 checkpoint.

Wrinkle vs. the GLM converter: colibri merges each expert's gate_proj/up_proj/
down_proj into a single blob (one pread instead of three at inference time),
but HuggingFace shards experts by parameter-count balance, not by expert
identity -- a given expert's three projections can land in three different
shard files. So instead of converting each shard fully independently, this
script accumulates partial experts across shards (kept in RAM only until their
triple completes -- a few MB each) and flushes a completed expert as soon as
its last projection arrives; the input shard is deleted as soon as everything
needed from it has been read, regardless of whether it completed any experts.

q_norm.weight / k_norm.weight (per-head RMSNorm, shape [head_dim]) pass through
UNQUANTIZED, exactly like input_layernorm.weight -- they must never be caught
by the expert regex or run through quantize_row.

Resumable: every output shard already on disk is scanned (header only, via
safe_open) before starting, and any tensor already present there is skipped —
rerunning after an interruption only redoes missing work, like
convert_fp8_to_int4.py's out-NNNNN.safetensors resume.

Usage:
  # local/pre-downloaded checkpoint (also what the tiny bench fixture uses)
  python tools/convert_qwen3moe.py --model ./qwen3moe_bench --out ./qwen3moe_bench_i8 --bits 8

  # real checkpoint: streams+deletes shard by shard, resumable
  python tools/convert_qwen3moe.py --repo Qwen/Qwen3-30B-A3B --out /nvme/qwen3moe_i4 --bits 4
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as exc:
    sys.exit(f"Missing dependencies: {exc}. Install: pip install torch safetensors")

EXPERT_KEY_RE = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
)
# tensors an inference engine never needs: training-only bookkeeping, if present.
SKIP_KEY_RE = re.compile(r"\.e_score_correction_bias$|^model\.rotary_emb\.")

TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "generation_config.json",
)


def quantize_row(w: "torch.Tensor", bits: int) -> tuple:
    """Row-wise symmetric quantization, identical math to qwen3moe.c's
    quantize_rows(): scale = amax(|row|)/qmax, q = round(w/scale)."""
    qmax = (1 << (bits - 1)) - 1
    w_f32 = w.float()
    row_max = w_f32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scales = row_max / qmax
    q = (w_f32 / scales).round().clamp(-qmax - 1, qmax).to(torch.int8)
    return q, scales.squeeze(1)


def free_gb(path) -> float:
    return shutil.disk_usage(path).free / 1e9


def scan_existing_output(out_dir: Path) -> set:
    """Tensor names already written to a *.safetensors file in out_dir --
    header-only scan (safe_open doesn't load tensor data), so this is cheap
    even against many GB of already-converted shards."""
    done = set()
    for shard in sorted(out_dir.glob("model-*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            done.update(f.keys())
    return done


class OutputWriter:
    """Buffers converted tensors and flushes them into new model-NNNNN.safetensors
    shard files once the buffer gets large -- qwen3moe.c's st.h indexes every
    *.safetensors file in a directory by scanning each one's own header, so any
    number of arbitrarily-named shards works with no index.json needed."""

    def __init__(self, out_dir: Path, flush_every: int = 512):
        self.out_dir = out_dir
        self.flush_every = flush_every
        self.buf = {}
        self.shard_idx = len(list(out_dir.glob("model-*.safetensors")))

    def add(self, name: str, tensor: "torch.Tensor"):
        self.buf[name] = tensor.contiguous()
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        fn = self.out_dir / f"model-{self.shard_idx:05d}.safetensors"
        save_file(self.buf, str(fn))
        print(f"  wrote {fn.name} ({len(self.buf)} tensors)", flush=True)
        self.buf = {}
        self.shard_idx += 1


def convert_expert(gate, up, down, bits):
    q_gate, s_gate = quantize_row(gate, bits)
    q_up, s_up = quantize_row(up, bits)
    q_down, s_down = quantize_row(down, bits)
    merged_q = torch.cat([q_gate.flatten(), q_up.flatten(), q_down.flatten()])
    merged_s = torch.cat([s_gate, s_up, s_down])
    return merged_q, merged_s


def copy_metadata(src_dir: Path, out_dir: Path):
    shutil.copy2(src_dir / "config.json", out_dir / "config.json")
    for fn in TOKENIZER_FILES:
        p = src_dir / fn
        if p.is_file():
            shutil.copy2(p, out_dir / fn)


# ---------- local / pre-downloaded path (also used by the tiny bench fixture) ----------

def convert_local(src_dir: Path, out_dir: Path, bits: int, flush_every: int):
    copy_metadata(src_dir, out_dir)
    shards = sorted(src_dir.glob("*.safetensors"))
    if not shards:
        sys.exit(f"No safetensors found in {src_dir}")

    name_to_shard = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for name in f.keys():
                name_to_shard[name] = shard

    done = scan_existing_output(out_dir)
    writer = OutputWriter(out_dir, flush_every)
    handles = {}

    def get(name):
        shard = name_to_shard[name]
        h = handles.get(shard)
        if h is None:
            h = safe_open(str(shard), framework="pt")
            handles[shard] = h
        return h.get_tensor(name)

    experts = {}
    dense_names = []
    for name in name_to_shard:
        if SKIP_KEY_RE.search(name):
            continue
        m = EXPERT_KEY_RE.match(name)
        if m:
            layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)
            experts.setdefault((layer, expert), {})[proj] = name
        else:
            dense_names.append(name)

    n_skip_dense = sum(1 for n in dense_names if n in done)
    for name in dense_names:
        if name in done:
            continue
        writer.add(name, get(name))
    writer.flush()

    n_total = len(experts)
    n_done = 0
    n_skip_experts = 0
    for (layer, expert), projs in sorted(experts.items()):
        mw = f"model.layers.{layer}.mlp.experts.{expert}.merged_weight"
        if mw in done:
            n_done += 1
            n_skip_experts += 1
            continue
        if not all(k in projs for k in ("gate_proj", "up_proj", "down_proj")):
            sys.exit(f"Missing projection for layer {layer} expert {expert}: have {list(projs)}")
        gate, up, down = get(projs["gate_proj"]), get(projs["up_proj"]), get(projs["down_proj"])
        merged_q, merged_s = convert_expert(gate, up, down, bits)
        writer.add(mw, merged_q)
        writer.add(f"model.layers.{layer}.mlp.experts.{expert}.qs", merged_s)
        n_done += 1
        if n_done % 100 == 0 or n_done == n_total:
            print(f"  {n_done}/{n_total} experts converted...", flush=True)
    writer.flush()
    print(f"Done: {n_total} experts, {len(dense_names)} dense tensors "
          f"({n_skip_dense} dense + {n_skip_experts} experts already present, skipped).")


# ---------- streaming --repo path: one source shard downloaded/converted/deleted at a time ----------

def convert_streaming(repo: str, out_dir: Path, bits: int, flush_every: int, min_free_gb: float):
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().repo_info(repo, files_metadata=True)
    shard_files = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".safetensors"))
    if not shard_files:
        sys.exit(f"No .safetensors shards found in {repo}")

    for fn in ("config.json",) + TOKENIZER_FILES:
        try:
            p = hf_hub_download(repo, fn, local_dir=str(out_dir) + "/_meta")
            shutil.copy2(p, out_dir / fn)
        except Exception:
            pass
    if not (out_dir / "config.json").is_file():
        sys.exit(f"config.json could not be fetched from {repo}")

    done = scan_existing_output(out_dir)
    writer = OutputWriter(out_dir, flush_every)
    # (layer, expert) -> {proj: tensor}; entries here are only the projections
    # whose sibling shard(s) haven't been seen yet -- freed the moment a triple
    # completes and gets flushed, so this never approaches full-checkpoint size.
    pending_experts = {}

    tmp = out_dir / "_inflight"
    tmp.mkdir(exist_ok=True)

    for si, shard_name in enumerate(shard_files):
        if free_gb(out_dir) < min_free_gb:
            print(f"STOP: free space below {min_free_gb} GB. Free space and rerun to resume.")
            break
        print(f"[{si+1}/{len(shard_files)}] downloading {shard_name} "
              f"({free_gb(out_dir):.0f} GB free)...", flush=True)
        local_path = hf_hub_download(repo, shard_name, local_dir=str(tmp))

        with safe_open(local_path, framework="pt") as f:
            for name in f.keys():
                if SKIP_KEY_RE.search(name) or name in done:
                    continue
                m = EXPERT_KEY_RE.match(name)
                if not m:
                    writer.add(name, f.get_tensor(name))
                    continue
                layer, expert, proj = int(m.group(1)), int(m.group(2)), m.group(3)
                key = (layer, expert)
                mw = f"model.layers.{layer}.mlp.experts.{expert}.merged_weight"
                if mw in done:
                    continue
                slot = pending_experts.setdefault(key, {})
                slot[proj] = f.get_tensor(name)
                if all(k in slot for k in ("gate_proj", "up_proj", "down_proj")):
                    merged_q, merged_s = convert_expert(
                        slot["gate_proj"], slot["up_proj"], slot["down_proj"], bits)
                    writer.add(mw, merged_q)
                    writer.add(f"model.layers.{layer}.mlp.experts.{expert}.qs", merged_s)
                    del pending_experts[key]

        Path(local_path).unlink(missing_ok=True)  # delete the source shard now -- disk-safe
        print(f"    -> extracted+deleted {shard_name} "
              f"({len(pending_experts)} experts still incomplete)", flush=True)

    writer.flush()
    shutil.rmtree(tmp, ignore_errors=True)
    if pending_experts:
        print(f"WARNING: {len(pending_experts)} experts never completed all 3 projections "
              f"(missing shards, or run stopped early on low disk space) -- rerun to resume.")
    else:
        print("DONE.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="HuggingFace repo ID, e.g. Qwen/Qwen3-30B-A3B "
                                     "(streams+deletes one shard at a time)")
    src.add_argument("--model", help="Local HF checkpoint directory (already fully downloaded)")
    ap.add_argument("--out", required=True, help="Output directory for the colibri container")
    ap.add_argument("--bits", type=int, default=8, choices=range(2, 9), metavar="[2-8]",
                     help="quantization bits for expert weights (default 8; use 4 for the ~15GB "
                          "real-checkpoint footprint once int8 has validated correctness)")
    ap.add_argument("--flush-every", type=int, default=512,
                     help="flush an output shard every N converted tensors (bounds RAM/disk-buffer)")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                     help="--repo only: stop (resumably) if free space drops below this")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        convert_local(Path(args.model), out_dir, args.bits, args.flush_every)
    else:
        convert_streaming(args.repo, out_dir, args.bits, args.flush_every, args.min_free_gb)


if __name__ == "__main__":
    main()
