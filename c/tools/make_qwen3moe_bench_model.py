"""Build a tiny, deterministic Qwen3-MoE fixture to validate qwen3moe.c's GQA
attention + per-head QK-norm before ever touching the real 30B checkpoint.

This is not a useful language model — dimensions are chosen so the engine's
riskiest new code paths are actually exercised: Dq = n_heads*head_dim,
Dkv = n_kv_heads*head_dim and D = hidden_size are all DIFFERENT (384, 96, 256),
mirroring the real Qwen3-30B-A3B shape (4096, 512, 2048) instead of the
degenerate case where they'd accidentally coincide and hide a shape bug.

Usage:
  python tools/make_qwen3moe_bench_model.py --output qwen3moe_bench
  ./qwen3moe 16 8 qwen3moe_bench/ref_qwen3moe.json     # SNAP=qwen3moe_bench implied by ref path's sibling dir
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import glm_fp8_emit when run from c/
from glm_fp8_emit import unfuse_experts


def build_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=256,       # unused: mlp_only_layers=[] means every layer is MoE
        moe_intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=48,                 # 8*48=384 (Dq) != 2*48=96 (Dkv) != 256 (hidden) -- deliberately all distinct
        num_experts=8,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rope_theta=1000000.0,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=False,
        max_position_embeddings=512,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="qwen3moe_bench")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = build_config()
    cfg._attn_implementation = "eager"
    model = Qwen3MoeForCausalLM(cfg).eval()
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() >= 2:
                param.normal_(0, 0.02)
        # q_norm/k_norm are RMSNorm weights (default init to 1.0, which would make
        # the tiny model's norm a no-op and hide an ordering bug) -- perturb them
        # so a norm-before-vs-after-RoPE mistake actually changes the output.
        for layer in model.model.layers:
            layer.self_attn.q_norm.weight.normal_(1.0, 0.1)
            layer.self_attn.k_norm.weight.normal_(1.0, 0.1)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())

    prompt = [3, 14, 159, 26, 53, 58, 200, 11]
    ids = torch.tensor([prompt])
    with torch.inference_mode():
        full = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)[0]
        logits = model(full.unsqueeze(0), use_cache=False).logits[0]

    # This transformers version stores Qwen3MoE experts fused (gate_up_proj [E,2*inter,hidden]
    # + down_proj [E,hidden,inter]) for batched-GEMM efficiency at runtime -- but the REAL
    # Qwen3-30B-A3B checkpoint on HuggingFace stores experts UNFUSED as per-expert 2-D
    # gate_proj/up_proj/down_proj tensors (verified against its model.safetensors.index.json).
    # Unfuse here so the fixture matches the real checkpoint's on-disk tensor naming that
    # convert_qwen3moe.py's EXPERT_KEY_RE and qwen3moe.c both expect.
    sd = model.state_dict()
    unfuse_experts(sd)
    save_file({k: v.contiguous() for k, v in sd.items()}, str(output / "model.safetensors"))
    (output / "config.json").write_text(json.dumps(cfg.to_dict()))

    ref = {
        "prompt_ids": prompt,
        "full_ids": full.cpu().tolist(),
        "tf_pred": logits.argmax(-1).cpu().tolist(),
    }
    (output / "ref_qwen3moe.json").write_text(json.dumps(ref))
    manifest = {
        "seed": args.seed,
        "parameters": params,
        "hidden": cfg.hidden_size, "n_heads": cfg.num_attention_heads,
        "n_kv_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim,
        "purpose": "qwen3moe.c GQA/QK-norm correctness fixture; random weights, not a language model",
    }
    (output / "bench_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"prompt_ids: {prompt}")
    print(f"full_ids  : {full.cpu().tolist()}")


if __name__ == "__main__":
    main()
