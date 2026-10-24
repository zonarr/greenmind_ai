"""GreenMind: make a W8A8 INT8 copy of MANGO1.5-Qwen3.5-9B for vLLM.

Why
  At the Final's peak the A100 is compute-bound on prompt reading. Its INT8
  tensor cores do about 2x the math per second of BF16, and vLLM runs
  "W8A8 INT8" checkpoints (compressed-tensors format) with CUTLASS INT8 kernels
  on the A100. AWQ (W4A16) only shrinks the weights; its math still runs in
  16-bit, so it does not speed up big prefill batches.

What it does (CPU only, no GPU, about 10 minutes, about 14 GB of disk)
  * Reads the official BF16 model from the Hugging Face cache.
  * Quantizes the big Linear layers of the language model and of the MTP head
    (attention q/k/v/o, GatedDeltaNet in_proj_qkv / in_proj_z / out_proj,
    MLP gate/up/down) to INT8 weights with one scale per output channel
    (round to nearest, symmetric). Activations are quantized per token at run
    time by vLLM (dynamic), so no calibration data is needed.
  * Leaves everything else in BF16: embeddings, lm_head, norms, the GDN
    conv1d and in_proj_a/b, mtp.fc and the vision tower.
  * Writes a normal model folder with a compressed-tensors quantization_config
    in config.json, which `vllm serve <folder>` picks up by itself.

Usage (inside the venv; the server can keep running):
  python make_w8a8.py                       # -> ~/models/MANGO1.5-Qwen3.5-9B-W8A8
  python make_w8a8.py --keep down_proj      # also leave down_proj in BF16
                                            # (use if accuracy drops too much)
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = "CMKL/MANGO1.5-Qwen3.5-9B"
QUANT_LAYERS = [
    "q_proj", "k_proj", "v_proj", "o_proj",          # full attention
    "in_proj_qkv", "in_proj_z", "out_proj",          # GatedDeltaNet
    "gate_proj", "up_proj", "down_proj",             # MLP
]
QUANT_PREFIXES = ("model.language_model.", "mtp.")


def find_source(path: str | None) -> str:
    if path:
        return os.path.expanduser(path)
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(REPO, local_files_only=True)
    except Exception:
        pass
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--CMKL--MANGO1.5-Qwen3.5-9B/snapshots/*"
    )
    found = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not found:
        sys.exit(f"BF16 model not found in the Hugging Face cache ({pattern})")
    return found[-1]


def should_quantize(name: str, tensor: torch.Tensor, layers: list[str]) -> bool:
    if tensor.ndim != 2 or not name.endswith(".weight"):
        return False
    if not name.startswith(QUANT_PREFIXES):
        return False
    module = name[: -len(".weight")].rsplit(".", 1)[-1]
    return module in layers


def quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel INT8, as vLLM's W8A8 INT8 scheme expects."""
    w = weight.to(torch.float32)
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.to(torch.float32)


def quantization_config(layers: list[str]) -> dict:
    target = r"re:.*\.(" + "|".join(layers) + r")$"
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "targets": [target],
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
            }
        },
        "ignore": ["lm_head"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", help="BF16 model folder (default: HF cache)")
    ap.add_argument("--out", default="~/models/MANGO1.5-Qwen3.5-9B-W8A8")
    ap.add_argument("--keep", default="",
                    help="comma-separated layer names to leave in BF16, e.g. down_proj")
    args = ap.parse_args()

    keep = {k.strip() for k in args.keep.split(",") if k.strip()}
    unknown = keep - set(QUANT_LAYERS)
    if unknown:
        sys.exit(f"--keep: unknown layer names {sorted(unknown)}; choose from {QUANT_LAYERS}")
    layers = [n for n in QUANT_LAYERS if n not in keep]

    src = find_source(args.src)
    out = os.path.expanduser(args.out)
    shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
    if not shards:
        sys.exit(f"no .safetensors files in {src}")
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"{out} already exists and is not empty; delete it or pick --out")
    os.makedirs(out, exist_ok=True)

    need = sum(os.path.getsize(os.path.realpath(s)) for s in shards)
    free = shutil.disk_usage(out).free
    print(f"source {src}\noutput {out}\nquantizing: {', '.join(layers)}")
    print(f"BF16 shards {need / 1e9:.1f} GB, free disk {free / 1e9:.1f} GB")
    if free < need * 0.8:
        sys.exit("not enough free disk space")

    weight_map: dict[str, str] = {}
    n_quant = n_kept = 0
    for i, shard in enumerate(shards, 1):
        base = os.path.basename(shard)
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if should_quantize(name, t, layers):
                    q, s = quantize(t)
                    scale_name = name[: -len("weight")] + "weight_scale"
                    tensors[name] = q
                    tensors[scale_name] = s
                    weight_map[name] = base
                    weight_map[scale_name] = base
                    n_quant += 1
                else:
                    tensors[name] = t.contiguous()
                    weight_map[name] = base
                    n_kept += 1
        save_file(tensors, os.path.join(out, base), metadata={"format": "pt"})
        print(f"  [{i}/{len(shards)}] {base}: done", flush=True)
        del tensors

    if n_quant == 0:
        sys.exit("no layers matched; the checkpoint layout is not what I expected")

    # Everything that is not a weight file: tokenizer, chat template, etc.
    for path in glob.glob(os.path.join(src, "*")):
        name = os.path.basename(path)
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        if os.path.isfile(path):
            shutil.copy(path, os.path.join(out, name))  # follows HF cache symlinks

    total = sum(os.path.getsize(os.path.join(out, os.path.basename(s))) for s in shards)
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=2)

    cfg_path = os.path.join(out, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = quantization_config(layers)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"quantized {n_quant} weight matrices, kept {n_kept} tensors as they were")
    print(f"new model: {out} ({total / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
