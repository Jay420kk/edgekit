"""edgekit - decide whether an LLM fits on a specific machine.

    edgekit probe                       what this machine is
    edgekit audit model.gguf            does it fit, and what to change
    edgekit audit model.gguf --context 8192 --budget 1024
    edgekit compare a.gguf b.gguf c.gguf
    edgekit plan --params 1.7 --quant Q4_K_M

Everything is read from the file and the host. No estimates where a measurement
is available, and every estimate is labelled as one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .gguf import GGUFError, audit, compare, human, parse
from .hardware import probe, recommend_context, tokens_per_second


def _fmt_bytes(n: float) -> str:
    return human(n)


def cmd_probe(a) -> int:
    m = probe()
    d = m.as_dict()
    if a.json:
        print(json.dumps(d, indent=2))
        return 0
    print(f"cpu          {d['cpu']}")
    print(f"cores        {d['cores_physical']} physical / "
          f"{d['cores_logical']} logical")
    print(f"ram          {d['ram_available_mb']:.0f} MB available "
          f"of {d['ram_total_mb']:.0f} MB total"
          + (f", {d['swap_mb']:.0f} MB swap" if d["swap_mb"] else ""))
    print(f"disk         {d['disk_free_mb']:.0f} MB free")
    print(f"gpu          {d['gpu']}"
          + (f" -- {d['gpu_free_mb']:.0f} MB free of {d['gpu_vram_mb']:.0f} MB"
             if d["gpu_vram_mb"] else ""))
    print(f"compute      {d['compute_total_mb']:.0f} MB total for a model")
    present = [k for k, v in d["tools"].items() if v]
    print(f"tools        {', '.join(present) if present else 'none found'}")
    if not present:
        print("\n  llama.cpp is not on PATH. Install it to actually run a model;")
        print("  this tool still gives correct sizing without it.")
    return 0


def cmd_audit(a) -> int:
    try:
        r = audit(a.model, context=a.context, budget_mb=a.budget, batch=a.batch,
                  mmap=not a.no_mmap, full=a.full, compute_mb=a.compute)
    except (GGUFError, OSError) as e:
        print(f"edgekit: {e}", file=sys.stderr)
        return 2

    if a.json:
        print(json.dumps(r, indent=2))
        return 0 if r["fits"] else 1

    print(f"{r['name']}  [{r['arch']}, {r['quant']}]")
    print(f"  file            {r['file_mb']:.0f} MB "
          f"({r['tensors_read']}/{r['tensors_total']} tensors read)")
    print(f"  weights         {r['weights_mb']:.0f} MB")
    if r.get("quant_mix"):
        print("  quant mix       " + ", ".join(
            f"{k} {v['mb']:.0f}MB/{v['tensors']}" for k, v in r["quant_mix"].items()))
    print(f"  architecture    {r['n_layer']}L  {r['n_embd']}d  "
          f"{r['n_head']}q/{r['n_head_kv']}kv heads  dim {r['head_dim']}"
          + (f"  (GQA {r['gqa_ratio']}x)" if r["gqa_ratio"] > 1 else ""))
    print(f"  native context  {r['native_context']}")
    print()
    print(f"  at -c {r['context']}  cache {r['cache_mb']:.0f} MB "
          f"({r['kv_dtype_bytes']}B/elem)  compute ~{r['compute_mb']:.0f} MB")
    print(f"  resident total  {r['resident_mb']:.0f} MB against a "
          f"{r['budget_mb']:.0f} MB budget")
    print()
    for rec in r["recommendations"]:
        print(f"  - {rec}")

    if r["curve"]:
        print("\n  KV cache by context:")
        for p in r["curve"]:
            need = r["weights_mb"] * 0.6 + p["cache_mb"] + r["compute_mb"]
            mark = "fits" if need <= r["budget_mb"] else "over"
            print(f"    {p['context']:>6}  {p['cache_mb']:>7.0f} MB cache  "
                  f"{need:>7.0f} MB resident  {mark}")
    return 0 if r["fits"] else 1


def cmd_compare(a) -> int:
    rows = compare({Path(p).stem: p for p in a.models},
                   context=a.context, budget_mb=a.budget)
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{'model':<34} {'file':>9} {'cache':>9} {'resident':>10}  verdict")
    for r in rows:
        if "error" in r:
            print(f"{r['quant']:<34} {'-':>9} {'-':>9} {'-':>10}  {r['error']}")
            continue
        print(f"{r['quant'][:33]:<34} {r['file_mb']:>8.0f}M "
              f"{r['cache_mb']:>8.0f}M {r['resident_mb']:>9.0f}M  "
              f"{'fits' if r['fits'] else 'DOES NOT FIT'}")
    return 0


def cmd_plan(a) -> int:
    """Sizing without a model file: what to look for on a given budget."""
    m = probe()
    budget = a.budget or m.compute_total_mb
    tps = tokens_per_second(a.params, a.quant, m)
    print(f"budget        {budget:.0f} MB "
          f"({'detected' if not a.budget else 'specified'})")
    print(f"target        {a.params}B params at {a.quant}")
    print(f"weights       ~{a.params * {'Q4_K_M': 4.5, 'Q5_K_M': 5.5, 'Q6_K': 6.5625, 'Q8_0': 8.5}.get(a.quant, 4.5) * 1e9 / 8 / 1024 / 1024:.0f} MB")
    print(f"decode        ~{tps['tok_s']} tok/s {tps.get('range', '')} "
          f"({tps['confident'] and 'measured' or 'estimate'})")
    print(f"              {tps['why']}")
    print()
    print("  Candidates worth trying at this budget, largest first:")
    for params, name in ((3.0, "llama-3.2-3b"), (1.7, "qwen3-1.7b"),
                         (1.5, "deepseek-r1-distill-qwen-1.5b"),
                         (1.1, "qwen2.5-coder-1.5b"), (1.0, "llama-3.2-1b"),
                         (0.6, "smollm2-360m"), (0.36, "qwen3-0.6b")):
        w = params * 4.5e9 / 8 / 1024 / 1024
        if w + 800 < budget:
            print(f"    {name:<32} {w:.0f} MB weights  +800 MB scratch")
    print()
    print("  Then audit the actual file before committing:")
    print("    edgekit audit model.gguf --context 4096 --budget "
          f"{budget:.0f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="edgekit",
                                description="Decide whether an LLM fits on a specific machine.")
    p.add_argument("--version", action="version", version=f"edgekit {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("probe", help="measure this machine")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(fn=cmd_probe)

    au = sub.add_parser("audit", help="audit a GGUF file against a budget")
    au.add_argument("model")
    au.add_argument("--context", type=int, default=4096)
    au.add_argument("--budget", type=float, default=1024, help="MB available")
    au.add_argument("--batch", type=int, default=1)
    au.add_argument("--compute", type=float, default=800.0,
                    help="MB of activation scratch to assume")
    au.add_argument("--no-mmap", action="store_true")
    au.add_argument("--full", action="store_true",
                    help="read every tensor for the quantisation mix")
    au.add_argument("--json", action="store_true")
    au.set_defaults(fn=cmd_audit)

    cm = sub.add_parser("compare", help="audit several models side by side")
    cm.add_argument("models", nargs="+")
    cm.add_argument("--context", type=int, default=4096)
    cm.add_argument("--budget", type=float, default=1024)
    cm.add_argument("--json", action="store_true")
    cm.set_defaults(fn=cmd_compare)

    pl = sub.add_parser("plan", help="sizing from a target, no file needed")
    pl.add_argument("--params", type=float, default=1.7, help="billions")
    pl.add_argument("--quant", default="Q4_K_M")
    pl.add_argument("--budget", type=float, default=0)
    pl.set_defaults(fn=cmd_plan)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
