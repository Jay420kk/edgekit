"""Tests for edgekit.

The parser is exercised against GGUF files written by hand, because the whole
tool's value is that it reads a real file correctly and a mock would happily
agree with a wrong implementation.

The 800 MB compute reserve is a real design choice and is tested as one: it is
by far the largest term in a 1 GB budget for a small model, and a tool that
ignores it will confidently certify a deployment that OOM-kills.
"""
from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgekit.gguf import (  # noqa: E402
    GGML_BITS, GGUFError, audit, human, parse,
)
from edgekit.hardware import Machine, recommend_context, tokens_per_second  # noqa: E402


# --------------------------------------------------------------------------
# a GGUF writer, so tests use real files
# --------------------------------------------------------------------------

def _wstr(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def write_gguf(path: Path, meta: dict, tensors: list[tuple[str, list[int], int]],
               version: int = 3, dim_bytes: int = 4, file_type=2) -> Path:
    """Minimal but valid GGUF. `dim_bytes=8` writes the non-spec variant."""
    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", version)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(meta))
    for k, v in meta.items():
        out += _wstr(k)
        if isinstance(v, str):
            out += struct.pack("<I", 8) + _wstr(v)
        elif isinstance(v, float):
            out += struct.pack("<I", 6) + struct.pack("<f", v)
        else:
            out += struct.pack("<I", 4) + struct.pack("<i", v)
    off = 0
    for name, dims, dtype in tensors:
        out += _wstr(name)
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<I" if dim_bytes == 4 else "<Q", d)
        out += struct.pack("<I", dtype)
        out += struct.pack("<Q", off)
        n = 1
        for d in dims:
            n *= d
        off += int(n * GGML_BITS.get(dtype, 8.5) / 8)
    out += b"\0" * (off + 32)          # payload stand-in
    path.write_bytes(bytes(out))
    return path


LLAMA_META = {
    "general.architecture": "llama",
    "general.name": "test-1b",
    "llama.block_count": 22,
    "llama.embedding_length": 2048,
    "llama.attention.head_count": 32,
    "llama.attention.head_count_kv": 8,
    "llama.context_length": 8192,
    "general.file_type": 15,
}


def test_parses_a_spec_compliant_file():
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "m.gguf", LLAMA_META,
                       [("tok.embed", [32000, 2048], 12),
                        ("blk.0.attn_q", [2048, 2048], 12),
                        ("blk.0.attn_k", [512, 2048], 12),
                        ("blk.0.attn_v", [512, 2048], 12)])
        g = parse(p)
    assert g.arch == "llama"
    assert g.n_layer == 22
    assert g.n_embd == 2048
    assert g.n_head == 32
    assert g.n_head_kv == 8
    assert g.head_dim == 64
    assert g.context_length == 8192
    assert g.quant_label == "Q4_K_M"
    assert not g.dims_are_u64
    assert len(g.tensors) == 4


def test_reads_a_non_llama_architecture():
    """The bug this file found.

    GGUF namespaces metadata under the architecture name, so a Qwen2 model
    writes `qwen2.block_count`. Every tool that hard-codes the `llama.` prefix
    -- which is most of them, because llama.cpp defined the format -- reads
    zero layers for every non-Llama model, computes a zero-byte KV cache, and
    reports a comfortable fit for a model that cannot run.
    """
    meta = {
        "general.architecture": "qwen2",
        "general.name": "qwen3-1.7b",
        "qwen2.block_count": 28,
        "qwen2.embedding_length": 2048,
        "qwen2.attention.head_count": 28,
        "qwen2.attention.head_count_kv": 4,
        "qwen2.context_length": 32768,
    }
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "q.gguf", meta,
                       [("tok_embd.weight", [151936, 2048], 12)])
        g = parse(p)
    assert g.n_layer == 28, "architecture-prefixed keys must be found"
    assert g.n_head_kv == 4
    assert g.kv_bytes(4096) > 0, "a zero layer count would give a zero cache"


def test_handles_a_custom_architecture_prefix():
    meta = {
        "general.architecture": "genesis",
        "general.name": "genesis-15m",
        "general.file_type": "q4_0",
        "genesis.context_length": 2048,
        "genesis.embedding_length": 192,
        "genesis.attention.head_count": 4,
        "genesis.block_count": 3,
    }
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "g.gguf", meta,
                       [("perception/token_embed/weight", [50257, 192], 2)])
        g = parse(p)
    assert g.arch == "genesis"
    assert g.n_layer == 3 and g.n_embd == 192
    assert g.quant_label == "Q4_0"


def test_file_type_may_be_a_string():
    """Some converters write general.file_type as text. int() on that raises
    and takes the whole file with it, over a label."""
    meta = dict(LLAMA_META, **{"general.file_type": "q4_0"})
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "m.gguf", meta, [("w", [4, 4], 2)],
                       file_type="q4_0")
        g = parse(p)
    assert g.quant_label == "Q4_0"


def test_label_falls_back_to_the_tensors_when_file_type_is_absent():
    """Not every exporter writes general.file_type.

    Falling back to a raw int is worse than useless: 'type -1' reads like a
    finding about the model rather than a gap in the file's metadata, and it
    hides a perfectly good Q4_0 label that the tensor table already knows.
    """
    meta = {k: v for k, v in LLAMA_META.items() if k != "general.file_type"}
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "m.gguf", meta, [("w", [4, 4], 2)])
        g = parse(p)
    assert g.quant_label == "Q4_0", "should come from the tensor dtype"


def test_non_spec_uint64_dimensions_are_detected():
    """A third-party exporter writes uint64 dimensions against a uint32 spec.

    The symptom is not an error message, which is what makes it dangerous. The
    parse desynchronises and reports plausible garbage: one real dimension
    beside a zero, then a zero-length tensor name, then an absurd dimension
    count. Detecting it per-entry is not enough either -- a 1-D tensor's first
    four bytes look identical under either width -- so the whole table has to
    validate.
    """
    meta = {"general.architecture": "genesis", "genesis.block_count": 3,
            "general.name": "u64"}
    tensors = [("perception/token_embed/weight", [50257, 192], 2),
               ("perception/embed_norm/weight", [192], 2),
               ("perception/complexity_estimator/0/weight", [96, 192], 2),
               ("perception/complexity_estimator/0/bias", [96], 2)]
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "u64.gguf", meta, tensors, dim_bytes=8)
        g = parse(p)
    assert g.dims_are_u64, "uint64 dimensions should be detected"
    assert len(g.tensors) == 4
    assert g.tensors[0].dims == [50257, 192]
    assert g.tensors[1].dims == [192], "1-D tensors are the ones that break " \
                                      "per-entry detection"
    assert all(t.name for t in g.tensors), "no desynchronised names"
    assert g.n_layer == 3


def test_rejects_a_non_gguf_file():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.gguf"
        p.write_bytes(b"NOTGGUF" + b"\0" * 64)
        try:
            parse(p)
        except GGUFError as e:
            assert "not a GGUF" in str(e)
        else:
            raise AssertionError("should have raised")


# --------------------------------------------------------------------------
# KV cache arithmetic
# --------------------------------------------------------------------------

def test_kv_cache_matches_the_formula():
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "m.gguf", LLAMA_META, [("w", [8, 8], 12)])
        g = parse(p)
    # 2 (K+V) * 22 layers * 8 kv heads * 64 dim * 4096 ctx * 2 bytes
    expect = 2 * 22 * 8 * 64 * 4096 * 2
    assert g.kv_bytes(4096) == expect


def test_cache_scales_linearly_with_context():
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "m.gguf", LLAMA_META, [("w", [8, 8], 12)])
        g = parse(p)
    assert g.kv_bytes(8192) == 2 * g.kv_bytes(4096)
    assert g.kv_bytes(16384) == 4 * g.kv_bytes(4096)


def test_grouped_query_attention_shrinks_the_cache():
    """The one structural change that reduces the cache without making the
    model worse. 32 query heads over 8 KV heads is 4x smaller than the head
    count implies, and anyone budgeting from n_head overestimates by 4x."""
    meta = dict(LLAMA_META, **{"llama.attention.head_count_kv": 32})
    with tempfile.TemporaryDirectory() as d:
        mha = parse(write_gguf(Path(d) / "mha.gguf", meta, [("w", [8, 8], 12)]))
        gqa = parse(write_gguf(Path(d) / "gqa.gguf", LLAMA_META, [("w", [8, 8], 12)]))
    assert gqa.kv_bytes(4096) * 4 == mha.kv_bytes(4096)


def test_missing_kv_head_count_falls_back_to_query_heads():
    """A model without GQA writes no head_count_kv. Returning 0 there is a
    silent zero-byte cache, which reads as 'tiny model, fits easily'."""
    meta = {k: v for k, v in LLAMA_META.items()
            if k != "llama.attention.head_count_kv"}
    with tempfile.TemporaryDirectory() as d:
        g = parse(write_gguf(Path(d) / "m.gguf", meta, [("w", [8, 8], 12)]))
    assert g.n_head_kv == g.n_head == 32
    assert g.kv_bytes(4096) > 0


# --------------------------------------------------------------------------
# the budget
# --------------------------------------------------------------------------

def test_compute_reserve_dominates_a_small_model_budget():
    """The number people forget.

    A 15M-param model has an 8 MB file and a 4 MB cache at 2k context. On a
    1 GB device it is not the model that decides the fit -- it is the ~800 MB
    of activation scratch llama.cpp allocates, which scales with batch and not
    with context at all.
    """
    meta = {"general.architecture": "genesis", "genesis.block_count": 3,
            "general.name": "tiny", "genesis.embedding_length": 192,
            "genesis.attention.head_count": 4, "general.file_type": "q4_0"}
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "t.gguf", meta, [("w", [50257, 192], 2)])
        a = audit(p, context=4096, budget_mb=1024, compute_mb=800)
    assert a["resident_mb"] > 700, "scratch should dominate for a tiny model"
    assert a["cache_mb"] < 10


def test_a_too_small_budget_is_reported_as_not_fitting():
    meta = dict(LLAMA_META)
    tensors = [(f"blk.{i}.w", [2048, 2048], 12) for i in range(22)]
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "m.gguf", meta, tensors)
        a = audit(p, context=8192, budget_mb=256, compute_mb=800)
    assert not a["fits"]
    assert a["headroom_mb"] < 0


def test_mmap_reduces_the_reported_resident_size():
    """With mmap the weights are demand-paged. Without it the whole file is
    resident, and a budget computed under mmap is simply wrong."""
    meta = {"general.architecture": "llama", "llama.block_count": 22,
            "general.name": "x", "llama.embedding_length": 2048,
            "llama.attention.head_count": 32, "general.file_type": 15}
    tensors = [(f"blk.{i}.w", [2048, 2048], 12) for i in range(22)]
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "m.gguf", meta, tensors)
        mm = audit(p, budget_mb=4096, mmap=True)
        no = audit(p, budget_mb=4096, mmap=False)
    assert no["resident_mb"] > mm["resident_mb"]


def test_q4_k_is_bigger_than_four_bits_per_weight():
    """Block quantisation carries per-superblock scales and minimums.

    Anyone budgeting a Q4_K_M file as '4 bits per weight' underestimates by
    12.5%, which on a 1 GB device is the difference between fitting and not.
    """
    assert GGML_BITS[12] == 4.5
    assert GGML_BITS[14] == 6.5625
    assert GGML_BITS[8] == 8.5


def test_weights_are_summed_from_the_tensor_table():
    """Declared shapes, not a nominal bit depth, so the number is what is
    actually on disk."""
    tensors = [("a", [1000, 1000], 12), ("b", [1000, 1000], 14)]
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "m.gguf", LLAMA_META, tensors)
        g = parse(p)
    # 1e6 * 4.5/8 + 1e6 * 6.5625/8 bytes
    assert g.weights_bytes() == int(1e6 * 4.5 / 8) + int(1e6 * 6.5625 / 8)


def test_quant_bits_measured_from_tensors():
    tensors = [("a", [1000, 1000], 12), ("b", [1000, 1000], 14)]
    with tempfile.TemporaryDirectory() as d:
        g = parse(write_gguf(Path(d) / "m.gguf", LLAMA_META, tensors))
    assert abs(g.quant_bits() - (4.5 + 6.5625) / 2) < 0.01


def test_audit_recommends_a_smaller_context_only_when_it_does_not_fit():
    """Recommending a smaller context to a model that already fits teaches the
    reader to ignore the recommendations."""
    meta = {"general.architecture": "genesis", "genesis.block_count": 3,
            "general.name": "tiny", "genesis.embedding_length": 192,
            "genesis.attention.head_count": 4, "general.file_type": "q4_0"}
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "t.gguf", meta, [("w", [50257, 192], 2)])
        fits = audit(p, context=4096, budget_mb=4096)
        # Budget above the 800 MB scratch reserve but below scratch + weights
        # + cache, so context is genuinely the binding constraint and capping
        # is the correct free fix.
        over = audit(p, context=4096, budget_mb=806, compute_mb=800)
    assert "Capping context" not in " ".join(fits["recommendations"])
    assert "Capping context" in " ".join(over["recommendations"])


def test_audit_says_when_context_cannot_help():
    """Scratch alone busts the budget, so no -c value fixes it. Recommending a
    smaller context here sends the reader off to try -c 128 for nothing."""
    meta = {"general.architecture": "genesis", "genesis.block_count": 3,
            "general.name": "tiny", "genesis.embedding_length": 192,
            "genesis.attention.head_count": 4, "general.file_type": "q4_0"}
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "t.gguf", meta, [("w", [50257, 192], 2)])
        a = audit(p, context=4096, budget_mb=200, compute_mb=800)
    joined = " ".join(a["recommendations"])
    assert "No context value fixes this" in joined
    assert "Capping context" not in joined


def test_audit_warns_when_context_exceeds_native():
    meta = dict(LLAMA_META, **{"llama.context_length": 4096})
    with tempfile.TemporaryDirectory() as d:
        p = write_gguf(Path(d) / "m.gguf", meta, [("w", [8, 8], 12)])
        a = audit(p, context=32768, budget_mb=8192)
    assert any("native" in r for r in a["recommendations"])


# --------------------------------------------------------------------------
# sizing without a file
# --------------------------------------------------------------------------

def test_recommend_context_inverts_the_cache_formula():
    # 2 (K+V) * 22 layers * 8 kv heads * 64 head dim * 2 bytes = 45056 B/token.
    # 1024 MB of pure cache is 23814 tokens, which rounds down to a 1024
    # boundary at 23552 -- and must not round up past the budget, which is the
    # whole point of the function.
    ctx = recommend_context(1024, 22, 8, 64, reserve_mb=0)
    assert ctx == 23552
    assert ctx % 1024 == 0
    assert 2 * 22 * 8 * 64 * ctx * 2 <= 1024 * 1024 * 1024
    assert 2 * 22 * 8 * 64 * (ctx + 1024) * 2 > 1024 * 1024 * 1024


def test_recommend_context_leaves_room_for_the_reserve():
    """The reserve is the whole reason this function exists. Ignoring it
    returns a context that is arithmetically right and operationally wrong."""
    bare = recommend_context(1024, 22, 8, 64, reserve_mb=0)
    reserved = recommend_context(1024, 22, 8, 64, reserve_mb=800)
    assert reserved < bare
    assert 800 * 1024 * 1024 + 2 * 22 * 8 * 64 * reserved * 2 <= 1024 * 1024 * 1024


def test_recommend_context_returns_zero_when_there_is_no_room():
    assert recommend_context(100, 22, 8, 64, reserve_mb=800) == 0


def test_throughput_estimate_is_labelled_as_an_estimate():
    """A single tok/s number here is a lie. Decode is bandwidth-bound, so the
    answer depends on the machine in ways that resist a formula, and a confident
    wrong number is worse than a labelled range."""
    cpu = Machine(cpu="i7-5600U", cores_physical=2, cores_logical=4,
                  ram_total_mb=7800, ram_available_mb=3500)
    t = tokens_per_second(1.7, "Q4_K_M", cpu)
    assert not t["confident"]
    assert "estimate" in t["why"] or "bandwidth" in t["why"]
    assert t["range"][0] < t["tok_s"] < t["range"][1]


def test_throughput_reports_zero_when_the_model_exceeds_vram():
    gpu = Machine(gpu="GeForce 840M", gpu_vram_mb=2048, gpu_free_mb=100,
                  cores_physical=2)
    t = tokens_per_second(70, "Q4_K_M", gpu)
    assert t["tok_s"] == 0
    assert not t["confident"]


def test_smaller_quantisation_is_faster():
    """The relationship that does hold: decode streams the whole model per
    token, so time is roughly proportional to model bytes."""
    cpu = Machine(cores_physical=4, ram_available_mb=8000)
    q2 = tokens_per_second(7.0, "Q2_K", cpu)["tok_s"]
    q8 = tokens_per_second(7.0, "Q8_0", cpu)["tok_s"]
    assert q2 > q8


def test_human_formatting():
    assert human(512) == "512 B"
    assert human(1536) == "1.5 KB"
    assert human(8 * 1024 * 1024) == "8.0 MB"


if __name__ == "__main__":
    import traceback
    fns = [(n, o) for n, o in sorted(globals().items())
           if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception:
            failed += 1
            print(f"  FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
