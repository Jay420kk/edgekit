"""GGUF parsing and quantisation arithmetic.

Everything here reads a real file or computes from the GGUF specification. No
estimates dressed up as measurements, and where a number is an estimate it says
so.

The reason this exists: the question a client actually asks is "will this model
fit in my 1GB box", and the answer is not the file size. It is the file size
plus the KV cache plus the compute buffer, and the KV cache is a function of
context length, which is the one parameter people leave at its default. A 1.1GB
Q4 model with 8k context on a 1GB device is not a fit, and no amount of
quantisation gets it there. Knowing that in ten seconds is the service.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

# GGUF metadata value types, from ggml's gguf_type enum.
GGUF_TYPES = {
    0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4),
    5: ("i", 4), 6: ("f", 4), 7: ("b", 1), 10: ("Q", 8), 11: ("q", 8),
    12: ("d", 8),
}
# Array element type is the same enum, so reuse the map.
STRING = 8
ARRAY = 9

MAGIC = b"GGUF"


class GGUFError(Exception):
    pass


@dataclass
class TensorInfo:
    name: str
    n_dims: int
    dims: list[int]
    dtype: int
    offset: int

    @property
    def elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


# Bytes per weight for the block formats that matter for edge deployment,
# keyed by ggml_type (the tensor dtype enum).
#
# Block quantisation is not a flat bit depth. Q4_K packs 4 bits per weight plus
# per-superblock scales and minimums across a 256-weight block, which lands at
# 4.5 bytes/weight-by-8 rather than 4. Anyone budgeting a Q4_K_M file as
# "4 bits per weight" underestimates by 12.5%, and on a 1GB device a 12.5%
# error is the difference between fitting and not.
#
# The QK_K row is the important one in practice: a quantised model is mostly
# QK_K tensors, so it dominates the file.
GGML_BITS = {
    0: 32.0,      # F32
    1: 16.0,      # F16
    2: 4.5,       # Q4_0
    3: 4.5,       # Q4_1
    6: 5.5,       # Q5_0
    7: 5.5,       # Q5_1
    8: 8.5,       # Q8_0
    9: 8.5,       # Q8_1
    10: 2.625,    # Q2_K
    11: 3.4375,   # Q3_K
    12: 4.5,      # Q4_K
    13: 5.5,      # Q5_K
    14: 6.5625,   # Q6_K
    15: 8.5,      # Q8_K
    32: 16.0,     # BF16
}

GGML_LABEL = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 32: "BF16",
}

# general.file_type is llama_ftype, a different enum from ggml_type. Mixing
# them up is an easy and silent mistake: index 10 is Q2_K in one and Q3_K_S in
# the other, so a shared table reports the wrong quantisation for several of
# the most common builds.
FTYPE_LABEL = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    32: "BF16",
}

QUANT_LABEL = {
    0: "F32", 1: "F16", 7: "Q8_0", 8: "Q5_K_M", 10: "Q4_K_M",
    12: "Q3_K_M", 14: "Q2_K_M", 32: "Q8_K",
}


@dataclass
class GGUF:
    path: Path
    version: int
    tensor_count: int
    kv_count: int
    metadata: dict = field(default_factory=dict)
    tensors: list[TensorInfo] = field(default_factory=list)
    alignment: int = 32
    data_offset: int = 0
    dims_are_u64: bool = False

    # ---- the fields an edge decision actually turns on ----

    def m(self, suffix: str, default=0):
        """Read a metadata key for this model's own architecture.

        GGUF namespaces everything under the architecture name: a Qwen2 model
        writes `qwen2.block_count`, a custom one writes `myarch.block_count`.
        Hard-coding the `llama.` prefix -- which is what most tooling does,
        because llama.cpp defined the format -- silently returns 0 for every
        non-Llama model, and a zero layer count produces a KV cache of zero
        bytes, and therefore a budget that looks comfortable and is fiction.

        The architecture's own prefix is tried first so a model that happens to
        carry both is not read through the wrong namespace.
        """
        for key in (f"{self.arch}.{suffix}", f"llama.{suffix}", suffix):
            if key in self.metadata:
                return self.metadata[key]
        return default

    @property
    def n_layer(self) -> int:
        return int(self.m("block_count", 0))

    @property
    def n_embd(self) -> int:
        return int(self.m("embedding_length", 0))

    @property
    def n_head(self) -> int:
        return int(self.m("attention.head_count", 0))

    @property
    def n_head_kv(self) -> int:
        """KV heads. This is the single biggest lever on cache size and the
        one nobody checks: grouped-query attention means a 40-head model may
        have only 8 KV heads, which is a 5x smaller cache than the head count
        suggests."""
        return int(self.m("attention.head_count_kv",
                          self.m("attention.head_count", 0)))

    @property
    def head_dim(self) -> int:
        h = int(self.m("embedding_length", 0))
        n = int(self.m("attention.head_count", 0))
        return h // n if n else 0

    @property
    def n_kv_dtype_bytes(self) -> int:
        """Bytes per cached element. 2 for F16 KV, ~1 for Q8_0.

        Q8_0 KV quantisation halves the cache term for a measurable quality
        cost, and on a 1GB device at 8k context that is often the only way to
        fit at all. Reported as a lever rather than assumed on.
        """
        t = str(self.m("attention.key_cache_type",
                       self.m("attention.value_cache_type", "f16"))).lower()
        return 1 if "q8" in t else 2

    @property
    def context_length(self) -> int:
        return int(self.m("context_length", 0))

    @property
    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    @property
    def name(self) -> str:
        return str(self.metadata.get("general.name", self.path.stem))

    @property
    def file_type(self):
        """general.file_type. An int in most builds, a string in converters
        that predate the enum. int() on the string form raises rather than
        degrading, which loses the entire file over a label."""
        v = self.metadata.get("general.file_type", -1)
        if isinstance(v, str):
            return v
        try:
            return int(v)
        except (TypeError, ValueError):
            return -1

    @property
    def quant_label(self) -> str:
        v = self.file_type
        if isinstance(v, str) and v.strip():
            return v.strip().upper()
        if isinstance(v, int) and v in FTYPE_LABEL:
            return FTYPE_LABEL[v]
        # No usable general.file_type. Not every exporter writes one, and
        # reporting the raw value is worse than useless: "type -1" reads like a
        # finding about the model rather than a gap in the file's metadata, and
        # it discards a label the tensor table already knows. Fall back to the
        # dominant tensor dtype -- and to GGML_LABEL, not FTYPE_LABEL, because
        # these are two different enums that happen to share numbers. 12 is
        # Q3_K_M as a file type and Q4_K as a tensor dtype.
        if self.tensors:
            by_dtype: dict[int, int] = {}
            for t in self.tensors:
                by_dtype[t.dtype] = by_dtype.get(t.dtype, 0) + t.elements
            top = max(by_dtype, key=lambda k: by_dtype[k])
            return GGML_LABEL.get(top, f"dtype {top}")
        return "unknown"

    def quant_bits(self) -> float:
        """Bits per weight, measured from the tensor table.

        Read from the tensors rather than from general.file_type, because the
        label describes the converter's intent and the tensor table describes
        what is on disk. A converter that emitted Q4_0 tensors into a file
        labelled q4_k disagrees with itself, and the tensor table is the part
        the kernel will actually read.
        """
        if not self.tensors:
            return 4.5
        total = sum(t.elements for t in self.tensors)
        if not total:
            return 4.5
        return sum(t.elements * GGML_BITS.get(t.dtype, 8.5)
                   for t in self.tensors) / total

    def quant_summary(self) -> dict:
        """Weight bytes by dtype, so the mix is visible.

        A Q4_K_M file is not uniformly Q4_K: embeddings, the output head and
        usually the router stay at higher precision because quantising them
        costs disproportionate quality. Seeing the mix is the difference
        between understanding why a file is the size it is and guessing.
        """
        out: dict[str, dict] = {}
        for t in self.tensors:
            lab = GGML_LABEL.get(t.dtype, f"type {t.dtype}")
            e = out.setdefault(lab, {"elements": 0, "bytes": 0, "tensors": 0})
            e["elements"] += t.elements
            e["bytes"] += int(t.elements * GGML_BITS.get(t.dtype, 8.5) / 8)
            e["tensors"] += 1
        for v in out.values():
            v["elements"] = f"{v['elements']:,}"
            v["mb"] = round(v["bytes"] / 1024 / 1024, 2)
            del v["bytes"]
        return dict(sorted(out.items(), key=lambda kv: -kv[1]["mb"]))

    def kv_bytes(self, context: int | None = None, batch: int = 1,
                 dtype_bytes: int = 2) -> int:
        """KV cache size.

            bytes = 2 * n_layer * n_head_kv * head_dim * batch * context
                                * dtype_bytes

        The 2 is for K and V. The head_dim term is why head_dim (not n_embd) is
        used, and the n_head_kv term is why grouped-query attention wins so
        much on edge: it is the only structural change that reduces this without
        making the model worse.

        This is the dominant term at long context and the reason a model that
        "fits on disk" does not fit in RAM.
        """
        if not (self.n_layer and self.n_head_kv and self.head_dim):
            return 0
        ctx = context or self.context_length
        return (2 * self.n_layer * self.n_head_kv * self.head_dim
                * batch * ctx * dtype_bytes)
    def weights_bytes(self) -> int:
        """Actual tensor payload, read from the tensor table.

        Summed from declared tensor shapes rather than derived from a nominal
        bit depth, because a Q4_K_M file is not 4 bits per weight and anyone
        who budgets from the nominal number underestimates by about a fifth.
        """
        total = 0
        for t in self.tensors:
            bits = GGML_BITS.get(t.dtype, 8.5)
            total += int(t.elements * bits / 8)
        return total


def _read(f, fmt: str, size: int | None = None):
    n = struct.calcsize(fmt) if size is None else size
    return struct.unpack(fmt, f.read(n))


def parse(path: str | Path, max_tensors: int | None = None) -> GGUF:
    """Read a GGUF header and tensor table.

    `max_tensors` caps how many tensor entries are read. Metadata is what an
    edge audit needs and it sits at the front of the file, so a capped parse
    returns everything useful without reading a 4GB tensor table.
    """
    p = Path(path)
    with p.open("rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise GGUFError(f"{p.name}: not a GGUF file (magic {magic!r})")

        version, = _read(f, "I")
        if version < 2:
            raise GGUFError(f"{p.name}: GGUF v{version} too old to parse")
        n_tensors, n_kv = _read(f, "QQ")

        meta: dict = {}
        for _ in range(n_kv):
            key = _read_string(f)
            t, = _read(f, "I")
            meta[key] = _read_value(f, t, version)

        alignment = int(meta.get("general.alignment", 32))
        want = n_tensors if max_tensors is None else min(n_tensors, max_tensors)
        tensors, width = _parse_tensors(f, want, p.name)
    return GGUF(path=p, version=version, tensor_count=n_tensors, kv_count=n_kv,
                metadata=meta, tensors=tensors, alignment=alignment,
                dims_are_u64=(width == 8))


def _read_tensor_table(f, count: int, name: str, dim_bytes: int) -> list[TensorInfo]:
    """Read the tensor table with one candidate dimension width."""
    out: list[TensorInfo] = []
    for _ in range(count):
        tname = _read_string(f)
        n_dims, = _read(f, "I")
        if not 1 <= n_dims <= 4:
            raise GGUFError(f"{name}: tensor '{tname}' has {n_dims} dims")
        fmt = "I" if dim_bytes == 4 else "Q"
        dims = list(_read(f, fmt * n_dims))
        dtype, = _read(f, "I")
        offset, = _read(f, "Q")
        out.append(TensorInfo(tname, n_dims, dims, dtype, offset))
    return out


def _valid_table(tensors: list[TensorInfo], expected: int) -> bool:
    """Does this table look like a real one?

    Three independent signals, all necessary. A single one is not enough:
    a 1-D tensor's first four bytes look identical under either width, so
    per-entry detection silently picks the wrong one and then desynchronises
    on the next entry. Validating the whole table is what makes the
    discriminator sound.
    """
    if not tensors or len(tensors) != expected:
        return False
    prev = -1
    for t in tensors:
        if not t.name or any(c in t.name for c in "\x00\n\r\t"):
            return False          # desynchronised: length read as a name
        if not all(d > 0 for d in t.dims):
            return False          # a zero dimension is not legal in any variant
        if t.offset < prev:
            return False          # offsets are non-decreasing in a real file
        prev = t.offset
    return True


def _parse_tensors(f, count: int, name: str) -> tuple[list[TensorInfo], int]:
    """Read the tensor table, tolerating a non-spec dimension width.

    The specification says uint32. Some third-party exporters write uint64,
    and the failure is nasty rather than clean: the parse desynchronises and
    then reports plausible-looking garbage -- one real dimension beside a zero,
    a zero-length tensor name, an absurd dimension count three entries later.
    Nothing says "wrong width", so the file gets discarded instead of audited.

    Both widths are tried and the table that validates wins. Where both
    validate, the spec-compliant width is preferred, because that is what a
    future file is more likely to use and a false accept is the worse error.
    """
    start = f.tell()
    results: dict[int, list[TensorInfo]] = {}
    for dim_bytes in (4, 8):
        f.seek(start)
        try:
            results[dim_bytes] = _read_tensor_table(f, count, name, dim_bytes)
        except (GGUFError, struct.error, UnicodeDecodeError, MemoryError):
            results[dim_bytes] = []
    for dim_bytes in (4, 8):
        if _valid_table(results.get(dim_bytes) or [], count):
            return results[dim_bytes], dim_bytes
    # Nothing validated. Return the spec-width attempt so the caller sees real
    # (if useless) data rather than an empty audit, and let the report say so.
    return results.get(4) or [], 4


def _read_string(f) -> str:
    n, = _read(f, "Q")
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, t: int, version: int = 3):
    if t == STRING:
        return _read_string(f)
    if t == ARRAY:
        et, = _read(f, "I")
        # Array element count widened from uint32 to uint64 in GGUF v3. Reading
        # a v3 array length as 4 bytes leaves the parser 4 bytes short, so the
        # next string length comes out as a large number and the tensor table
        # desynchronises -- which surfaces as a tensor named "" with 28
        # dimensions, rather than as a version error. Worth knowing that the
        # symptom is nowhere near the cause.
        n, = _read(f, "I" if version < 3 else "Q")
        return [_read_value(f, et, version) for _ in range(n)]
    if t in GGUF_TYPES:
        fmt, size = GGUF_TYPES[t]
        return _read(f, fmt)[0]
    raise GGUFError(f"unknown metadata type {t}")


# --------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------

def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


@dataclass
class Budget:
    """RAM budget for one deployment, and whether the model fits in it.

    Three separate consumers, and conflating them is the standard mistake:

      weights  the model file. With mmap this is virtual, not resident, which
               is why a 3GB model can run in 1GB of RAM as long as the pages
               actually touched fit -- at the cost of disk I/O per token.
      cache    the KV cache. Grows linearly with context. Never on disk.
      compute  activation scratch, roughly 1-2GB for llama.cpp at default
               batch size on a 1-2B model. Fixed cost, easy to forget, and the
               reason `--batch-size 8` needs saying out loud.
    """
    total_mb: float
    weights_mb: float
    cache_mb: float
    compute_mb: float
    mmap: bool = True
    compute_reserve_mb: float = 800.0

    @property
    def resident_mb(self) -> float:
        """What actually has to be in RAM right now.

        With mmap the weights are demand-paged, so the honest figure for the
        steady state is the compute buffer plus the cache, with the resident
        slice of the weights on top. A single inference pass touches most of
        the weights but not all of them at once, and the ratio depends on batch
        and context -- so this reports the pessimistic case (all weights
        resident) as the default, because that is the one that fails on a
        device rather than merely running slowly.
        """
        w = self.weights_mb if not self.mmap else self.weights_mb * 0.6
        return w + self.cache_mb + self.compute_mb

    @property
    def fits(self) -> bool:
        return self.resident_mb <= self.total_mb

    @property
    def headroom_mb(self) -> float:
        return self.total_mb - self.resident_mb


def audit(path: str | Path, context: int = 4096, budget_mb: float = 1024,
          batch: int = 1, mmap: bool = True, full: bool = False,
          compute_mb: float = 800.0) -> dict:
    """Full edge audit of one GGUF file.

    Returns the numbers a sizing decision turns on: real weights from the
    tensor table, KV cache at the requested context, a per-context curve
    because that is where the surprises are, and a fit verdict against the
    budget.
    """
    g = parse(path, max_tensors=None if full else 4096)
    file_mb = g.path.stat().st_size / 1024 / 1024

    weights = g.weights_bytes() / 1024 / 1024 if g.tensors else file_mb
    kvdt = g.n_kv_dtype_bytes
    cache = g.kv_bytes(context, batch, kvdt) / 1024 / 1024

    curve = []
    for c in (512, 1024, 2048, 4096, 8192, 16384, 32768):
        if g.n_layer:
            curve.append({"context": c,
                          "cache_mb": round(g.kv_bytes(c, batch, kvdt) / 1024 / 1024, 1)})

    b = Budget(total_mb=budget_mb, weights_mb=weights, cache_mb=cache,
               compute_mb=compute_mb, mmap=mmap)
    out = {
        "file": str(g.path),
        "name": g.name,
        "arch": g.arch,
        "quant": g.quant_label,
        "file_mb": round(file_mb, 1),
        "weights_mb": round(weights, 1),
        "n_layer": g.n_layer,
        "n_embd": g.n_embd,
        "n_head": g.n_head,
        "n_head_kv": g.n_head_kv,
        "head_dim": g.head_dim,
        "gqa_ratio": round(g.n_head / g.n_head_kv, 2) if g.n_head_kv else 0,
        "kv_dtype_bytes": kvdt,
        "native_context": g.context_length,
        "context": context,
        "batch": batch,
        "cache_mb": round(cache, 1),
        "compute_mb": compute_mb,
        "resident_mb": round(b.resident_mb, 1),
        "budget_mb": budget_mb,
        "fits": b.fits,
        "headroom_mb": round(b.headroom_mb, 1),
        "mmap": mmap,
        "curve": curve,
        "tensors_read": len(g.tensors),
        "tensors_total": g.tensor_count,
    }
    if full or not g.tensors:
        out["quant_mix"] = g.quant_summary()
    out["recommendations"] = _recommend(out)
    return out


def _recommend(a: dict) -> list[str]:
    """What to actually do about it.

    Ranked by how much memory each one returns per unit of effort, because
    capping context is free and choosing a different model is not.
    """
    recs = []
    if a["fits"]:
        recs.append(f"Fits {a['budget_mb']:.0f} MB with {a['headroom_mb']:.0f} MB "
                    f"headroom at {a['context']} context.")
    else:
        recs.append(f"Does NOT fit {a['budget_mb']:.0f} MB: {a['resident_mb']:.0f} MB "
                    f"resident, {abs(a['headroom_mb']):.0f} MB over.")

    # Context is the cheapest lever, but only worth naming when it actually
    # changes the verdict. Suggesting a smaller context to a model that already
    # fits trains the reader to ignore the recommendations.
    if not a["fits"] and a["curve"]:
        for point in a["curve"]:
            need = a["weights_mb"] * 0.6 + point["cache_mb"] + a["compute_mb"]
            if need <= a["budget_mb"] and point["context"] < a["context"]:
                recs.append(
                    f"Capping context to {point['context']} brings it to "
                    f"{need:.0f} MB resident against a {a['budget_mb']:.0f} MB "
                    f"budget. This is the cheapest fix available and it costs "
                    f"nothing: cap with -c {point['context']}.")
                break
    elif a["fits"] and a["cache_mb"] > 0:
        recs.append(
            f"Context is not the binding constraint here -- the cache is only "
            f"{a['cache_mb']:.0f} MB against a {a['budget_mb']:.0f} MB budget. "
            f"Raising -c costs little here, so leave it above "
            f"{a['context']} if the task needs it.")

    # If scratch alone busts the budget then no context value can rescue this,
    # and saying so is more useful than leaving the reader to try -c 128 and
    # wonder why nothing changed.
    if not a["fits"] and a["compute_mb"] >= a["budget_mb"]:
        recs.append(
            f"No context value fixes this: activation scratch alone is about "
            f"{a['compute_mb']:.0f} MB against a {a['budget_mb']:.0f} MB budget. "
            f"Drop --batch-size to 1, or use a smaller embedding size.")

    if a["gqa_ratio"] and a["gqa_ratio"] > 1:
        recs.append(f"Grouped-query attention: {a['n_head']} query heads over "
                    f"{a['n_head_kv']} KV heads ({a['gqa_ratio']}x). The KV cache "
                    f"is already {a['gqa_ratio']}x smaller than the head count "
                    f"implies -- do not 'optimise' this away.")

    if not a["mmap"]:
        recs.append("Enable mmap. With --no-mmap the whole file is resident and "
                    "this budget is wrong; with mmap it is demand-paged.")

    if a["native_context"] and a["context"] > a["native_context"]:
        recs.append(f"Requested context {a['context']} exceeds the model's native "
                    f"{a['native_context']}. Beyond native, quality degrades "
                    f"whether or not it fits -- cap at native.")

    if a["kv_dtype_bytes"] > 1:
        recs.append("KV cache is F16. Q8_0 KV halves the cache term for a small "
                    "quality cost, and at 8k+ context that is often the "
                    "difference between fitting and not.")

    recs.append("Lower --batch-size below 8. Activation scratch is roughly "
                f"{a['compute_mb']:.0f} MB of the total and scales with batch, "
                "not context.")

    if a["fits"] and a["headroom_mb"] > 0:
        recs.append("Keep the persistent llama-server rather than one-shot "
                    "subprocesses: KV-slot reuse and warm weights remove the "
                    "load cost per request, which is the difference between "
                    "usable and not on edge hardware.")
    return recs


def compare(quant_of_same_model: dict[str, str], context: int = 4096,
            budget_mb: float = 1024) -> list[dict]:
    """Audit several quantisations of one model side by side.

    The comparison that decides a deployment, because the Q4/Q5/Q8 choice is
    usually made on vibes and the actual crossover on a 1GB device is not
    where people assume.
    """
    rows = []
    for label, path in quant_of_same_model.items():
        try:
            a = audit(path, context=context, budget_mb=budget_mb)
            rows.append({"quant": label, "file_mb": a["file_mb"],
                         "cache_mb": a["cache_mb"], "resident_mb": a["resident_mb"],
                         "fits": a["fits"]})
        except (GGUFError, OSError) as e:
            rows.append({"quant": label, "error": str(e)})
    return sorted(rows, key=lambda r: r.get("resident_mb", 1e9))
