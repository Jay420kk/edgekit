"""What this machine actually is, measured.

Not `nproc` and not a vendor string. The numbers an edge decision turns on are
available bandwidth, available memory and whether a GPU is present and how much
of it is free -- and on a laptop with a shared display buffer, "how much is
free" is a different question from "how much exists".

Each probe is defensive: a machine missing a tool reports unavailable rather
than failing the audit, because the audit is worth running anyway.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


def _run(cmd: list[str], timeout: int = 10) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout if p.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


@dataclass
class Machine:
    cpu: str = "unknown"
    cores_logical: int = 0
    cores_physical: int = 0
    ram_total_mb: float = 0.0
    ram_available_mb: float = 0.0
    swap_mb: float = 0.0
    disk_free_mb: float = 0.0
    gpu: str = ""
    gpu_vram_mb: float = 0.0
    gpu_free_mb: float = 0.0
    tools: dict = field(default_factory=dict)

    @property
    def compute_total_mb(self) -> float:
        """Everything the model can use, GPU included.

        Unified memory counts once. Discrete VRAM adds to system RAM because
        the host still needs RAM for everything that is not the model, and
        because llama.cpp's host-side buffers are in system RAM regardless of
        where the weights went.
        """
        return self.ram_available_mb + self.gpu_free_mb

    def as_dict(self) -> dict:
        return {
            "cpu": self.cpu, "cores_logical": self.cores_logical,
            "cores_physical": self.cores_physical,
            "ram_total_mb": round(self.ram_total_mb),
            "ram_available_mb": round(self.ram_available_mb),
            "swap_mb": round(self.swap_mb),
            "disk_free_mb": round(self.disk_free_mb),
            "gpu": self.gpu or "none detected",
            "gpu_vram_mb": round(self.gpu_vram_mb),
            "gpu_free_mb": round(self.gpu_free_mb),
            "compute_total_mb": round(self.compute_total_mb),
            "tools": self.tools,
        }


def probe() -> Machine:
    m = Machine()
    m.cpu = _cpu_model()
    m.cores_logical = os.cpu_count() or 0
    m.cores_physical = _physical_cores(m.cores_logical)
    m.ram_total_mb, m.ram_available_mb, m.swap_mb = _mem()
    m.disk_free_mb = _disk_free()
    m.gpu, m.gpu_vram_mb, m.gpu_free_mb = _gpu()
    m.tools = {
        t: bool(shutil.which(t)) for t in
        ("llama-cli", "llama-server", "ollama", "nvidia-smi", "quantize")
    }
    return m


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _physical_cores(logical: int) -> int:
    """Physical cores, which is what inference actually scales on.

    A thread count is a number the OS reports; a core count is a number the
    work divides across. On a 2c/4t laptop the difference is 2x and it is the
    difference between "interactive" and "not".
    """
    try:
        ids = set()
        text = Path("/proc/cpuinfo").read_text()
        cur = []
        for line in text.splitlines() + [""]:
            if not line.strip():
                if any(l.lower().startswith("core id") for l in cur):
                    ids.add(tuple(sorted(l for l in cur if "core id" in l.lower())))
                cur = []
            else:
                cur.append(line)
        if ids:
            return len(ids)
    except OSError:
        pass
    # "cpu cores" in lscpu is the cheaper path when /proc is unreadable.
    out = _run(["lscpu"])
    m = re.search(r"^CPU\(s\):\s*(\d+)", out, re.M)
    c = re.search(r"^Core\(s\) per socket:\s*(\d+)", out, re.M)
    s = re.search(r"^Socket\(s\):\s*(\d+)", out, re.M)
    if m and c:
        return int(c.group(1)) * int(s.group(1) if s else 1)
    return logical


def _mem() -> tuple[float, float, float]:
    """(total, available, swap) in MB.

    `available` rather than `free`: `free` on Linux reports MemAvailable's
    neighbour and understates what a process can actually claim by a gigabyte
    on a machine with page cache. Sizing a deployment against `free` is how a
    model that "fits" gets OOM-killed.
    """
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            info[k] = int(v.strip().split()[0]) / 1024
        return (info.get("MemTotal", 0.0), info.get("MemAvailable", 0.0),
                info.get("SwapTotal", 0.0))
    except (OSError, ValueError, IndexError):
        return 0.0, 0.0, 0.0


def _disk_free() -> float:
    try:
        st = os.statvfs(str(Path.home()))
        return st.f_bavail * st.f_frsize / 1024 / 1024
    except OSError:
        return 0.0


def _gpu() -> tuple[str, float, float]:
    """(name, vram_mb, free_mb).

    A discrete GPU with a display attached is running a desktop compositor that
    has already claimed part of its VRAM, and the difference between total and
    free is the number that decides whether a model loads.
    """
    out = _run(["nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits"])
    if out.strip():
        first = out.strip().splitlines()[0]
        parts = [p.strip() for p in first.split(",")]
        name = parts[0] if parts else "nvidia"
        total = float(parts[1]) if len(parts) > 1 else 0.0
        used = float(parts[2]) if len(parts) > 2 else 0.0
        return name, total, max(0.0, total - used)

    for d in ("/sys/class/drm/card0/device", "/sys/class/drm/card1/device"):
        p = Path(d)
        if not p.exists():
            continue
        name = ""
        for f in ("product_name", "uevent"):
            fp = p / f
            if fp.exists():
                m = re.search(r"DRIVER=([^\n]+)", fp.read_text()) if f == "uevent" \
                    else None
                name = (m.group(1) if m else fp.read_text().strip()) or d
                break
        mem_mb = 0.0
        for f in p.glob("mem_info_vram_total"):
            mem_mb = int(f.read_text().strip()) / 1024 / 1024
        return name or "gpu", mem_mb, mem_mb
    return "", 0.0, 0.0


# --------------------------------------------------------------------------
# throughput: measured where possible, labelled as an estimate otherwise
# --------------------------------------------------------------------------

def gpu_bandwidth_gbs(name: str) -> tuple[float, float]:
    """A defensible memory-bandwidth range for a GPU name.

    The happy path would read the real figure from nvidia-smi, but nvidia-smi
    does not report bandwidth, so this is a heuristic keyed on the generation
    and the memory flavour, which the model string usually implies. The
    numbers exist to stop a wild guess: on a GTX 1080 320 GB/s and an RTX
    4090 1008 GB/s differ by 3x, and the old "small VRAM = 800 GB/s" shortcut
    put a 2 GB 840M in the same bin as a 4090, which was off by 50x.

    Bins are deliberately wide; effective bandwidth is less than the bus peak
    and varies with batch. The caller already returns a range and an explicit
    "measure with llama-bench", so a wide bin is honest where a precise number
    would be a lie.
    """
    n = name.lower()
    if not n or name == "unknown":
        return 10.0, 60.0                       # unknown device
    if "intel" in n or "iris" in n or "uhd" in n or "hd graphics" in n:
        return 30.0, 100.0                      # iGPU/system RAM
    if "radeon" in n or "amd" in n:
        if "rx 7" in n or "vega" in n:
            return 150.0, 500.0
        return 40.0, 200.0
    if "apple" in n or n.startswith(("m1", "m2", "m3", "m4")):
        return 60.0, 550.0                      # unified, M1..M4 Max
    if "quadro" in n or "tesla" in n or "a100" in n or "h100" in n:
        return 500.0, 3000.0
    if "p100" in n or "v100" in n or "a40" in n or "mi100" in n:
        return 700.0, 1200.0
    # NVIDIA consumer. GeForce "M" parts are laptop chips on a DDR3/GDDR5
    # bus that peaks around 15-60 GB/s; desktop GDDR5 does 200-350; and the
    # last two GDDR6/GDDR6X generations bracket 400-1000+.
    if "geforce" in n:
        if "mx" in n or "m " in n or n.endswith("m"):
            return 15.0, 60.0
        if "4090" in n or "5090" in n:
            return 900.0, 1300.0
        if "4080" in n or "5080" in n or "4070" in n or "5070" in n:
            return 500.0, 1000.0
        if "4060" in n or "5060" in n or "3080" in n or "3090" in n:
            return 400.0, 900.0
        if "3060" in n or "3070" in n or "2080" in n or "2070" in n:
            return 300.0, 600.0
        if "1080" in n or "1070" in n or "2060" in n or "1660" in n:
            return 200.0, 400.0
        return 60.0, 350.0                      # older/unknown discrete
    return 40.0, 300.0


def tokens_per_second(params_b: float, quant: str, m: Machine) -> dict:
    """Rough decode rate, with the reasoning shown.

    Deliberately returns a range and a confidence rather than a single number,
    because a single number here is a lie: decode speed on CPU is set by memory
    bandwidth and how much of the model has to be streamed per token, so it
    depends on the machine in ways that resist a formula.

    The relationship that does hold is the shape: decode is bandwidth-bound,
    so time-per-token is roughly proportional to model bytes, and halving the
    quantisation nearly halves the time. Prefill is compute-bound and orders of
    magnitude faster per token, which is why prompt processing feels instant
    and generation does not.

    With a GPU present, the ceiling is memory bandwidth at 1.5-3 TB/s, which
    puts a 4GB Q4 model at 300-700 tok/s regardless of compute.
    """
    bits = {"Q4_K_M": 4.5, "Q5_K_M": 5.5, "Q6_K": 6.5625, "Q8_0": 8.5,
            "Q2_K": 2.625, "Q3_K": 3.4375, "F16": 16.0, "F32": 32.0}
    b = bits.get(quant, 4.5) * params_b * 1e9 / 8

    if m.gpu_free_mb > 0 and m.gpu_vram_mb > 0:
        # GPU decode ceiling. Bandwidth-derived, and the honest answer is a
        # range because effective bandwidth varies with batch and whether the
        # model fits in VRAM.
        if b / 1024 / 1024 > m.gpu_free_mb:
            return {"tok_s": 0, "confident": False,
                    "why": f"{b / 1024 / 1024:.0f} MB model exceeds "
                           f"{m.gpu_free_mb:.0f} MB free VRAM; it will not "
                           f"run GPU-only"}
        lo_bw, hi_bw = gpu_bandwidth_gbs(m.gpu)
        lo, hi = lo_bw * 1e9 * 0.35 / b, hi_bw * 1e9 * 0.7 / b
        return {"tok_s": round((lo + hi) / 2), "range": [round(lo), round(hi)],
                "confident": False,
                "why": f"bandwidth estimate at {lo_bw:.0f}-{hi_bw:.0f} GB/s for "
                       f"{m.gpu or 'the GPU'}; measure with llama-bench"}

    # CPU decode. 10-25 GB/s across desktop parts, 5-12 GB/s across mobile.
    lo_bw, hi_bw = (10e9, 25e9) if m.cores_physical >= 4 else (5e9, 12e9)
    lo, hi = lo_bw * 0.5 / b, hi_bw * 0.8 / b
    return {"tok_s": round((lo + hi) / 2), "range": [round(lo), round(hi)],
            "confident": False,
            "why": f"bandwidth estimate for {m.cores_physical} physical cores; "
                   f"run `llama-bench -m model.gguf` for the real number"}


def recommend_context(budget_mb: float, n_layer: int, n_head_kv: int,
                      head_dim: int, reserve_mb: float = 800.0) -> int:
    """Largest context that fits, from the cache formula inverted.

        context = (budget - weights - reserve) / (2 * L * H_kv * D * 2)

    This is the calculation people get wrong by leaving context at the model's
    default. A 1.7B model with 28 layers, 4 KV heads and 128 head dim at 8k
    context holds 28 MB of cache; at 32k it holds 115 MB. On a 1GB device that
    is the difference between a working deployment and an OOM.
    """
    if not (n_layer and n_head_kv and head_dim):
        return 0
    per_token = 2 * n_layer * n_head_kv * head_dim * 2  # K and V, F16
    room = max(0.0, budget_mb - reserve_mb)
    return int((room * 1024 * 1024) // per_token // 1024) * 1024
