# edgekit

**Decide whether an LLM fits on a specific machine.** Reads the real GGUF
header, measures this machine's real memory, and gives you a run plan —
RAM headroom, KV-cache by context, quant mix, batch advice. No guessing from
parameter counts.

```bash
pip install edgekit-audit  # import name stays `edgekit`
edgekit audit model.gguf --context 4096 --budget 1024
```

```
genesis-15m  [genesis, Q4_0]
  file            8 MB (366/366 tensors read)
  weights         8 MB
  architecture    3L  192d  4q/4kv heads  dim 48
  native context  2048

  at -c 4096  cache 9 MB (2B/elem)  compute ~800 MB
  resident total  814 MB against a 1024 MB budget

  - Fits 1024 MB with 210 MB headroom at 4096 context.
  - Requested context 4096 exceeds the model's native 2048. Beyond native,
    quality degrades whether or not it fits -- cap at native.
  - KV cache is F16. Q8_0 KV halves the cache term for a small quality cost.
```

## Why it exists

Every "will it run" answer you find online is `params × bytes-per-weight`, and
every one of them is wrong in the same direction: too optimistic. Three real
terms are missing from that arithmetic:

1. **KV-cache.** Grows with context, often dwarfs the weights at long context
   on small models. The audit prints it per context level so you can see which
   one binds.
2. **Compute scratch.** Activation memory for the forward pass — hundreds of
   MB that scale with batch, not context. edgekit assumes 800 MB by default
   because that is what a real server process holds, and a tool that ignores it
   will confidently certify a deployment that OOM-kills on the first request.
3. **The OS and everything else already running.** edgekit takes `--budget` as
   what is *available*, and `probe` measures it for you.

The result is a tool that sometimes says *no* — and explains which term said it.

## Commands

```bash
edgekit probe                          # this machine: RAM, GPU, free memory
edgekit audit model.gguf -c 4096 --budget 1024 --full
edgekit compare a.gguf b.gguf c.gguf   # side by side against one budget
edgekit plan --params 7 --quant Q4_K_M --budget 4096
```

`plan` needs no file: give it a target and a budget and it ranks real model
families that fit, largest first:

```
budget        4096 MB (specified)
target        7.0B params at Q4_K_M
weights       ~3755 MB
decode        ~0 tok/s  (estimate)
              3755 MB model exceeds 2043 MB free VRAM; it will not run GPU-only

  Candidates worth trying at this budget, largest first:
    llama-3.2-3b                     1609 MB weights  +800 MB scratch
    qwen3-1.7b                        912 MB weights  +800 MB scratch
    deepseek-r1-distill-qwen-1.5b     805 MB weights  +800 MB scratch
```

## What it reads

The GGUF header is the ground truth: tensor names, shapes, quantisation type,
architecture metadata, native context. edgekit validates the whole metadata
table before trusting a field (real files use non-spec integer widths — see the
parser tests), reads the quant mix with `--full`, and derives achievable
bandwidth from the actual GPU it finds, not a marketing number.

Speed figures are bandwidth-derived estimates and are labelled as such. Fits /
does-not-fit is arithmetic and is not.

## Honest limits

- An estimate, not a guarantee: driver overhead and other tenants can eat
  headroom. Treat 200 MB of headroom as "fits", 20 MB as "test it first".
- Multi-GPU and offload strategies are out of scope — this sizes one process
  on one box.
- No network calls, ever. The point is to run on the machine you are sizing,
  which may not have the internet.

## Tests

27 tests, all passing, no network. The parser is exercised against GGUF files
written byte-by-byte in the tests — including the non-spec layouts real
exporters emit — because a mock would happily agree with a wrong parser.

```bash
python3 tests/test_edgekit.py
```

## License

Apache-2.0. Commercial licensing available — see the parent repository.
