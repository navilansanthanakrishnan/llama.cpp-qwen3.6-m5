# Qwen3.6-27B-Q4_K_M on an Apple M5 Pro

Metal-backend work for one model on one machine: Qwen3.6-27B-Q4_K_M (hybrid
SSM + attention, 48 gated-delta-net layers, 16 attention layers, one MTP block)
on a binned M5 Pro — 16 GPU cores, 24 GiB unified memory, 270.8 GB/s attained
bandwidth measured with a frozen probe.

Context is 4096 throughout. The quantization is never changed.

The measurement harness, the ledger of every change tried and rejected, and the
standalone Metal probes are in the research repo:
**[qwen36-metal-research](https://github.com/navilansanthanakrishnan/qwen36-metal-research)**.

## Result

| | before | after |
|---|---|---|
| decode, MTP speculative | 14.57 tok/s | **23.76 tok/s** |
| width-8 verification | 215.2 ms | **110.4 ms** |
| Q4_K mat-vec, m=4096 k=14336, n=8 | 510 µs | **235 µs** |

Decode at one token per forward pass is capped at **16.40 tok/s** — 16.5 GB of
weights over 270.8 GB/s — so everything above that line comes from accepting
more than one token per pass, and every speculative figure below is reported
with its acceptance rate.

## What the kernels do

`kernel_mul_mv_sgq4k_f32`, `..._sgq5k_f32`, `..._sgq6k_f32` in
`ggml/src/ggml-metal/ggml-metal.metal` dequantize K-quantized weights **directly
into simdgroup matrix-unit registers** and multiply there, with no threadgroup
staging. They are used for Q4_K / Q5_K / Q6_K against F32 at batch widths 4–8 —
together 99.1% of this model's weight bytes.

Upstream has no kernel on this path. `mul_mv` and `mul_mv_ext` are
register-resident but scalar. `mul_mm` uses the matrix units but stages
dequantized weights through threadgroup memory, which is why it costs about the
same as mat-vec at the widths speculative verification actually uses.

Two facts make it expressible, both recovered by measurement rather than
assumed:

1. `simdgroup_matrix::thread_elements()` is **writable**, so dequantized values
   can be placed straight into the matrix registers. Its lane→element layout is
   implementation-defined, so it was mapped empirically by loading a known
   pattern and reading it back:
   `row = 4*(lane/16) + (lane%8)/2`, `col = 4*((lane%16)/8) + 2*(lane%2)`,
   two contiguous elements per lane.
2. The sum over K is **order-invariant**, so the map from fragment column to
   element within a quant block is free to choose. Choosing `elem = k*4 + t`
   turns each lane's eight weights into one aligned 8-byte load, and lets both
   nibble halves of that load feed two sub-blocks.

Cost is now **flat in batch width** — 233–235 µs from n=2 to n=8, where the
previous path climbed 163 → 510 µs — so verification cost no longer grows with
speculative draft depth. Width 1 and large batches are untouched.

`GGML_METAL_SGMV_DISABLE=1` restores the previous path, so both arms of a
comparison run on a single binary.

## Also here

- **A `dequantize_q4_K` scale bug.** The second-half super-block scale was
  divided in half precision. Over 14,919,680 super-blocks of this model the
  median `d` is 6.187e-05, so `d/16` is subnormal in half for **99.9998%** of
  blocks — median relative error 3.66e-03, p99 9.60e-03, some blocks flushing to
  zero. One character, and it applies to every Q4_K model.
- **`test-backend-ops` coverage.** K-quant mat-vec row gates, realistic shapes,
  large-k overflow cases, and partial `mul_mm` tiles. This caught two wrong
  kernels during this work before either was ever timed.
- **Three fixes that make MTP speculative decoding fit in memory**, totalling
  ~1087 MiB: `token_embd` was wired into the device mmap buffer as a hole in a
  min→max span despite being host-assigned; the draft context reserved a full
  `n_ubatch` for a one-block graph; and rollback depth was hardwired to draft
  depth.
- **A union draft extension.** llama.cpp's speculative implementations are
  alternatives — the first to produce a draft wins and the rest are skipped.
  With verification flat in width, that discards free tokens, so a prompt-lookup
  continuation is appended *after* the model's draft rather than offered instead
  of it. The draft stays a linear chain, needing no tree attention and no change
  to verification.

## Output is unchanged

Greedy generation is token-exact against the pre-change references on all 14
test prompts. Wikitext-2 perplexity is 5.9079 against a frozen 5.9079 at
`-c 512 --chunks 40`. `test-backend-ops` passes for all three types, including
broadcast, permuted, and partial-threadgroup shapes.

The union draft extension cannot affect output by construction: every appended
token is verified by the target model exactly like a model-drafted one and
discarded on mismatch.

## How the numbers were taken

Every performance figure is an interleaved ABBA A/B on a **single binary**, six
pairs, an exact sign-flip permutation test, held under an exclusive measurement
lock. The 3.0% minimum detectable effect was established by a null test and a
positive control — a deliberate 20 ms/token slowdown that the harness detected
at p=0.0078.

Runs are discarded above 25000 swap pages or 5% relative standard deviation.
The environment gate refuses to benchmark on battery, under Spotlight indexing,
above a load average of 3.0, or with another process over 50% CPU.

## Reproducing

```bash
git clone https://github.com/navilansanthanakrishnan/llama.cpp-qwen3.6-m5 llama.cpp
cd llama.cpp && git checkout sgmv-q4k
cmake -B build -DGGML_METAL=ON && cmake --build build -j
```

Run decode with the configuration these numbers were taken at:

```bash
LLAMA_ARG_SPEC_N_RS_SEQ=3 LLAMA_ARG_SPEC_MTP_MAX=4 LLAMA_ARG_SPEC_EXT_N=3 \
build/bin/llama-server -m Qwen3.6-27B-Q4_K_M.gguf \
  -ngl 99 -fa on -ctk f16 -ctv f16 -c 4096 -b 512 -ub 512 -np 1 \
  --spec-type draft-mtp --spec-draft-n-max 7 --spec-draft-n-min 0 -ctxcp 0 -cram 0
```

`-b` must equal the ubatch and `-ctxcp 0 -cram 0` are required: without them the
server reserves up to 32 context checkpoints at 149.6 MiB each plus an 8 GiB
prompt cache and dies on a 24 GiB machine.

`GGML_METAL_SGMV_DISABLE=1` selects the previous path, so both arms of a
comparison run on one binary. `GGML_METAL_SGMV_NO_Q5K=1` isolates the Q5_K
kernel alone.

Per-shape kernel timings, no model needed:

```bash
build/bin/test-backend-ops perf -o MUL_MAT -p "type_a=q4_K,type_b=f32,m=4096"
build/bin/test-backend-ops test -o MUL_MAT -p "type_a=q4_K"   # correctness
```

The A/B harness, the environment gate, and the standalone kernel probe are in
the [research repo](https://github.com/navilansanthanakrishnan/qwen36-metal-research).

## Branches

`sgmv-q4k` is the tuned stack and the default branch. `trunk` is the upstream
commit it forks from. `frspec-test` carries a frequency-ranked draft vocabulary
ported from prior work on an M4 Max: its mechanism works — the decode cycle drops
from ~160 ms to 112 ms — but it loses on acceptance here and is deliberately not
merged. Other branches hold experiments that were measured and **rejected** — split-K `mul_mm` (−57%), a narrow-N `mul_mm` tile (−23%), a
multi-column K-quant mat-vec, and a device-to-host memcpy fast path (noise) —
kept as the record of what was tried.
