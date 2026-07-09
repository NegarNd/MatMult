# MOAI Col×Col BSGS — column-streaming memory schedule

**Status: NOT YET RUNTIME-VERIFIED.** The code compiles-and-runs check and the
correctness/parity verification are to be done on a separate machine — see
[How to verify](#how-to-verify) below. This document explains what changed, why,
and exactly what to check.

## TL;DR

`MoaiColColBSGSHE` (in `moai_cipher.go`) was rewritten to **stream one column
`i` at a time** instead of pre-materializing the full baby-step cache
`beta[dPrime][b]` **and** the full giant-step cache `qRotAll[dPrime][·]` and
holding both for the entire computation. The math is **identical** (same
rotations, same accumulation order, same relin/rescale/final-rotate); only the
*order of construction and the memory lifetime* change. Peak live rotated
ciphertexts drop from **Θ(dPrime·√m)** to **Θ(√m)**.

## Why (the problem)

The previous implementation built, before the multiply loop and kept resident
throughout:

- `beta[i][r]` for **all** `i ∈ [0, dPrime)` — all baby-step rotations of every
  K column = `dPrime·(b−1)` ciphertexts, and
- `qRotAll[i][shift]` for **all** `i` — all giant-step rotations of every Q
  column = `dPrime·|giantShifts|` ciphertexts.

With `b = g = ⌈√m⌉`, that is a resident working set of **Θ(dPrime·√m)**
ciphertexts. For the suite's `size = 1024` (m = d' = 1024, so b = g = 32) that is
~31,744 + ~31,744 ≈ **63k ciphertexts held simultaneously**.

At the ring degree the runner actually uses this OOMs a large-RAM host:

| ring | per-ciphertext (level 4) | 2 caches (~63k cts) | + Q/K/V inputs | peak |
|---|---|---|---|---|
| **LogN=16** (N=65536) | ~5.0 MiB | ~310 GiB | ~15 GiB | **~327 GiB → OOM on a 250 GB host** |
| **LogN=13** (N=8192)  | ~0.62 MiB | ~39 GiB | ~2 GiB | ~41 GiB (fits) |

> Note: `init_lattigo.go` `DefaultParams` currently sets `LogN: 16` while the
> line's comment says "ring degree 2^13 → 4096 slots." That mismatch is a
> *separate* issue (it 8×'s every ciphertext); this change fixes the algorithm's
> memory scaling regardless of ring size, so the kernel no longer needs the full
> Θ(dPrime·√m) resident set even at LogN=16.

## What changed (the schedule)

**Before:** `precompute all beta` → `precompute all qRotAll` → for each
`(alpha, r)=j`: inner-sum over **all** `i` (needs every column's rotations
resident) → relin/rescale/rotate.

**After:** for each column `i`:
1. build only `betaI` (that column's `b` baby rotations — one hoist), and
2. `qRotI` (that column's giant rotations — one hoist),
3. fold column `i` into every output accumulator `out[j]`
   (`MulNew` for the first column to touch `j`, `MulThenAdd` after), then
4. drop `betaI`/`qRotI` (they go out of scope → **Go GC reclaims** them before
   column `i+1`).

After the column loop, a **final pass** does `Relinearize` + `Rescale` + the
per-giant-step output rotation on each `out[j]`.

Peak live rotations = **Θ(√m)** (one column's `betaI` + `qRotI`) plus the `m`
output accumulators `out[j]` (held either way). No extra hoisting: still exactly
one baby hoist and one giant hoist per column, same as before.

## Why it is correctness-preserving

The result for each output is unchanged:

```
out[j] = Σ_i  Rot(Q[i], giantShiftByAlpha[alpha]) · Rot(K[i], r*rotStride)
         (accumulated over i = 0..dPrime-1, in that order)
         then Relinearize, Rescale, and Rot by finalShift
```

- **Same rotations:** `betaI[r]` and `qRotI[shift]` are the identical hoisted
  rotations the old code stored in `beta[i][r]` / `qRotAll[i][shift]`.
- **Same accumulation order:** columns are folded `i = 0, 1, …, dPrime-1`, the
  same order as the old inner loop (`MulNew` at `i=0`, `MulThenAdd` after).
- **Same finalization:** one `Relinearize` then `Rescale` then final `Rotate`
  per `j`. The only difference is *when* — deferred to a final pass — which is
  valid because `out[j]` receives its last contribution only after the final
  column. The degree-2 accumulator carries the un-relinearized sum until then.

This is the **same schedule already used in the Orion GPU port**
(`benchmarks/matmul_encodings/kernels/moai_cipher.py :: moai_col_col_bsgs_he`),
which passes the kernel oracle suite (34/34, plaintext-oracle vs numpy and CKKS
vs plaintext-oracle). Negar signed off on this optimization during that port.

## How to verify

Run these on a machine with a Go 1.24 toolchain (module needs network to fetch
`lattigo/v6`). All commands from `matmult/`.

1. **Compiles + vets:**
   ```
   go build ./...
   go vet ./...
   ```

2. **Correctness vs plaintext (small, fast):** the HE runner has a built-in
   `verify` path that compares the decrypted result to a plaintext QKᵀ reference
   and prints `PASS ✓` when `max_err < 1e-3`. Exercise it at a small `m` first
   (edit `MoaiCiphertextSuite` in `moai_runner.go`, or add a `_test.go`, to call
   e.g. `RunMoaiHE(ctx, 16, 16, true, true, 4, 1, true)`), confirm `PASS ✓` and
   `max_err` ~1e-4 or better.

3. **Correctness at the real size:** run the full suite
   (`MoaiCiphertextSuite(verify=true, …)`, i.e. `size = 1024`). Expect `PASS ✓`.
   With this change it should now complete **without OOM** even at `LogN=16`
   (peak ~20 GiB instead of ~327 GiB). Watch RSS to confirm the memory drop.

4. **Parity vs the previous implementation (strongest check):** check out the
   parent commit's `MoaiColColBSGSHE` under a different name (e.g.
   `MoaiColColBSGSHEOld`) in a scratch build, run both on identical encrypted
   inputs, decrypt both, and assert the outputs match to ~CKKS noise
   (max abs diff ≲ 1e-6). This proves the rewrite is arithmetically equivalent,
   not merely "close to plaintext."

5. **Timing sanity:** wall-clock of the kernel should be within noise of the old
   version (identical op counts / hoists). A large regression would indicate an
   accidental extra hoist or a lost alias.

## Files

- `moai_cipher.go` — `MoaiColColBSGSHE` rewritten (this change).
- No signature change; callers (`moai_runner.go`) and the diag×col path are
  untouched.

## Related, out of scope for this change

- The `DefaultParams` `LogN: 16` vs "2^13" comment mismatch in
  `init_lattigo.go` (see note above) — worth reconciling separately.
- The diag×col variant (`MoaiDiagColBSGSHE`) already streams per-`j` and was not
  modified.
