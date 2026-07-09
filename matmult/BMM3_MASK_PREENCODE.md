# BMM-III: pre-encode plaintext masks outside the timed region

**Status: NOT YET RUNTIME-VERIFIED.** Compiles-and-runs + correctness checks are
to be done on a machine with a Go toolchain — see [How to verify](#how-to-verify).

## What this does

BMM-III's cached/hoisted modes used to build a **fresh** `maskCache` inside every
`MatMulBmm3HE` call, so the plaintext masks were **re-encoded inside the timed
region on every trial**. This change makes the masks **pre-encoded and reused**,
so the timed region measures the matmul only — matching how THOR (pre-encoded
`EllMasks`) and rowenc (`BuildRowMasks`) are already timed.

Purpose: measure how much the mask *encode* contributes to the BMM-III runtime
(and remove it from the reported kernel time). On a GPU backend the same masks
cost ~75–98% of the bmm3 "runtime"; on CPU it is smaller but measurable — this
lets you quantify it.

## What changed (3 small pieces, ~40 lines)

1. **Persistent cache on the context** (`init_lattigo.go`, `bmm3_cipher.go`):
   `HEContext` gains `bmm3Masks map[int]*maskCache`, exposed via
   `(*HEContext).bmm3MaskCache(level)`. It returns a cache **shared across all
   `MatMulBmm3HE` calls** on that context (keyed by level).
2. **Cached/hoisted modes use it** (`bmm3_cipher.go`): `bmm3HECached` /
   `bmm3HEHoisted` now call `ctx.bmm3MaskCache(inputLevel-1)` instead of building
   a throwaway cache each call. **Naive mode is unchanged** (fresh encode every
   call — it stays the "masks-in-timer" reference).
3. **Runner pre-encodes before timing** (`bmm3_runner.go`): `RunBmm3HE` makes one
   **untimed warmup call** before the timed loop, which encodes the masks into
   the persistent cache. So the timed trials reuse them.

## You get the benefit in a SINGLE run

- Because the cache is **persistent on the context**, repeated kernel calls reuse
  encoded masks **even without any warmup** (first call encodes, later calls hit
  the cache) — you don't have to run the program twice.
- Because the runner also does **one untimed warmup call before the timed loop**,
  a single run already gets the *full* benefit: **every** timed trial (including
  trial 1) measures the matmul only. The masks are pre-computed before the run's
  timed section starts.

## Why it is correctness-preserving

The masks are **data-independent shape constants**: `maskCache.get(start,end)`
builds a 0/1 indicator over slots `[start,end)` where `start,end` come only from
the chunk boundaries / rotation pattern (functions of `n,m,p,nHE`), never from
`A`/`B`. So encoding a mask once and reusing it across trials — and across calls
on the same context — yields identical ciphertexts. No change to the HE math.

(Not concurrency-safe across goroutines, matching the rest of this benchmark.)

## How to verify

From `matmult/`, on a machine with Go 1.24:

1. `go build ./...` and `go vet ./...`
2. Correctness: run the bmm3 HE suite with `verify=true` (option 12 / whatever
   invokes `BMM3CiphertextSuite(true, ...)`), expect `PASS ✓` (`max_err < 1e-3`)
   at the current `{128,131,129}` config (uncomment larger configs as desired).
3. **The measurement (the point of this PR):** compare the reported runtime
   *before* vs *after* this change — or, on a single build, compare
   `Bmm3ModeHoisted` (masks now pre-encoded → matmul only) vs `Bmm3ModeNaive`
   (masks re-encoded every call). The difference is the mask-encode cost on CPU.

## Files

- `init_lattigo.go` — `HEContext.bmm3Masks` field.
- `bmm3_cipher.go` — `bmm3MaskCache` accessor; cached/hoisted use it.
- `bmm3_runner.go` — untimed warmup that pre-encodes the masks.
- Naive mode and all HE arithmetic are unchanged.
