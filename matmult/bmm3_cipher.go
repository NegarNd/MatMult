// bmm3_cipher.go
//
// HE implementation of BMM-III bicyclic matrix multiplication on
// Lattigo v6. Direct port of ciphertext/bmm3_cipher.py.
//
// BMM-III handles matrices whose bicyclic encodings exceed a single
// ciphertext (n·m > n_he or m·p > n_he). Each encoded vector is split
// into w = ceil(enc_len / n_he) chunks of n_he slots and rotated across
// chunk boundaries using LongRot.
//
// THREE MODES (selected via Bmm3Mode)
// -----------------------------------
//   Bmm3ModeNaive   : generate plaintext masks fresh on every LongRot call.
//                     Highest mask-encode cost, no cache overhead.
//
//   Bmm3ModeCached  : cache encoded masks by (start, end) key. Masks are
//                     computed once and reused across iterations,
//                     saving repeated Encoder.Encode calls.
//
//   Bmm3ModeHoisted : cached masks + block-hoisted rotations. For every
//                     block of `HoistBlockSize` outer iterations, we take
//                     the union of rotation shifts each chunk needs, call
//                     RotateHoistedNew once per chunk, then apply each
//                     LongRot by reading from the hoisted map.
//
// Scale and degree management (lazy rescale + lazy relinearization)
// -----------------------------------------------------------------
// Input chunks encrypted at default scale S, level L.
//
// Plaintext masks are encoded at scale Q[L] so that ct × mask after a
// single Rescale lands at scale S, level L − 1. After LongRot, every
// rotated chunk is at (level L − 1, scale S, degree 1).
//
// In the inner accumulation loop (bmm3Loop) we deliberately:
//   • Use Mul  — not MulRelin — for every ct × ct product, leaving each
//     product as a degree-2 ciphertext (3 polynomial components) at
//     scale S² and level L − 1. Relinearization is deferred.
//   • Skip the per-product Rescale, so accumulated dest[s] stays at S².
//     Rescaling is deferred.
//
// Across all m iterations this pays the inner-loop cost of m·⌈np/n_he⌉
// Mul's with ZERO Relin's and ZERO Rescale's. The deferred work is
// performed once per output chunk in bmm3Finalize:
//   • Rescale     : (deg 2, S², L−1) → (deg 2, S, L−2).
//   • Relinearize : (deg 2, S,  L−2) → (deg 1, S, L−2).
//
// Net cost: m·⌈np/n_he⌉ Mul's and only ⌈np/n_he⌉ Relin's per BMM-III,
// compared with m·⌈np/n_he⌉ Relin's under eager relinearization. Total
// multiplicative depth is 2 (one CMul + one Mul), matching BMM-I and
// the bound in Theorem 2 of the BMM paper.

package main

import (
	"fmt"
	"sort"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Bmm3Mode names the three strategies above. Matches the Python `mode=`
// keyword argument strings for easy side-by-side benchmarking.
type Bmm3Mode string

const (
	Bmm3ModeNaive   Bmm3Mode = "naive"
	Bmm3ModeCached  Bmm3Mode = "cached"
	Bmm3ModeHoisted Bmm3Mode = "hoisted"
)

// ---------------------------------------------------------------------------
// Mask cache
// ---------------------------------------------------------------------------

// maskKey is the (start, end) tuple the Python dict uses. Go map keys
// must be comparable; a struct works.
type maskKey struct {
	start, end int
}

// maskCache encodes plaintext masks lazily on first use, at the scale
// needed for the ct × mask → Rescale to land at default scale.
type maskCache struct {
	ctx   *HEContext
	level int
	scale rlwe.Scale
	cache map[maskKey]*rlwe.Plaintext
}

func newMaskCache(ctx *HEContext, level int) *maskCache {
	return &maskCache{
		ctx:   ctx,
		level: level,
		scale: rlwe.NewScale(ctx.Params.Q()[level]),
		cache: make(map[maskKey]*rlwe.Plaintext),
	}
}

// get returns the plaintext mask with 1.0 in slots [start, end) and 0.0
// elsewhere, encoded at `scale = Q[level]`.
//
// In naive mode we pass `cache == nil` and each call is a fresh encode.
func (mc *maskCache) get(start, end int) *rlwe.Plaintext {
	key := maskKey{start, end}
	if mc.cache != nil {
		if pt, ok := mc.cache[key]; ok {
			return pt
		}
	}
	vec := make([]float64, mc.ctx.NHE)
	for k := start; k < end && k < mc.ctx.NHE; k++ {
		vec[k] = 1.0
	}
	pt := EncodePlaintext(mc.ctx.Encoder, mc.ctx.Params, vec, mc.level, mc.scale)
	if mc.cache != nil {
		mc.cache[key] = pt
	}
	return pt
}

// maskFn is the signature LongRot calls when it needs a plaintext mask.
// It takes `(start, end)` and returns the encoded plaintext. In naive
// mode it encodes fresh; in cached/hoisted mode it hits a maskCache.
type maskFn func(start, end int) *rlwe.Plaintext

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

// posMod3 is Python's `a % b` where the result is always non-negative.
// (Go's `%` can return negative values for negative operands.)
func posMod3(a, b int) int {
	r := a % b
	if r < 0 {
		r += b
	}
	return r
}

// ctMulMaskRescale returns a fresh ct × mask ciphertext, rescaled so it
// lands at the default scale at one level below the input.
func ctMulMaskRescale(eval *ckks.Evaluator, ct *rlwe.Ciphertext, mask *rlwe.Plaintext) *rlwe.Ciphertext {
	out, err := eval.MulNew(ct, mask)
	if err != nil {
		panic(fmt.Errorf("ct × mask: %w", err))
	}
	if err := eval.Rescale(out, out); err != nil {
		panic(fmt.Errorf("ct × mask Rescale: %w", err))
	}
	return out
}

// addCt returns a + b as a new ciphertext. Caller must ensure matching
// scale and level (which the BMM-III LongRot structure guarantees).
func addCt(eval *ckks.Evaluator, a, b *rlwe.Ciphertext) *rlwe.Ciphertext {
	out, err := eval.AddNew(a, b)
	if err != nil {
		panic(fmt.Errorf("add: %w", err))
	}
	return out
}

// ---------------------------------------------------------------------------
// LongRot — the workhorse of BMM-III
//
// Direct translation of the Python `_long_rot_he`. The code is dense, but
// every branch below mirrors a branch in the Python — comments mark
// which "Step" and sub-case each block corresponds to.
// ---------------------------------------------------------------------------

// longRotSelect is the Python `_select`:
//
//	v_t == 0     : identity  → return ct_a
//	v_t == n_he  : next chunk → return ct_b
//	otherwise    : ct_a · mask[0, n_he−v_t] + ct_b · mask[n_he−v_t, n_he]
//
// The result is a "stitched" chunk that covers slot positions [v_t, n_he)
// from ct_a and [0, v_t) from ct_b.
func longRotSelect(eval *ckks.Evaluator, mfn maskFn,
	vT, nHE int, ctA, ctB *rlwe.Ciphertext,
) *rlwe.Ciphertext {
	switch {
	case vT == 0:
		return ctA
	case vT == nHE:
		return ctB
	default:
		l, _ := eval.MulNew(ctA, mfn(0, nHE-vT))   // L, S·Q[L-1]
		r, _ := eval.MulNew(ctB, mfn(nHE-vT, nHE)) // L, S·Q[L-1]
		sum, _ := eval.AddNew(l, r)                // L, S·Q[L-1]
		eval.Rescale(sum, sum)                     // L-1, S
		return sum
	}
}

// longRotHE: rotate a logical vector of length encLen (stored as encCts,
// w chunks of nHE slots each) by `rot` positions, producing
// stopLength = ceil(outputLen / nHE) output chunks.
//
// If `rotateDict != nil`, Step 1 (per-chunk rotation) reads from that
// precomputed map instead of calling eval.RotateNew. This enables the
// hoisted mode.
func longRotHE(
	eval *ckks.Evaluator,
	encCts []*rlwe.Ciphertext,
	rot, outputLen, encLen, nHE int,
	mfn maskFn,
	rotateDict map[int]map[int]*rlwe.Ciphertext, // chunk_idx → v_tmp → rotated ct
) []*rlwe.Ciphertext {

	rot = posMod3(rot, encLen)
	w := len(encCts)
	v := posMod3(rot, nHE)
	u := rot / nHE
	stopLength := intCeilDiv(outputLen, nHE)
	lastR := outputLen % nHE
	rWrap := encLen % nHE

	// ── Step 1: rotate each source chunk by v_tmp. ───────────────────────
	rotateCipher := make([]*rlwe.Ciphertext, 0, stopLength+2)
	{
		vTmp := v
		k := 1
		i := 0
		for i < stopLength+k {
			idx := posMod3(u+i, w)
			var ct *rlwe.Ciphertext
			switch {
			case vTmp == 0:
				ct = encCts[idx]
			case rotateDict != nil:
				ct = rotateDict[idx][vTmp]
				if ct == nil {
					panic(fmt.Errorf("longRotHE: hoist dict missing (idx=%d, vTmp=%d)", idx, vTmp))
				}
			default:
				var err error
				ct, err = eval.RotateNew(encCts[idx], vTmp)
				if err != nil {
					panic(fmt.Errorf("longRotHE Step 1 Rotate (idx=%d, vTmp=%d): %w",
						idx, vTmp, err))
				}
			}
			rotateCipher = append(rotateCipher, ct)

			// Advance v_tmp — mirrors the Python branches exactly.
			if idx == w-1 {
				if vTmp <= rWrap {
					vTmp = posMod3(nHE-rWrap+vTmp, nHE)
					if vTmp == 0 {
						k++
					}
				} else {
					vTmp = posMod3(vTmp-rWrap, nHE)
					k++
				}
			}
			i++
		}
	}

	// ── Step 2: stitch rotated chunks into output chunks. ────────────────
	destination := make([]*rlwe.Ciphertext, 0, stopLength)
	vTmp := v
	i := 0

	for len(destination) < stopLength-1 {
		// Case A: last input chunk, v_tmp fits within the short wrap —
		// skip ahead (no output this iter).
		if posMod3(u+i, w) == w-1 && vTmp <= rWrap {
			vTmp = posMod3(nHE-rWrap+vTmp, nHE)
			if vTmp == 0 {
				i++
				continue
			}
		}

		// Case B: second-to-last chunk, v_tmp > rWrap — output spans
		// THREE source chunks.
		if posMod3(u+i, w) == w-2 && vTmp > rWrap {
			m1 := mfn(0, nHE-vTmp)
			m2 := mfn(nHE-vTmp, nHE-vTmp+rWrap)
			m3 := mfn(nHE-vTmp+rWrap, nHE)

			p1 := ctMulMaskRescale(eval, rotateCipher[posMod3(i, len(rotateCipher))], m1)
			p2 := ctMulMaskRescale(eval, rotateCipher[posMod3(i+1, len(rotateCipher))], m2)
			p3 := ctMulMaskRescale(eval, rotateCipher[posMod3(i+2, len(rotateCipher))], m3)

			out := addCt(eval, p1, p2)
			out = addCt(eval, out, p3)
			destination = append(destination, out)

			vTmp = posMod3(vTmp-rWrap, nHE)
			i += 2
			continue
		}

		// Default case: classic rotate-stitch across two source chunks.
		destination = append(destination, longRotSelect(eval, mfn, vTmp, nHE,
			rotateCipher[posMod3(i, len(rotateCipher))],
			rotateCipher[posMod3(i+1, len(rotateCipher))]))
		i++
	}

	// ── Step 3: produce the final (possibly partial) output chunk. ───────
	if len(destination) == stopLength-1 {
		lr := lastR
		if lr == 0 {
			lr = nHE
		}

		if posMod3(u+i, w) == w-2 && vTmp > rWrap {
			switch {
			case lr <= nHE-vTmp:
				destination = append(destination,
					ctMulMaskRescale(eval,
						rotateCipher[posMod3(i, len(rotateCipher))],
						mfn(0, lr)))
			case lr <= nHE-vTmp+rWrap:
				p1 := ctMulMaskRescale(eval,
					rotateCipher[posMod3(i, len(rotateCipher))],
					mfn(0, nHE-vTmp))
				p2 := ctMulMaskRescale(eval,
					rotateCipher[posMod3(i+1, len(rotateCipher))],
					mfn(nHE-vTmp, lr))
				destination = append(destination, addCt(eval, p1, p2))
			default:
				p1 := ctMulMaskRescale(eval,
					rotateCipher[posMod3(i, len(rotateCipher))],
					mfn(0, nHE-vTmp))
				p2 := ctMulMaskRescale(eval,
					rotateCipher[posMod3(i+1, len(rotateCipher))],
					mfn(nHE-vTmp, nHE-vTmp+rWrap))
				p3 := ctMulMaskRescale(eval,
					rotateCipher[posMod3(i+2, len(rotateCipher))],
					mfn(nHE-vTmp+rWrap, lr))
				out := addCt(eval, p1, p2)
				out = addCt(eval, out, p3)
				destination = append(destination, out)
			}
		} else {
			if posMod3(u+i, w) == w-1 && vTmp <= rWrap {
				vTmp = posMod3(nHE-rWrap+vTmp, nHE)
				if vTmp == 0 {
					i++
				}
			}
			if lr <= nHE-vTmp {
				destination = append(destination,
					ctMulMaskRescale(eval,
						rotateCipher[posMod3(i, len(rotateCipher))],
						mfn(0, lr)))
			} else {
				p1 := ctMulMaskRescale(eval,
					rotateCipher[posMod3(i, len(rotateCipher))],
					mfn(0, nHE-vTmp))
				p2 := ctMulMaskRescale(eval,
					rotateCipher[posMod3(i+1, len(rotateCipher))],
					mfn(nHE-vTmp, lr))
				destination = append(destination, addCt(eval, p1, p2))
			}
		}
	}
	return destination
}

// ---------------------------------------------------------------------------
// Hoisted-mode planning + precompute
// ---------------------------------------------------------------------------

// buildPlan mirrors Python `_build_plan`: walk Step 1 of LongRot in
// plan-only mode, recording which chunks need which rotation amounts.
// No ciphertexts are touched.
func buildPlan(rot, outputLen, encLen, nHE, w int) map[int]map[int]struct{} {
	rot = posMod3(rot, encLen)
	v := posMod3(rot, nHE)
	u := rot / nHE
	stopLength := intCeilDiv(outputLen, nHE)
	rWrap := encLen % nHE

	needed := make(map[int]map[int]struct{})
	vTmp := v
	k := 1
	i := 0
	for i < stopLength+k {
		idx := posMod3(u+i, w)
		if vTmp != 0 {
			if _, ok := needed[idx]; !ok {
				needed[idx] = make(map[int]struct{})
			}
			needed[idx][vTmp] = struct{}{}
		}
		if idx == w-1 {
			if vTmp <= rWrap {
				vTmp = posMod3(nHE-rWrap+vTmp, nHE)
				if vTmp == 0 {
					k++
				}
			} else {
				vTmp = posMod3(vTmp-rWrap, nHE)
				k++
			}
		}
		i++
	}
	return needed
}

// precomputeHoisted takes a block of rotation amounts, unions the v_tmp
// values each chunk needs, and calls eval.RotateHoistedNew once per chunk.
func precomputeHoisted(
	eval *ckks.Evaluator,
	rotsBlock []int,
	outputLen, encLen, nHE int,
	encCts []*rlwe.Ciphertext,
) map[int]map[int]*rlwe.Ciphertext {
	w := len(encCts)

	union := make(map[int]map[int]struct{})
	for _, rot := range rotsBlock {
		per := buildPlan(rot, outputLen, encLen, nHE, w)
		for idx, vset := range per {
			if _, ok := union[idx]; !ok {
				union[idx] = make(map[int]struct{})
			}
			for v := range vset {
				union[idx][v] = struct{}{}
			}
		}
	}

	hoisted := make(map[int]map[int]*rlwe.Ciphertext, len(union))
	for idx, vset := range union {
		if len(vset) == 0 {
			continue
		}
		shifts := make([]int, 0, len(vset))
		for v := range vset {
			shifts = append(shifts, v)
		}
		sort.Ints(shifts)
		rotated, err := eval.RotateHoistedNew(encCts[idx], shifts)
		if err != nil {
			panic(fmt.Errorf("precomputeHoisted chunk %d: %w", idx, err))
		}
		hoisted[idx] = rotated
	}
	return hoisted
}

// ---------------------------------------------------------------------------
// Required-rotations helper (for ctx.WithRotations)
// ---------------------------------------------------------------------------

// encodedChunkCount returns the number of chunks `break_into_chunks`
// produces for an encoding of length `encLen` at `nHE` slots, given that
// LongRot may read up to `outputLen + nHE` slots. Matches the Python
// `break_into_chunks` length exactly.
func encodedChunkCount(encLen, outputLen, nHE int) int {
	needed := outputLen + nHE
	length := encLen
	if encLen < needed {
		reps := intCeilDiv(needed, encLen) + 1
		length = reps * encLen
	}
	return intCeilDiv(length, nHE)
}

// RequiredBmm3Rotations enumerates every distinct v_tmp any LongRot call
// will use across all m iterations, for both the A and B encodings. Pass
// to ctx.WithRotations to generate exactly the needed Galois keys.
func RequiredBmm3Rotations(n, m, p, nHE int) []int {
	r := smallestR(n, m, p)
	nm, mp, np := n*m, m*p, n*p
	wA := encodedChunkCount(nm, np, nHE)
	wB := encodedChunkCount(mp, np, nHE)

	set := map[int]struct{}{}
	for i := 0; i < m; i++ {
		rotA := posMod3(-i*n, nm)
		rotB := posMod3((r*m-n)*i, mp)
		for _, vset := range buildPlan(rotA, np, nm, nHE, wA) {
			for v := range vset {
				set[v] = struct{}{}
			}
		}
		for _, vset := range buildPlan(rotB, np, mp, nHE, wB) {
			for v := range vset {
				set[v] = struct{}{}
			}
		}
	}
	out := make([]int, 0, len(set))
	for v := range set {
		out = append(out, v)
	}
	return out
}

// ---------------------------------------------------------------------------
// Top-level dispatcher — three modes
// ---------------------------------------------------------------------------

// MatMulBmm3HE dispatches to the chosen mode. Returns the output chunks;
// the caller concatenates and BicyclicDecode's them.
//
// Regardless of mode, returned ciphertexts are in canonical form:
// degree 1, scale S, level L − 2, ready to be consumed by any subsequent
// HE operation.
func MatMulBmm3HE(
	eval *ckks.Evaluator,
	ctx *HEContext,
	aCts, bCts []*rlwe.Ciphertext,
	n, m, p, nHE, inputLevel int,
	mode Bmm3Mode,
	hoistBlockSize int,
) []*rlwe.Ciphertext {
	switch mode {
	case Bmm3ModeNaive:
		fmt.Println("BMM3 hoisting: none")
		return bmm3HENaive(eval, ctx, aCts, bCts, n, m, p, nHE, inputLevel)
	case Bmm3ModeCached:
		fmt.Println("BMM3 hoisting: cached")
		return bmm3HECached(eval, ctx, aCts, bCts, n, m, p, nHE, inputLevel)
	case Bmm3ModeHoisted:
		fmt.Println("BMM3 hoisting: hoisted")
		return bmm3HEHoisted(eval, ctx, aCts, bCts, n, m, p, nHE, inputLevel, hoistBlockSize)
	default:
		panic(fmt.Errorf("unknown BMM-III mode %q", mode))
	}
}

// bmm3Loop is the common per-iteration body. It performs the LongRot of
// A and B for one outer iteration, multiplies the rotated chunks ct × ct
// using `Mul` (NOT `MulRelin`), and accumulates the products into dest
// at scale S² without per-iteration Rescale. Both Relinearize and the
// post-product Rescale are deferred to bmm3Finalize.
//
// Invariants per output slot s after this function returns:
//   - dest[s] is degree 2 (3 polynomial components)
//   - dest[s] is at level L − 1 and scale S²
//   - dest[s] holds the partial sum across all calls so far
func bmm3Loop(
	eval *ckks.Evaluator,
	aCts, bCts []*rlwe.Ciphertext,
	rotA, rotB int,
	nm, mp, np, nHE int,
	mfn maskFn,
	rotateDictA, rotateDictB map[int]map[int]*rlwe.Ciphertext,
	dest []*rlwe.Ciphertext,
) []*rlwe.Ciphertext {

	stop := intCeilDiv(np, nHE)
	aRot := longRotHE(eval, aCts, rotA, np, nm, nHE, mfn, rotateDictA)
	bRot := longRotHE(eval, bCts, rotB, np, mp, nHE, mfn, rotateDictB)

	if dest == nil {
		dest = make([]*rlwe.Ciphertext, stop)
	}
	for s := 0; s < stop; s++ {
		// Both a_rot[s] and b_rot[s] are at level L-1, scale S, degree 1.
		// MulNew (not MulRelinNew) → degree-2 product at scale S². The
		// matching Rescale and Relinearize are deferred to bmm3Finalize.
		prod, err := eval.MulNew(aRot[s], bRot[s])
		if err != nil {
			panic(fmt.Errorf("bmm3Loop Mul (s=%d): %w", s, err))
		}

		if dest[s] == nil {
			dest[s] = prod
		} else {
			// dest[s] and prod are both degree 2 at level L-1, scale S²;
			// Add preserves degree, level, and scale.
			if err := eval.Add(dest[s], prod, dest[s]); err != nil {
				panic(fmt.Errorf("bmm3Loop Add (s=%d): %w", s, err))
			}
		}
	}
	return dest
}

// bmm3Finalize completes the deferred work for every accumulated output
// chunk. It is called exactly once per BMM-III invocation, after all m
// outer iterations have been folded into dest.
//
// For each non-nil dest[s] (which is degree 2, scale S², level L − 1):
//  1. Rescale     : (deg 2, S², L − 1) → (deg 2, S, L − 2).
//  2. Relinearize : (deg 2, S,  L − 2) → (deg 1, S, L − 2).
//
// This is where lazy rescaling and lazy relinearization pay their bill.
// Total cost per call: ⌈np/n_he⌉ Rescales and ⌈np/n_he⌉ Relins —
// independent of m (the number of outer iterations).
func bmm3Finalize(eval *ckks.Evaluator, dest []*rlwe.Ciphertext) {
	for s := range dest {
		if dest[s] == nil {
			continue
		}
		if err := eval.Rescale(dest[s], dest[s]); err != nil {
			panic(fmt.Errorf("bmm3Finalize Rescale (s=%d): %w", s, err))
		}
		if err := eval.Relinearize(dest[s], dest[s]); err != nil {
			panic(fmt.Errorf("bmm3Finalize Relinearize (s=%d): %w", s, err))
		}
	}
}

// bmm3HENaive — Mode: naive — fresh mask encode every call.
func bmm3HENaive(
	eval *ckks.Evaluator, ctx *HEContext,
	aCts, bCts []*rlwe.Ciphertext,
	n, m, p, nHE, inputLevel int,
) []*rlwe.Ciphertext {
	r := smallestR(n, m, p)
	nm, mp, np := n*m, m*p, n*p

	mc := newMaskCache(ctx, inputLevel-1)
	mc.cache = nil // naive: every get() re-encodes
	mfn := mc.get

	var dest []*rlwe.Ciphertext
	for i := 0; i < m; i++ {
		rotA := posMod3(-i*n, nm)
		rotB := posMod3((r*m-n)*i, mp)
		dest = bmm3Loop(eval, aCts, bCts, rotA, rotB, nm, mp, np, nHE, mfn, nil, nil, dest)
	}
	bmm3Finalize(eval, dest)
	return dest
}

// bmm3HECached — Mode: cached — reuse encoded plaintext masks.
func bmm3HECached(
	eval *ckks.Evaluator, ctx *HEContext,
	aCts, bCts []*rlwe.Ciphertext,
	n, m, p, nHE, inputLevel int,
) []*rlwe.Ciphertext {
	r := smallestR(n, m, p)
	nm, mp, np := n*m, m*p, n*p

	mc := newMaskCache(ctx, inputLevel-1)
	mfn := mc.get

	beforeLoop := TakeMemSnap()
	var dest []*rlwe.Ciphertext
	for i := 0; i < m; i++ {
		rotA := posMod3(-i*n, nm)
		rotB := posMod3((r*m-n)*i, mp)
		dest = bmm3Loop(eval, aCts, bCts, rotA, rotB, nm, mp, np, nHE, mfn, nil, nil, dest)
	}
	afterLoop := TakeMemSnap()
	PrintMemDelta("BMM3 bmm3Loop", beforeLoop, afterLoop)

	bmm3Finalize(eval, dest)
	return dest
}

// bmm3HEHoisted — Mode: hoisted — cached masks + block hoisting of rotations.
func bmm3HEHoisted(
	eval *ckks.Evaluator, ctx *HEContext,
	aCts, bCts []*rlwe.Ciphertext,
	n, m, p, nHE, inputLevel int,
	hoistBlockSize int,
) []*rlwe.Ciphertext {
	if hoistBlockSize <= 0 {
		hoistBlockSize = 8
	}
	r := smallestR(n, m, p)
	nm, mp, np := n*m, m*p, n*p

	mc := newMaskCache(ctx, inputLevel-1)
	mfn := mc.get

	// Precompute all rotation amounts up front.
	rotsA := make([]int, m)
	rotsB := make([]int, m)
	for i := 0; i < m; i++ {
		rotsA[i] = posMod3(-i*n, nm)
		rotsB[i] = posMod3((r*m-n)*i, mp)
	}

	var dest []*rlwe.Ciphertext
	for base := 0; base < m; base += hoistBlockSize {
		end := base + hoistBlockSize
		if end > m {
			end = m
		}

		beforeHoist := TakeMemSnap()
		hoistedA := precomputeHoisted(eval, rotsA[base:end], np, nm, nHE, aCts)
		hoistedB := precomputeHoisted(eval, rotsB[base:end], np, mp, nHE, bCts)
		afterHoist := TakeMemSnap()
		PrintMemDelta("BMM3 precomputeHoisted", beforeHoist, afterHoist)

		for i := base; i < end; i++ {
			dest = bmm3Loop(eval, aCts, bCts, rotsA[i], rotsB[i],
				nm, mp, np, nHE, mfn, hoistedA, hoistedB, dest)
		}
		// Go GC reclaims hoistedA/hoistedB when this iteration exits.
	}

	bmm3Finalize(eval, dest)
	return dest
}
