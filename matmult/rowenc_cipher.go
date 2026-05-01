// row_cipher.go
//
// HE implementation of row-packing matrix multiplication on Lattigo v6.
// Port of ciphertext/rowEnc_cipher.py.
//
// Multiplicative depth
// --------------------
//   extract_col / extract_row : ct × pt + Rescale → consumes 1 level
//   final ct × ct + Rescale   :                   → consumes 1 level
//   Total depth: 2
//
// Scale management
// ----------------
// Same staggered-scale trick as the other kernels:
//   - Inputs A, B encrypted at the default scale S, level L.
//   - Extraction masks encoded at scale Q[L], so ct × mask after Rescale
//     lands at scale S, level L−1.
//   - Final ct_A_rep × ct_B_rep is two ciphertexts at scale S; product at
//     scale S² lands at scale S after one Rescale at level L−2.
//
// All additions in the kernel happen at matching scale; no runtime
// reconciliation needed.

package main

import (
	"fmt"
	"math/bits"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ---------------------------------------------------------------------------
// Rotation set helper
// ---------------------------------------------------------------------------

// RequiredRowRotations enumerates the distinct slot shifts the kernel
// performs at problem size n. Pass to ctx.WithRotations before invoking
// the kernel.
//
// Lattigo rotations are cyclic over the full nHE slot range, NOT over
// s = n². The replication-step shifts are translated accordingly:
//
//   - rowReplicateCol :  nHE − 2^k        for k ∈ [0, log₂n)
//   - rowReplicateRow :  n·i              for i ∈ [1, n)         (initial align)
//     nHE − (n·2^k mod nHE)  for k ∈ [0, log₂n)
//   - diagonal align  :  i                for i ∈ [1, n)
func RequiredRowRotations(n, nHE int) []int {
	log2n := bits.TrailingZeros(uint(n))

	set := map[int]struct{}{}

	// replicate_col shifts (same for every iteration)
	for k := 0; k < log2n; k++ {
		shift := nHE - (1 << k)
		set[shift] = struct{}{}
	}

	// replicate_row downstream shifts (same for every iteration)
	for k := 0; k < log2n; k++ {
		shift := (nHE - (n*(1<<k))%nHE) % nHE
		if shift != 0 {
			set[shift] = struct{}{}
		}
	}

	// per-iteration shifts
	for i := 1; i < n; i++ {
		// initial align in replicate_row
		if (n*i)%nHE != 0 {
			set[n*i] = struct{}{}
		}
		// diagonal alignment of A_rep
		set[i] = struct{}{}
	}

	out := make([]int, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	return out
}

// ---------------------------------------------------------------------------
// Mask plaintext cache
// ---------------------------------------------------------------------------

// RowMasks bundles the 2n pre-encoded extraction masks the kernel needs.
// Built once per problem size and reused across trials and across multiple
// kernel calls.
type RowMasks struct {
	Cols []*rlwe.Plaintext // ColMasks[i] isolates column i
	Rows []*rlwe.Plaintext // RowMasks[i] isolates row i
}

// BuildRowMasks pre-encodes every extraction mask once. They are encoded
// at scale Q[level] so that ct × mask after Rescale lands at scale S.
func BuildRowMasks(ctx *HEContext, n, level int) RowMasks {
	s := n * n
	scaleMask := rlwe.NewScale(ctx.Params.Q()[level])

	cols := make([]*rlwe.Plaintext, n)
	rows := make([]*rlwe.Plaintext, n)
	for i := 0; i < n; i++ {
		cols[i] = EncodePlaintext(ctx.Encoder, ctx.Params,
			PadToSlots(makeColMask(s, i, n), ctx.NHE), level, scaleMask)
		rows[i] = EncodePlaintext(ctx.Encoder, ctx.Params,
			PadToSlots(makeRowMask(s, i, n), ctx.NHE), level, scaleMask)
	}
	return RowMasks{Cols: cols, Rows: rows}
}

// ---------------------------------------------------------------------------
// Per-step HE helpers
// ---------------------------------------------------------------------------

// extractColHE = ct × ColMask[i] + Rescale.
// Result: degree 1, level L−1, scale S.
func extractColHE(eval *ckks.Evaluator, ct *rlwe.Ciphertext, mask *rlwe.Plaintext) *rlwe.Ciphertext {
	out, err := eval.MulNew(ct, mask)
	if err != nil {
		panic(fmt.Errorf("extractColHE Mul: %w", err))
	}
	if err := eval.Rescale(out, out); err != nil {
		panic(fmt.Errorf("extractColHE Rescale: %w", err))
	}
	return out
}

// extractRowHE = ct × RowMask[i] + Rescale.
func extractRowHE(eval *ckks.Evaluator, ct *rlwe.Ciphertext, mask *rlwe.Plaintext) *rlwe.Ciphertext {
	out, err := eval.MulNew(ct, mask)
	if err != nil {
		panic(fmt.Errorf("extractRowHE Mul: %w", err))
	}
	if err := eval.Rescale(out, out); err != nil {
		panic(fmt.Errorf("extractRowHE Rescale: %w", err))
	}
	return out
}

// replicateColHE spreads each isolated column value rightward.
//
// Implementation note: this is a sequence of ct + Rot(ct, shift_k), where
// each shift depends only on k, not on the iteration. The chain is short
// (log₂n) and rotations are applied to the running accumulator (a moving
// target), so hoisting does not directly apply here.
//
// IMPORTANT: Lattigo rotations are cyclic over the FULL nHE slot range,
// not over s = n². The Python code's `roll(-(2^k))` corresponds to a
// right-shift by 2^k, which in Lattigo terms is left-rotation by
// (nHE − 2^k). Slots [s, nHE) start as zero, so the result inside the
// first s slots matches the intended cyclic-on-s behavior, while the
// tail slots remain zero — exactly what BicyclicDecode-style readout
// expects.
func replicateColHE(eval *ckks.Evaluator, ct *rlwe.Ciphertext, n, nHE int) *rlwe.Ciphertext {
	log2n := bits.TrailingZeros(uint(n))
	out := ct
	for k := 0; k < log2n; k++ {
		shift := nHE - (1 << k)
		rot, err := eval.RotateNew(out, shift)
		if err != nil {
			panic(fmt.Errorf("replicateColHE Rotate (k=%d): %w", k, err))
		}
		if err := eval.Add(out, rot, rot); err != nil {
			panic(err)
		}
		out = rot
	}
	return out
}

// replicateRowHE first brings row i to slot 0, then spreads it downward.
//
// The initial alignment is left-rotation by n·i (the Python `roll(n*i)`
// translates directly because positive shifts are the same direction in
// both conventions). The downstream replication shifts are right-shifts
// by n·2^k in the active window, which in Lattigo's nHE-cyclic model is
// left-rotation by (nHE − n·2^k).
func replicateRowHE(eval *ckks.Evaluator, ct *rlwe.Ciphertext, i, n, nHE int) *rlwe.Ciphertext {
	log2n := bits.TrailingZeros(uint(n))

	var out *rlwe.Ciphertext
	if (n*i)%nHE == 0 {
		out = ct
	} else {
		var err error
		out, err = eval.RotateNew(ct, n*i)
		if err != nil {
			panic(fmt.Errorf("replicateRowHE align Rotate: %w", err))
		}
	}

	for k := 0; k < log2n; k++ {
		shift := (nHE - (n*(1<<k))%nHE) % nHE
		if shift == 0 {
			continue
		}
		rot, err := eval.RotateNew(out, shift)
		if err != nil {
			panic(fmt.Errorf("replicateRowHE Rotate (k=%d): %w", k, err))
		}
		if err := eval.Add(out, rot, rot); err != nil {
			panic(err)
		}
		out = rot
	}
	return out
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------

// RowPackMatMulHE runs the row-packing matmul on encrypted inputs.
//
// ctA, ctB : row-packed ciphertexts at level inputLevel, default scale.
// masks    : output of BuildRowMasks at level=inputLevel (scale Q[L]).
// n        : matrix dimension; must be a power of 2.
// nHE      : number of HE slots (Lattigo rotations are cyclic over nHE,
//
//	so the per-step rotation amounts depend on it).
//
// Returns one ciphertext at level inputLevel − 2 holding the row-packed
// product C = A · B.
func RowPackMatMulHE(
	eval *ckks.Evaluator,
	ctA, ctB *rlwe.Ciphertext,
	masks RowMasks,
	n, nHE int,
) *rlwe.Ciphertext {
	if n == 0 || n&(n-1) != 0 {
		panic(fmt.Errorf("RowPackMatMulHE: n=%d must be a positive power of 2", n))
	}

	var acc *rlwe.Ciphertext

	for i := 0; i < n; i++ {
		aCol := extractColHE(eval, ctA, masks.Cols[i])
		bRow := extractRowHE(eval, ctB, masks.Rows[i])

		aRep := replicateColHE(eval, aCol, n, nHE)
		bRep := replicateRowHE(eval, bRow, i, n, nHE)

		// Diagonal alignment of A_rep (left rotate by i).
		if i != 0 {
			rot, err := eval.RotateNew(aRep, i)
			if err != nil {
				panic(fmt.Errorf("diag align Rotate (i=%d): %w", i, err))
			}
			aRep = rot
		}

		// Final ct × ct.
		prod, err := eval.MulNew(aRep, bRep)
		if err != nil {
			panic(fmt.Errorf("final MulRelin (i=%d): %w", i, err))
		}
		if err := eval.Rescale(prod, prod); err != nil {
			panic(fmt.Errorf("final Rescale (i=%d): %w", i, err))
		}

		if acc == nil {
			acc = prod
		} else {
			if err := eval.Add(acc, prod, acc); err != nil {
				panic(err)
			}
		}
	}
	if acc != nil {
		if err := eval.Relinearize(acc, acc); err != nil {
			panic(fmt.Errorf("final relin: %w", err))
		}
	}
	return acc
}
