// row_plain.go
//
// Plaintext simulation of row-packing matrix multiplication.
// Port of plaintext/rowEnc_plain.py.
//
// Encoding
// --------
// An n×n matrix M is row-major flattened into a length-n² slot vector:
//   slot[i*n + j] = M[i][j]
//
// Algorithm (per outer iteration i ∈ [0, n)):
//   1. extract_col(A, i)  — isolate column i of A: every n-th slot
//   2. extract_row(B, i)  — isolate row i of B: contiguous n-slot block
//   3. replicate_col      — spread each column value rightward
//                           via log₂n rotate-and-add steps
//   4. replicate_row(i)   — bring row i to slot 0, then spread downward
//   5. diagonal-align     — rotate A_rep left by i
//   6. accumulate         — C += A_rep · B_rep
//
// Constraint: n must be a power of 2 so the log₂n replication steps land
// on integer rotation amounts.

package main

import (
	"fmt"
	"math/bits"
)

// ---------------------------------------------------------------------------
// Encoding / decoding
// ---------------------------------------------------------------------------

// RowPack flattens an n×n matrix row-major into a length-n² slot vector.
// Port of plaintext/rowEnc_plain.py → row_pack.
func RowPack(M [][]float64) []float64 {
	n := len(M)
	out := make([]float64, n*n)
	for i := 0; i < n; i++ {
		copy(out[i*n:(i+1)*n], M[i])
	}
	return out
}

// RowUnpack inverts RowPack into an n×n matrix.
// Port of plaintext/rowEnc_plain.py → row_unpack.
func RowUnpack(vec []float64, n int) [][]float64 {
	M := make([][]float64, n)
	for i := 0; i < n; i++ {
		M[i] = make([]float64, n)
		copy(M[i], vec[i*n:(i+1)*n])
	}
	return M
}

// ---------------------------------------------------------------------------
// Mask generation
// ---------------------------------------------------------------------------

// makeColMask builds a length-s mask with ones at indices i, i+n, i+2n, ...
// — the "every n-th slot starting at column i" pattern that isolates a
// single column from a row-packed matrix.
func makeColMask(s, i, n int) []float64 {
	mask := make([]float64, s)
	for k := i; k < s; k += n {
		mask[k] = 1.0
	}
	return mask
}

// makeRowMask builds a length-s mask with ones in slots [i·n, i·n + n) and
// zeros elsewhere — isolates row i from a row-packed matrix.
//
// Note: the Python uses `step=1` plus a manual zero-out of the tail; this
// has the same end effect but expresses the intent directly.
func makeRowMask(s, i, n int) []float64 {
	mask := make([]float64, s)
	start := i * n
	for k := 0; k < n; k++ {
		mask[start+k] = 1.0
	}
	return mask
}

// ---------------------------------------------------------------------------
// Extraction + replication helpers (plaintext)
// ---------------------------------------------------------------------------

// rowExtractCol isolates column i of the row-packed matrix in vec.
// Counts as one ct×pt multiplication.
func rowExtractCol(vec []float64, i, n int, counts *OpCounts) []float64 {
	mask := makeColMask(len(vec), i, n)
	out := mulSlots(vec, mask)
	counts.CtPtMuls++
	return out
}

// rowExtractRow isolates row i.
func rowExtractRow(vec []float64, i, n int, counts *OpCounts) []float64 {
	mask := makeRowMask(len(vec), i, n)
	out := mulSlots(vec, mask)
	counts.CtPtMuls++
	return out
}

// rowReplicateCol spreads each isolated column value rightward via
// log₂n rotate-and-add steps. Right shift by 2^k is left rotation by
// (s − 2^k); we use the rotateSlots convention from util.go (left-shift).
func rowReplicateCol(vec []float64, n int, counts *OpCounts) []float64 {
	log2n := bits.TrailingZeros(uint(n))
	out := append([]float64(nil), vec...)
	s := len(vec)
	for k := 0; k < log2n; k++ {
		shift := s - (1 << k) // == roll(-(2^k)) in the Python convention
		rot := rotateSlots(out, shift)
		counts.Rotations++
		out = addSlots(out, rot)
	}
	return out
}

// rowReplicateRow first brings row i to slot 0, then spreads it downward
// over the whole vector with log₂n rotate-and-add steps.
func rowReplicateRow(vec []float64, i, n int, counts *OpCounts) []float64 {
	log2n := bits.TrailingZeros(uint(n))
	s := len(vec)

	out := rotateSlots(vec, n*i) // left-rotate by n*i (Python: roll(n*i))
	if (n*i)%s != 0 {
		counts.Rotations++
	}

	pow2log := 1 << log2n // == n
	for k := 0; k < log2n; k++ {
		// Python:  roll(-(2^log2n) * (2^k))   = right-shift by n · 2^k
		shift := s - (pow2log * (1 << k) % s)
		shift %= s
		rot := rotateSlots(out, shift)
		counts.Rotations++
		out = addSlots(out, rot)
	}
	return out
}

// ---------------------------------------------------------------------------
// Plaintext kernel
// ---------------------------------------------------------------------------

// rowPackMatMulPlain runs the row-packing matmul on flattened inputs and
// returns (C_flat, counts). C_flat has length n².
//
// The structure mirrors the HE version in row_cipher.go line-for-line, so
// the OpCounts returned here predict exactly what the HE kernel performs.
func rowPackMatMulPlain(Af, Bf []float64, n int) ([]float64, OpCounts) {
	if n == 0 || n&(n-1) != 0 {
		panic(fmt.Errorf("row_plain: n=%d must be a positive power of 2", n))
	}

	var counts OpCounts
	C := make([]float64, n*n)

	for i := 0; i < n; i++ {
		aCol := rowExtractCol(Af, i, n, &counts)
		bRow := rowExtractRow(Bf, i, n, &counts)

		aRep := rowReplicateCol(aCol, n, &counts)
		bRep := rowReplicateRow(bRow, i, n, &counts)

		// Diagonal alignment: rotate A_rep left by i.
		// In Python: A_rep.roll(i) — same convention as our rotateSlots.
		if i != 0 {
			aRep = rotateSlots(aRep, i)
			counts.Rotations++
		}

		prod := mulSlots(aRep, bRep)
		counts.CtCtMuls++
		C = addSlots(C, prod)
	}
	return C, counts
}

// ---------------------------------------------------------------------------
// Theoretical cost formulas
// ---------------------------------------------------------------------------

// TheoreticalRowCosts returns (n_rot, n_pmult, n_mult, n_add, n_ks) for
// one row-packing n×n matmul. Matches the formula in
// plaintext/rowEnc_plain.py → theoretical_costs.
//
//	Rot   = n · (2·log₂n + 2)
//	PMult = 2n
//	Mult  = n
//	Add   = n · (2·log₂n + 1)
func TheoreticalRowCosts(n int) (nRot, nPMult, nMult, nAdd, nKs int) {
	log2n := bits.TrailingZeros(uint(n))
	nRot = n * (2*log2n + 2)
	nPMult = 2 * n
	nMult = n
	nAdd = n * (2*log2n + 1)
	nKs = nRot + nMult
	return
}
