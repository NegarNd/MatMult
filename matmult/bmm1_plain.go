// bmm1_plain.go
//
// Plaintext simulation of BMM-I bicyclic matrix multiplication
// (Zheng et al., IEEE TIFS 2024). Port of plaintext/bmm1_plain.py.
//
// Bicyclic encoding
// -----------------
// An n×m matrix A is encoded into a length-nm slot vector by
//
//     vec[k] = A[k mod n][k mod m]      for k = 0 .. n·m − 1
//
// When s_n, s_m, s_p are pairwise coprime, this lets matrix multiplication
// be done with only rotations and pointwise products — no masks.
//
// Single-block algorithm bmm1(A, B, n, m, p):
//   1. Tile A_enc and B_enc so each is long enough for any rotation below.
//   2. For i in [0, m):
//        C += Rot(A_tile, (i·n·p) mod (n·m))[:n·p]
//           · Rot(B_tile, (i·n·p) mod (m·p))[:n·p]
//
// Block algorithm bmm1_matmul runs bmm1 on each (i, k, j) block and
// accumulates. When dims == s_dims there is exactly one block and this
// degenerates to a single bmm1 call.

package main

import "math"

// ---------------------------------------------------------------------------
// Encoding / decoding
// ---------------------------------------------------------------------------

// BicyclicEncode returns the length-(n·m) slot vector with
//
//	vec[k] = M[k mod n][k mod m].
//
// Port of plaintext/bmm1_plain.py → bicyclic_encode.
func BicyclicEncode(M [][]float64) []float64 {
	n := len(M)
	m := len(M[0])
	out := make([]float64, n*m)
	for k := 0; k < n*m; k++ {
		out[k] = M[k%n][k%m]
	}
	return out
}

// BicyclicDecode recovers an n×p matrix C from a bicyclic-encoded vector
// of length n·p:  C[k mod n][k mod p] = vec[k].
// Port of plaintext/bmm1_plain.py → bicyclic_decode.
func BicyclicDecode(vec []float64, n, p int) [][]float64 {
	C := make([][]float64, n)
	for i := range C {
		C[i] = make([]float64, p)
	}
	for k := 0; k < n*p; k++ {
		C[k%n][k%p] = vec[k]
	}
	return C
}

// RepeatVector tiles vec `times` times. This is the plaintext analogue of
// the Python `repeat_vector`: it produces a vector long enough that any
// rotation used in bmm1 leaves the first n·p slots populated with valid
// (non-zero-padding) data.
func RepeatVector(vec []float64, times int) []float64 {
	out := make([]float64, len(vec)*times)
	for i := 0; i < times; i++ {
		copy(out[i*len(vec):(i+1)*len(vec)], vec)
	}
	return out
}

// EncodeBlocks splits A (N×M) and B (M×P) into blocks of shape
// (s_n×s_m) and (s_m×s_p) respectively, bicyclic-encodes each one.
// Returns
//
//	aEnc[i][k] = BicyclicEncode(A[i·s_n : (i+1)·s_n, k·s_m : (k+1)·s_m])
//	bEnc[k][j] = BicyclicEncode(B[k·s_m : (k+1)·s_m, j·s_p : (j+1)·s_p])
func EncodeBlocks(A, B [][]float64, sN, sM, sP int) (aEnc, bEnc [][][]float64) {
	N := len(A)
	M := len(A[0])
	P := len(B[0])

	aRows := N / sN
	kBlocks := M / sM
	bCols := P / sP

	aEnc = make([][][]float64, aRows)
	for i := 0; i < aRows; i++ {
		aEnc[i] = make([][]float64, kBlocks)
		for k := 0; k < kBlocks; k++ {
			aEnc[i][k] = BicyclicEncode(subMatrix(A, i*sN, (i+1)*sN, k*sM, (k+1)*sM))
		}
	}
	bEnc = make([][][]float64, kBlocks)
	for k := 0; k < kBlocks; k++ {
		bEnc[k] = make([][]float64, bCols)
		for j := 0; j < bCols; j++ {
			bEnc[k][j] = BicyclicEncode(subMatrix(B, k*sM, (k+1)*sM, j*sP, (j+1)*sP))
		}
	}
	return aEnc, bEnc
}

// subMatrix returns M[rowStart:rowEnd, colStart:colEnd] as a freshly
// allocated [][]float64.
func subMatrix(M [][]float64, rowStart, rowEnd, colStart, colEnd int) [][]float64 {
	out := make([][]float64, rowEnd-rowStart)
	for i := range out {
		row := make([]float64, colEnd-colStart)
		copy(row, M[rowStart+i][colStart:colEnd])
		out[i] = row
	}
	return out
}

// ---------------------------------------------------------------------------
// Single-block BMM-I (plaintext, with op counter)
// ---------------------------------------------------------------------------

// bmm1Plain executes BMM-I on a single bicyclic-encoded block and returns
// (C_vec, counts). C_vec has length n·p.
//
// Rotations and pointwise products are accumulated into the counter, which
// the caller can merge across all block calls.
func bmm1Plain(aEnc, bEnc []float64, n, m, p int, counts *OpCounts) []float64 {
	step := n * p

	// Tile so that rotations don't wrap into zero padding.
	aTiles := intCeilDiv(n*m+n*p, n*m) // enough to cover max rotation (n*m − 1) + n·p slots
	bTiles := intCeilDiv(m*p+n*p, m*p)
	aRep := RepeatVector(aEnc, aTiles)
	bRep := RepeatVector(bEnc, bTiles)

	C := make([]float64, step)
	for i := 0; i < m; i++ {
		rotA := (i * step) % (n * m)
		rotB := (i * step) % (m * p)

		aRot := rotateVec(aRep, rotA)
		bRot := rotateVec(bRep, rotB)
		if rotA != 0 {
			counts.Rotations++
		}
		if rotB != 0 {
			counts.Rotations++
		}

		for k := 0; k < step; k++ {
			C[k] += aRot[k] * bRot[k]
		}
		counts.CtCtMuls++
	}
	return C
}

// rotateVec returns v left-rotated by k positions (modulo len(v)). This is
// a plaintext analogue; the HE kernel uses eval.Rotate which is cyclic on
// the slot vector and matches this semantics on the first len(v) slots.
func rotateVec(v []float64, k int) []float64 {
	n := len(v)
	if n == 0 {
		return nil
	}
	k = ((k % n) + n) % n
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = v[(i+k)%n]
	}
	return out
}

func intCeilDiv(a, b int) int {
	return (a + b - 1) / b
}

// ---------------------------------------------------------------------------
// Block BMM-I — falls back to single-block when dims == s_dims
// ---------------------------------------------------------------------------

// bmm1MatMulPlain runs block BMM-I over pre-encoded block arrays and
// returns the (N×P) product matrix alongside the cumulative op counts.
//
// aEnc[i][k] and bEnc[k][j] are the bicyclic encodings produced by
// EncodeBlocks.
func bmm1MatMulPlain(
	aEnc, bEnc [][][]float64,
	N, M, P, sN, sM, sP int,
) ([][]float64, OpCounts) {

	aRows := N / sN
	kBlocks := M / sM
	bCols := P / sP

	var counts OpCounts

	C := make([][]float64, N)
	for i := range C {
		C[i] = make([]float64, P)
	}

	for i := 0; i < aRows; i++ {
		for j := 0; j < bCols; j++ {
			acc := make([]float64, sN*sP)
			for k := 0; k < kBlocks; k++ {
				block := bmm1Plain(aEnc[i][k], bEnc[k][j], sN, sM, sP, &counts)
				for t := range acc {
					acc[t] += block[t]
				}
			}
			// Decode into the right sub-block of C.
			dec := BicyclicDecode(acc, sN, sP)
			for ii := 0; ii < sN; ii++ {
				copy(C[i*sN+ii][j*sP:(j+1)*sP], dec[ii])
			}
		}
	}
	return C, counts
}

// ---------------------------------------------------------------------------
// Theoretical cost formulas
// ---------------------------------------------------------------------------

// TheoreticalBmm1Costs returns (n_rot, n_mult, n_add, n_ks) for block
// BMM-I at the given full and block dimensions. Matches the Python
// `theoretical_costs` in plaintext/bmm1_plain.py.
//
//	Per single-block call : 2·s_m rotations, s_m mults, s_m adds.
//	Total blocks          : (N/s_n)·(M/s_m)·(P/s_p).
func TheoreticalBmm1Costs(N, M, P, sN, sM, sP int) (nRot, nMult, nAdd, nKs int) {
	totalBlocks := (N / sN) * (M / sM) * (P / sP)
	nRot = 2 * sM * totalBlocks
	nMult = sM * totalBlocks
	nAdd = sM * totalBlocks
	nKs = nRot + nMult // same total-KS definition as Python
	return
}

// ---------------------------------------------------------------------------
// Buffer-size helper shared with the HE side
// ---------------------------------------------------------------------------

// Bmm1RequiredSlots returns the minimum n_he that encode_and_encrypt_blocks
// expects to find in the slot vector. Matches the Python check:
//
//	max_slots = max(s_n·s_m · ceil((s_m + s_p) / s_m),
//	                s_m·s_p · ceil((s_m + s_n) / s_m))
func Bmm1RequiredSlots(sN, sM, sP int) int {
	a := sN * sM * intCeilDiv(sM+sP, sM)
	b := sM * sP * intCeilDiv(sM+sN, sM)
	return int(math.Max(float64(a), float64(b)))
}
