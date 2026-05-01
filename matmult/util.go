// util.go
//
// Small utilities shared by every runner: random data generation, the
// plaintext reference matmul, and a few numeric helpers.

package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// RandomMatrix returns an (rows × cols) matrix of standard-normal values
// drawn from the given RNG.
func RandomMatrix(rows, cols int, rng *rand.Rand) [][]float64 {
	M := make([][]float64, rows)
	for i := range M {
		M[i] = make([]float64, cols)
		for j := range M[i] {
			M[i][j] = rng.NormFloat64()
		}
	}
	return M
}

// MatmulPlain is the textbook O(rows · inner · cols) matrix product,
// used as a correctness reference against which HE / plaintext kernels
// are compared.
func MatmulPlain(A, B [][]float64) [][]float64 {
	rows, inner, cols := len(A), len(A[0]), len(B[0])
	C := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		C[i] = make([]float64, cols)
		for k := 0; k < inner; k++ {
			a := A[i][k]
			for j := 0; j < cols; j++ {
				C[i][j] += a * B[k][j]
			}
		}
	}
	return C
}

// MaxAbsError returns the maximum |A - B| over every entry of two
// (H × rows × cols) tensors. Used for PASS / FAIL decisions.
func MaxAbsError(A, B [][][]float64) float64 {
	var maxErr float64
	for z := range A {
		for i := range A[z] {
			for j := range A[z][i] {
				if e := math.Abs(A[z][i][j] - B[z][i][j]); e > maxErr {
					maxErr = e
				}
			}
		}
	}
	return maxErr
}

// PadToSlots returns a new slice of length s, with v copied into its prefix
// and the remainder zero-padded. Required because the encoder wants exactly
// s values.
func PadToSlots(v []float64, s int) []float64 {
	out := make([]float64, s)
	copy(out, v)
	return out
}

// AnyNonZero reports whether any entry of v is non-zero. Used to skip the
// multiplication by identically-zero masks in THOR.
func AnyNonZero(v []float64) bool {
	for _, x := range v {
		if x != 0 {
			return true
		}
	}
	return false
}

// EncodePlaintext encodes vec at the given level and scale. Centralised here
// so every runner handles level / scale alignment the same way.
func EncodePlaintext(
	ecd *ckks.Encoder,
	params ckks.Parameters,
	vec []float64,
	level int,
	scale rlwe.Scale,
) *rlwe.Plaintext {
	pt := ckks.NewPlaintext(params, level)
	pt.Scale = scale
	if err := ecd.Encode(vec, pt); err != nil {
		panic(err)
	}
	return pt
}

// ===========================================================================
// Operation counter (shared by every plaintext kernel).
//
// OpCounts tallies the three operation classes that dominate HE cost:
//
//   Rotations : slot rotations            (one Galois key-switch each)
//   CtCtMuls  : ciphertext × ciphertext   (one relinearisation + rescale)
//   CtPtMuls  : ciphertext × plaintext    (cheaper: no relin, just rescale)
//
// Because every plaintext kernel in this project mirrors its HE counterpart
// line-for-line, these counts are exactly what the HE kernel would perform
// at the same configuration — so plaintext runs double as an analytical
// tool for predicting HE cost without running the HE kernel.
//
// Fields the kernel does not exercise (e.g. CtPtMuls for MOAI) simply stay
// at zero, keeping the report format consistent across algorithms.
// ===========================================================================

type OpCounts struct {
	Rotations int
	CtCtMuls  int
	CtPtMuls  int
}

func (c OpCounts) String() string {
	return fmt.Sprintf("rot=%d  ct·ct=%d  ct·pt=%d  (total=%d)",
		c.Rotations, c.CtCtMuls, c.CtPtMuls,
		c.Rotations+c.CtCtMuls+c.CtPtMuls)
}

// ===========================================================================
// Slot-level primitives shared by every plaintext kernel.
// Semantics match Lattigo: rotateSlots(v, k)[i] = v[(i + k) mod len(v)]
// (left shift by k).
// ===========================================================================

func mulSlots(a, b []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] * b[i]
	}
	return out
}

func addSlots(a, b []float64) []float64 {
	if a == nil {
		out := make([]float64, len(b))
		copy(out, b)
		return out
	}
	for i := range a {
		a[i] += b[i]
	}
	return a
}

func rotateSlots(v []float64, k int) []float64 {
	n := len(v)
	k = ((k % n) + n) % n
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = v[(i+k)%n]
	}
	return out
}
