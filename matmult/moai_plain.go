// moai_plain.go
//
// Packing, unpacking, and plaintext kernels for MOAI Col×Col → Diag
// (Algorithm 3) and Diag×Col → Col (Algorithm 4).
//
// Port of plaintext/moai_plain.py. The plaintext kernels mirror their HE
// counterparts in moai_cipher.go line-for-line, so the OpCounts returned
// here are exactly the ops an HE run would perform at the same config.
//
// Slot layout (interleaved over n_batch matrices packed per ciphertext):
//
//     slot[r · n_batch + s]  =  sequence s, position r
//
// A batch of n_batch = n_he / m matrices can be processed in lock-step
// inside a single ciphertext.

package main

import "math"

// ---------------------------------------------------------------------------
// Interleaved packing helpers
// ---------------------------------------------------------------------------

// InterleavedColumnPack packs n_batch (m × d) matrices into d slot-vectors
// of length n_he, one per column.
//
// Port of plaintext/moai_plain.py → interleaved_column_pack.
func InterleavedColumnPack(Xs [][][]float64, nHE int) [][]float64 {
	nBatch := len(Xs)
	m := len(Xs[0])
	d := len(Xs[0][0])
	out := make([][]float64, d)
	for j := 0; j < d; j++ {
		vec := make([]float64, nHE)
		for r := 0; r < m; r++ {
			for s := 0; s < nBatch; s++ {
				vec[r*nBatch+s] = Xs[s][r][j]
			}
		}
		out[j] = vec
	}
	return out
}

// InterleavedColumnUnpack inverts InterleavedColumnPack, recovering
// n_batch matrices of shape (m × d) from the d slot-vectors.
//
// Port of plaintext/moai_plain.py → interleaved_column_unpack.
func InterleavedColumnUnpack(cts [][]float64, m, nBatch int) [][][]float64 {
	d := len(cts)
	out := make([][][]float64, nBatch)
	for s := 0; s < nBatch; s++ {
		out[s] = make([][]float64, m)
		for r := 0; r < m; r++ {
			out[s][r] = make([]float64, d)
		}
	}
	for j, ct := range cts {
		for r := 0; r < m; r++ {
			for s := 0; s < nBatch; s++ {
				out[s][r][j] = ct[r*nBatch+s]
			}
		}
	}
	return out
}

// InterleavedDiagPack packs n_batch (m × m) matrices into m diagonal
// slot-vectors. The i-th output vector holds the i-th lower-diagonal of
// every matrix, interleaved in the n_batch dimension.
//
// Port of plaintext/moai_plain.py → interleaved_diag_pack.
func InterleavedDiagPack(Cs [][][]float64, nHE int) [][]float64 {
	nBatch := len(Cs)
	m := len(Cs[0])
	out := make([][]float64, m)
	for i := 0; i < m; i++ {
		vec := make([]float64, nHE)
		for r := 0; r < m; r++ {
			for s := 0; s < nBatch; s++ {
				vec[r*nBatch+s] = Cs[s][r][(r+i)%m]
			}
		}
		out[i] = vec
	}
	return out
}

// InterleavedDiagUnpack inverts InterleavedDiagPack.
//
// Port of plaintext/moai_plain.py → interleaved_diag_unpack.
func InterleavedDiagUnpack(cts [][]float64, m, nBatch int) [][][]float64 {
	out := make([][][]float64, nBatch)
	for s := 0; s < nBatch; s++ {
		out[s] = make([][]float64, m)
		for r := 0; r < m; r++ {
			out[s][r] = make([]float64, m)
		}
	}
	for i, ct := range cts {
		for r := 0; r < m; r++ {
			for s := 0; s < nBatch; s++ {
				out[s][r][(r+i)%m] = ct[r*nBatch+s]
			}
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Algorithm 3 — Col × Col → Diag (plaintext)
// ---------------------------------------------------------------------------

// moaiColColNaive implements the naive version of Algorithm 3:
//
//	diag_j  =  Σ_i  Q[i] ⊙ Rot_{j · stride}( K[i] )
//
// where stride = n_batch keeps the interleaved packing intact. This costs
// O((m − 1) · d') rotations and m · d' multiplications.
func moaiColColNaive(Q, K [][]float64, m, dPrime, rotStride int) ([][]float64, OpCounts) {
	nHE := len(Q[0])
	var counts OpCounts

	out := make([][]float64, m)
	for j := 0; j < m; j++ {
		acc := make([]float64, nHE)
		for i := 0; i < dPrime; i++ {
			var rotK []float64
			if j == 0 {
				rotK = K[i] // Rot(·, 0) is identity — no key-switch.
			} else {
				rotK = rotateSlots(K[i], j*rotStride)
				counts.Rotations++
			}
			prod := mulSlots(Q[i], rotK)
			counts.CtCtMuls++
			acc = addSlots(acc, prod)
		}
		out[j] = acc
	}
	return out, counts
}

// moaiColColBSGS implements the baby-step / giant-step decomposition of
// Algorithm 3: j = α · b + r, with baby-step rotations on K precomputed
// once per i, and giant-step rotations on Q done per α. Rotation count
// drops from (m − 1) · d' to (b + g − 2) · d' + (g − 1).
func moaiColColBSGS(Q, K [][]float64, m, dPrime, rotStride int) ([][]float64, OpCounts) {
	b := intCeilSqrt(m)
	g := (m + b - 1) / b
	nHE := len(Q[0])
	var counts OpCounts

	// Baby steps: β[i][r] = Rot_{r · stride}(K[i]).   r=0 is identity.
	beta := make([][][]float64, dPrime)
	for i := 0; i < dPrime; i++ {
		beta[i] = make([][]float64, b)
		beta[i][0] = K[i]
		for r := 1; r < b; r++ {
			beta[i][r] = rotateSlots(K[i], r*rotStride)
			counts.Rotations++
		}
	}

	out := make([][]float64, m)
	for j := range out {
		out[j] = make([]float64, nHE)
	}

	for alpha := 0; alpha < g; alpha++ {
		shift := (alpha * b) % m
		// Giant-step rotation on Q by (m − shift) · stride. When alpha=0
		// this reduces to 0 mod nHE (since m·stride = nHE in typical
		// configs) and collapses to the identity, so we skip it.
		qRotShift := posModInt((m-shift)*rotStride, nHE)
		qRot := make([][]float64, dPrime)
		for i := 0; i < dPrime; i++ {
			if qRotShift == 0 {
				qRot[i] = Q[i]
			} else {
				qRot[i] = rotateSlots(Q[i], qRotShift)
				counts.Rotations++
			}
		}

		for r := 0; r < b; r++ {
			j := alpha*b + r
			if j >= m {
				break
			}
			partial := mulSlots(qRot[0], beta[0][r])
			counts.CtCtMuls++
			for i := 1; i < dPrime; i++ {
				prod := mulSlots(qRot[i], beta[i][r])
				counts.CtCtMuls++
				partial = addSlots(partial, prod)
			}

			finalShift := posModInt(shift*rotStride, nHE)
			var rolled []float64
			if finalShift == 0 {
				rolled = partial
			} else {
				rolled = rotateSlots(partial, finalShift)
				counts.Rotations++
			}
			out[j] = addSlots(out[j], rolled)
		}
	}
	return out, counts
}

// ---------------------------------------------------------------------------
// Algorithm 4 — Diag × Col → Col (plaintext, BSGS)
// ---------------------------------------------------------------------------

// moaiDiagColBSGS implements Algorithm 4 with BSGS:
//
//	(C · V)_j  =  Σ_i  diag_i(C) ⊙ Rot_{i · stride}( V[j] )
//
// decomposed as i = α · b + r, so the outer loop is over j ∈ [0, d') and
// the two inner loops are the baby-step / giant-step decomposition.
func moaiDiagColBSGS(C, V [][]float64, m, dPrime, rotStride int) ([][]float64, OpCounts) {
	b := intCeilSqrt(m)
	g := (m + b - 1) / b
	nHE := len(V[0])
	var counts OpCounts

	out := make([][]float64, dPrime)
	for j := 0; j < dPrime; j++ {
		// Baby steps on V[j]: β[r] = Rot_{r · stride}(V[j]).
		beta := make([][]float64, b)
		beta[0] = V[j]
		for r := 1; r < b; r++ {
			beta[r] = rotateSlots(V[j], r*rotStride)
			counts.Rotations++
		}

		acc := make([]float64, nHE)
		for alpha := 0; alpha < g; alpha++ {
			shift := (alpha * b) % m
			cRotShift := posModInt((m-shift)*rotStride, nHE)

			inner := make([]float64, nHE)
			for r := 0; r < b; r++ {
				idx := alpha*b + r
				if idx >= m {
					break
				}
				var rotC []float64
				if cRotShift == 0 {
					rotC = C[idx]
				} else {
					rotC = rotateSlots(C[idx], cRotShift)
					counts.Rotations++
				}
				prod := mulSlots(rotC, beta[r])
				counts.CtCtMuls++
				inner = addSlots(inner, prod)
			}

			finalShift := posModInt(shift*rotStride, nHE)
			var rolled []float64
			if finalShift == 0 {
				rolled = inner
			} else {
				rolled = rotateSlots(inner, finalShift)
				counts.Rotations++
			}
			acc = addSlots(acc, rolled)
		}
		out[j] = acc
	}
	return out, counts
}

// ---------------------------------------------------------------------------
// Theoretical cost formulas (paper-side, ignoring identity rotations)
// ---------------------------------------------------------------------------

// TheoreticalNaive returns (multiplications, rotations, total) for the
// naive Col × Col variant of Algorithm 3.
func TheoreticalNaive(m, d int) (mul, rot, ks int) {
	mul = m * d
	rot = (m - 1) * d
	return mul, rot, mul + rot
}

// TheoreticalBSGS returns (multiplications, rotations, total) for the
// BSGS variant of Algorithm 3.
func TheoreticalBSGS(m, d int) (mul, rot, ks int) {
	b := intCeilSqrt(m)
	g := (m + b - 1) / b
	mul = m * d
	rot = (b-1)*d + (g-1)*d + (g - 1)
	return mul, rot, mul + rot
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

func intCeilSqrt(n int) int {
	return int(math.Ceil(math.Sqrt(float64(n))))
}

func posModInt(a, m int) int {
	r := a % m
	if r < 0 {
		r += m
	}
	return r
}
