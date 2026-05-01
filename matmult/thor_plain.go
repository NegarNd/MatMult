// thor_plain.go
//
// Plaintext packing, replication, and routing-mask construction for THOR,
// ported from plaintext/thor_plain.py.
//
// These functions fully determine the slot layout consumed by the HE kernel
// in thor_cipher.go. The layout of every packed vector of length s = c·n·H
// is
//                idx  =  r · (n·H)  +  h  +  H · t
// with
//   r ∈ [0, c)  — block index (which diagonal group)
//   t ∈ [0, n) — position within an interleaved diagonal
//   h ∈ [0, H) — head index  (heads are interleaved, not stacked)

package main

// lowerDiagonal extracts the lower diagonal at offset ell from an (m, n)
// matrix:   lowerDiagonal(A, ell)[t] = A[(t + ell) mod m][t], t ∈ [0, n).
//
// Port of plaintext/thor_plain.py → _lower_diagonal.
func lowerDiagonal(A [][]float64, ell int) []float64 {
	m := len(A)
	n := len(A[0])
	out := make([]float64, n)
	for t := 0; t < n; t++ {
		out[t] = A[(t+ell)%m][t]
	}
	return out
}

// EncodeBatched packs H matrices of shape (m, n) into m_c = m/c flat vectors
// of length s = c · n · H.
//
// Each output vector contains c interleaved diagonals, concatenated. The
// j-th output vector encodes diagonals {c·j, c·j + 1, …, c·j + c − 1}.
//
// Port of plaintext/thor_plain.py → encode_batched.
func EncodeBatched(Ms [][][]float64, c, H int) [][]float64 {
	m := len(Ms[0])
	n := len(Ms[0][0])
	nH := n * H
	s := c * nH
	mc := m / c

	out := make([][]float64, mc)
	for j := 0; j < mc; j++ {
		vec := make([]float64, s)
		for r := 0; r < c; r++ {
			ell := c*j + r
			base := r * nH
			// Interleaved diagonal:  vec[base + z + H·t] = Ms[z][(t+ell) mod m][t]
			for z := 0; z < H; z++ {
				for t := 0; t < n; t++ {
					vec[base+z+H*t] = Ms[z][(t+ell)%m][t]
				}
			}
		}
		out[j] = vec
	}
	return out
}

// DecodeBatched inverts EncodeBatched, recovering H matrices of shape (m, n)
// from the m_c decrypted vectors.
//
// Port of plaintext/thor_plain.py → decode_batched.
func DecodeBatched(vecs [][]float64, m, n, c, H int) [][][]float64 {
	nH := n * H

	out := make([][][]float64, H)
	for z := 0; z < H; z++ {
		out[z] = make([][]float64, m)
		for i := 0; i < m; i++ {
			out[z][i] = make([]float64, n)
		}
	}

	for j, vec := range vecs {
		for r := 0; r < c; r++ {
			ell := c*j + r
			base := r * nH
			for z := 0; z < H; z++ {
				for t := 0; t < n; t++ {
					out[z][(t+ell)%m][t] = vec[base+z+H*t]
				}
			}
		}
	}
	return out
}

// Replication expands the n_c = n/c packed B vectors into the n ciphertext-
// ready vectors consumed by the kernel. Each output vector holds c copies of
// one interleaved diagonal (length nH, tiled c times to length s).
//
// Port of plaintext/thor_plain.py → replication.
func Replication(bPacked [][]float64, n, c, H int) [][]float64 {
	nH := n * H
	s := c * nH
	nc := n / c

	out := make([][]float64, 0, n)
	for j := 0; j < nc; j++ {
		ct := bPacked[j]
		for k := 0; k < c; k++ {
			diag := ct[k*nH : (k+1)*nH] // length nH
			rep := make([]float64, s)
			for i := 0; i < c; i++ {
				copy(rep[i*nH:(i+1)*nH], diag)
			}
			out = append(out, rep)
		}
	}
	return out
}

// BuildEllMasks builds the three Algorithm-2 routing masks for a given ell.
//
// For each slot (r, t, h), Algorithm 2 computes
//
//	r'        = (r − [ell]_c + ⌊(t + ell)/n⌋)   mod c
//	r_out     = (r' + ell)                      mod c
//	sameBlock = r' < c − [ell]_c
//	needsRot  = r_out ≠ r
//
// and routes the slot's contribution to one of four buckets:
//
//	(!sameBlock, !needsRot) → μ_{ℓ,0} (next block, no-rotation accumulator)
//	( sameBlock, !needsRot) → μ_{ℓ,1} (same block, no-rotation accumulator)
//	(!sameBlock,  needsRot) → μ_{ℓ,2} (next block, rotation accumulator)
//	( sameBlock,  needsRot) → μ_{ℓ,3} (same block, rotation accumulator)
//
// μ_{ℓ,3} is not returned: the kernel computes it "for free" as
// v − v·μ_{ℓ,0} − v·μ_{ℓ,1} − v·μ_{ℓ,2}, saving one plaintext multiplication.
//
// Port of plaintext/thor_plain.py → _build_ell_masks.
func BuildEllMasks(ell, c, n, H, s int) (mu0, mu1, mu2 []float64) {
	nH := n * H
	ellC := posMod(ell, c)

	mu0 = make([]float64, s)
	mu1 = make([]float64, s)
	mu2 = make([]float64, s)

	for r := 0; r < c; r++ {
		for t := 0; t < n; t++ {
			rPrime := posMod(r-ellC+(t+ell)/n, c)
			sameBlock := rPrime < (c - ellC)
			rOut := posMod(rPrime+ell, c)
			needsRot := rOut != r

			for h := 0; h < H; h++ {
				idx := r*nH + h + H*t
				switch {
				case !sameBlock && !needsRot:
					mu0[idx] = 1.0
				case sameBlock && !needsRot:
					mu1[idx] = 1.0
				case !sameBlock && needsRot:
					mu2[idx] = 1.0
					// (sameBlock && needsRot) → μ_{ℓ,3}, obtained for free.
				}
			}
		}
	}
	return mu0, mu1, mu2
}

// posMod returns a mod m in the range [0, m). Go's built-in % can be
// negative when a is negative; Python's % cannot. We match Python here.
func posMod(a, m int) int {
	r := a % m
	if r < 0 {
		r += m
	}
	return r
}
