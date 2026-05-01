// thor_cipher.go
//
// THOR lower-lower CC-MM (Algorithm 2 of Moon et al., CCS 2025), implemented
// over the CKKS scheme using the Lattigo v6 library.
//
// This file contains only the homomorphic kernel. All parameter selection,
// key generation, encoding, encryption and verification lives in main.go.

package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// EllMask holds the three pre-encoded mask plaintexts used for one value of
// ell in Algorithm 2. HasMu0 / HasMu2 indicate whether the corresponding mask
// is identically zero; when it is, the multiplication can be skipped and the
// μ₃ "free via subtraction" branch simply omits the corresponding term.
type EllMask struct {
	Mu0, Mu1, Mu2  *rlwe.Plaintext
	HasMu0, HasMu2 bool
}

// RequiredRotations returns the deduplicated set of slot shifts that
// ThorCCMatMulHE rotates by. Callers must generate Galois keys for every
// element of this slice before invoking the kernel.
func RequiredRotations(d, n, H, s int) []int {
	c := s / (n * H)
	nH := n * H

	set := map[int]struct{}{-nH: {}}
	for ell := 1; ell < n; ell++ {
		set[(-n*(ell%c)+ell)*H] = struct{}{}
	}
	out := make([]int, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	return out
}

// ThorCCMatMulHE evaluates C = A · B homomorphically, where A, B, and C are
// packed using the THOR lower-lower layout described in the paper.
//
// Inputs:
//   - ctAs     : m_c = d/c ciphertexts encrypting A, at level L, default scale.
//   - ctBRep   : n ciphertexts encrypting the replicated B, at level L,
//     scale Q[L] (see main.go for the scale trick).
//   - masks    : n−1 pre-encoded masks at level L−1, scale Q[L−1].
//   - d, n, H, s : packing parameters.
//
// Output : m_c ciphertexts encrypting C, at level L−2, default scale.
//
// Multiplicative depth : 2 (one ct×ct, one ct×pt).
func ThorCCMatMulHE(
	eval *ckks.Evaluator,
	ctAs, ctBRep []*rlwe.Ciphertext,
	masks []EllMask,
	d, n, H, s int,
) []*rlwe.Ciphertext {

	c := s / (n * H)
	nH := n * H
	mc := d / c

	// --------------------------------------------------------------------
	// Lines 4–8 of Algorithm 2 : intermediate products.
	//
	//     ct.C_{j,ℓ} = MulRelin( Rot(ct.A_j ; shift_ℓ) , ct.B_ℓ )
	//
	// After MulRelin + Rescale, each ct.C_{j,ℓ} sits at level L−1 with the
	// default scale.
	// --------------------------------------------------------------------
	ctCjl := make([][]*rlwe.Ciphertext, mc)

	// Added for hoisting
	shifts := make([]int, 0, n-1)
	for ell := 1; ell < n; ell++ {
		shifts = append(shifts, (-n*(ell%c)+ell)*H)
	}

	for j := 0; j < mc; j++ {
		ctCjl[j] = make([]*rlwe.Ciphertext, n)

		// ℓ = 0 : no rotation
		prod, err := eval.MulRelinNew(ctAs[j], ctBRep[0])
		if err != nil {
			panic(fmt.Errorf("MulRelin (j=%d, ell=0): %w", j, err))
		}
		if err := eval.Rescale(prod, prod); err != nil {
			panic(fmt.Errorf("Rescale (j=%d, ell=0): %w", j, err))
		}
		ctCjl[j][0] = prod

		// ℓ ≥ 1 : rotate A_j by shift_ℓ, then multiply with B_ℓ
		// Hoisted rotations for this ctAs[j]
		rots, err := eval.RotateHoistedNew(ctAs[j], shifts) // map[shift]*Ciphertext
		if err != nil {
			panic(err)
		}

		for ell := 1; ell < n; ell++ {
			shift := (-n*(ell%c) + ell) * H

			// Without Hoisting

			// // rot, err := eval.RotateNew(ctAs[j], shift)
			// // if err != nil {
			// // 	panic(fmt.Errorf("Rotate (j=%d, ell=%d): %w", j, ell, err))
			// // }
			// prod, err := eval.MulRelinNew(rot, ctBRep[ell])

			prod, err := eval.MulRelinNew(rots[shift], ctBRep[ell])

			if err != nil {
				panic(fmt.Errorf("MulRelin (j=%d, ell=%d): %w", j, ell, err))
			}
			if err := eval.Rescale(prod, prod); err != nil {
				panic(fmt.Errorf("Rescale (j=%d, ell=%d): %w", j, ell, err))
			}
			ctCjl[j][ell] = prod
		}
	}

	// --------------------------------------------------------------------
	// Lines 9–17 : masking and accumulation.
	//
	// Each ct.C_{j,ℓ} is split across four buckets by the four masks
	// μ_{ℓ,0..3}. The paper's line 14 avoids one PMult by computing the
	// μ_{ℓ,3} contribution as v − v₀ − v₁ − v₂ (the "free subtraction").
	//
	// Accumulators are lazy-initialised so their level/scale match the first
	// contribution (level L−2, default scale).
	// --------------------------------------------------------------------
	accPrime := make([]*rlwe.Ciphertext, mc)  // ct.C'_j
	accDPrime := make([]*rlwe.Ciphertext, mc) // ct.C''_j

	for ell := 1; ell < n; ell++ {
		m := masks[ell-1]
		ellQ := ell / c

		for j := 0; j < mc; j++ {
			v := ctCjl[j][ell] // level L−1, scale S
			jSame := (j + ellQ) % mc
			jNext := (j + ellQ + 1) % mc

			// μ_{ℓ,1}  →  C'_{jSame}
			v1, err := eval.MulNew(v, m.Mu1)
			if err != nil {
				panic(err)
			}
			// if err := eval.Rescale(v1, v1); err != nil {
			// 	panic(err)
			// }
			// accumulate(eval, &accPrime[jSame], v1)

			// μ_{ℓ,0}  →  C'_{jNext}   (skip if the mask is identically 0)
			var v0 *rlwe.Ciphertext
			if m.HasMu0 {
				v0, err = eval.MulNew(v, m.Mu0)
				if err != nil {
					panic(err)
				}
				// if err := eval.Rescale(v0, v0); err != nil {
				// 	panic(err)
				// }
				// accumulate(eval, &accPrime[jNext], v0)
			}

			// ρ^{nH}( μ_{ℓ,2} )  →  C''_{jNext}
			var v2 *rlwe.Ciphertext
			if m.HasMu2 {
				v2, err = eval.MulNew(v, m.Mu2)
				if err != nil {
					panic(err)
				}
				// if err := eval.Rescale(v2, v2); err != nil {
				// 	panic(err)
				// }
				// accumulate(eval, &accDPrime[jNext], v2)
			}

			// now rescale only at the point of use, not immediately after every MulNew
			if err := eval.Rescale(v1, v1); err != nil {
				panic(err)
			}
			accumulate(eval, &accPrime[jSame], v1)

			if v0 != nil {
				if err := eval.Rescale(v0, v0); err != nil {
					panic(err)
				}
				accumulate(eval, &accPrime[jNext], v0)
			}

			if v2 != nil {
				if err := eval.Rescale(v2, v2); err != nil {
					panic(err)
				}
				accumulate(eval, &accDPrime[jNext], v2)
			}

			// μ_{ℓ,3} for free (paper line 14) :
			//     v₃ = v − v₀ − v₁ − v₂   →   C''_{jSame}
			//
			// v is at level L−1, whereas v₀/v₁/v₂ are at level L−2. Their
			// scales already agree (both are exactly the default scale, see
			// the scale trick in main.go), so DropLevel is enough to align.
			v3 := v.CopyNew()
			if diff := v3.Level() - v1.Level(); diff > 0 {
				eval.DropLevel(v3, diff)
			}
			if v0 != nil {
				if err := eval.Sub(v3, v0, v3); err != nil {
					panic(err)
				}
			}
			if err := eval.Sub(v3, v1, v3); err != nil {
				panic(err)
			}
			if v2 != nil {
				if err := eval.Sub(v3, v2, v3); err != nil {
					panic(err)
				}
			}
			accumulate(eval, &accDPrime[jSame], v3)
		}
	}

	// --------------------------------------------------------------------
	// Line 18 : final assembly
	//
	//     C_j = ct.C_{j,0} + C'_j + Rot( C''_j ; −nH )
	//
	// ct.C_{j,0} is at level L−1, the accumulators at level L−2. Drop the
	// base term by one level to match.
	// --------------------------------------------------------------------
	out := make([]*rlwe.Ciphertext, mc)
	for j := 0; j < mc; j++ {
		base := ctCjl[j][0].CopyNew()

		targetLevel := -1
		if accPrime[j] != nil {
			targetLevel = accPrime[j].Level()
		}
		if accDPrime[j] != nil && (targetLevel < 0 || accDPrime[j].Level() < targetLevel) {
			targetLevel = accDPrime[j].Level()
		}
		if targetLevel >= 0 {
			if diff := base.Level() - targetLevel; diff > 0 {
				eval.DropLevel(base, diff)
			}
		}

		if accPrime[j] != nil {
			if err := eval.Add(base, accPrime[j], base); err != nil {
				panic(err)
			}
		}
		if accDPrime[j] != nil {
			rolled, err := eval.RotateNew(accDPrime[j], -nH)
			if err != nil {
				panic(err)
			}
			if err := eval.Add(base, rolled, base); err != nil {
				panic(err)
			}
		}
		out[j] = base
	}
	return out
}

func ThorCCMatMulHELazyRelin(
	eval *ckks.Evaluator,
	ctAs, ctBRep []*rlwe.Ciphertext,
	masks []EllMask,
	d, n, H, s int,
) []*rlwe.Ciphertext {

	c := s / (n * H)
	nH := n * H
	mc := d / c

	// CHANGED: base terms are now extended degree-2 ciphertexts.
	baseTermsExt := make([]*rlwe.Ciphertext, mc)

	accPrimeExt := make([]*rlwe.Ciphertext, mc)
	accDPrimeExt := make([]*rlwe.Ciphertext, mc)

	shifts := make([]int, 0, n-1)
	for ell := 1; ell < n; ell++ {
		shifts = append(shifts, (-n*(ell%c)+ell)*H)
	}

	for j := 0; j < mc; j++ {

		// CHANGED:
		// Use MulNew instead of MulRelinNew.
		// This keeps C_{j,0} as degree-2.
		base, err := eval.MulNew(ctAs[j], ctBRep[0])
		if err != nil {
			panic(fmt.Errorf("Mul base lazy (j=%d): %w", j, err))
		}
		if err := eval.Rescale(base, base); err != nil {
			panic(fmt.Errorf("Rescale base lazy (j=%d): %w", j, err))
		}
		baseTermsExt[j] = base

		rots, err := eval.RotateHoistedNew(ctAs[j], shifts)
		if err != nil {
			panic(fmt.Errorf("RotateHoistedNew (j=%d): %w", j, err))
		}

		for ell := 1; ell < n; ell++ {
			shift := (-n*(ell%c) + ell) * H
			m := masks[ell-1]
			ellQ := ell / c

			jSame := (j + ellQ) % mc
			jNext := (j + ellQ + 1) % mc

			v, err := eval.MulNew(rots[shift], ctBRep[ell]) // degree-2
			if err != nil {
				panic(fmt.Errorf("MulNew ct-ct (j=%d, ell=%d): %w", j, ell, err))
			}
			if err := eval.Rescale(v, v); err != nil {
				panic(fmt.Errorf("Rescale ct-ct (j=%d, ell=%d): %w", j, ell, err))
			}

			v1, err := eval.MulNew(v, m.Mu1) // still degree-2
			if err != nil {
				panic(fmt.Errorf("MulNew mu1 (j=%d, ell=%d): %w", j, ell, err))
			}

			var v0 *rlwe.Ciphertext
			if m.HasMu0 {
				v0, err = eval.MulNew(v, m.Mu0)
				if err != nil {
					panic(fmt.Errorf("MulNew mu0 (j=%d, ell=%d): %w", j, ell, err))
				}
			}

			var v2 *rlwe.Ciphertext
			if m.HasMu2 {
				v2, err = eval.MulNew(v, m.Mu2)
				if err != nil {
					panic(fmt.Errorf("MulNew mu2 (j=%d, ell=%d): %w", j, ell, err))
				}
			}

			if err := eval.Rescale(v1, v1); err != nil {
				panic(err)
			}
			accumulate(eval, &accPrimeExt[jSame], v1)

			if v0 != nil {
				if err := eval.Rescale(v0, v0); err != nil {
					panic(err)
				}
				accumulate(eval, &accPrimeExt[jNext], v0)
			}

			if v2 != nil {
				if err := eval.Rescale(v2, v2); err != nil {
					panic(err)
				}
				accumulate(eval, &accDPrimeExt[jNext], v2)
			}

			v3 := v.CopyNew()

			if diff := v3.Level() - v1.Level(); diff > 0 {
				eval.DropLevel(v3, diff)
			}

			if v0 != nil {
				if err := eval.Sub(v3, v0, v3); err != nil {
					panic(fmt.Errorf("Sub v0 (j=%d, ell=%d): %w", j, ell, err))
				}
			}
			if err := eval.Sub(v3, v1, v3); err != nil {
				panic(fmt.Errorf("Sub v1 (j=%d, ell=%d): %w", j, ell, err))
			}
			if v2 != nil {
				if err := eval.Sub(v3, v2, v3); err != nil {
					panic(fmt.Errorf("Sub v2 (j=%d, ell=%d): %w", j, ell, err))
				}
			}

			accumulate(eval, &accDPrimeExt[jSame], v3)
		}
	}

	// CHANGED:
	// Instead of relinearizing base separately, merge:
	//   xExt = baseTermsExt[j] + accPrimeExt[j]
	// then relinearize xExt once.
	accPrime := make([]*rlwe.Ciphertext, mc)
	accDPrime := make([]*rlwe.Ciphertext, mc)

	for j := 0; j < mc; j++ {

		xExt := baseTermsExt[j].CopyNew()

		if accPrimeExt[j] != nil {
			targetLevel := accPrimeExt[j].Level()
			if diff := xExt.Level() - targetLevel; diff > 0 {
				eval.DropLevel(xExt, diff)
			}
			if diff := accPrimeExt[j].Level() - xExt.Level(); diff > 0 {
				eval.DropLevel(accPrimeExt[j], diff)
			}

			if err := eval.Add(xExt, accPrimeExt[j], xExt); err != nil {
				panic(fmt.Errorf("Add baseExt + accPrimeExt (j=%d): %w", j, err))
			}
		}

		tmp, err := eval.RelinearizeNew(xExt)
		if err != nil {
			panic(fmt.Errorf("Relinearize basePlusPrime (j=%d): %w", j, err))
		}
		accPrime[j] = tmp

		if accDPrimeExt[j] != nil {
			tmp, err := eval.RelinearizeNew(accDPrimeExt[j])
			if err != nil {
				panic(fmt.Errorf("Relinearize accDPrime (j=%d): %w", j, err))
			}
			accDPrime[j] = tmp
		}
	}

	out := make([]*rlwe.Ciphertext, mc)

	for j := 0; j < mc; j++ {
		base := accPrime[j]

		if accDPrime[j] != nil {
			rolled, err := eval.RotateNew(accDPrime[j], -nH)
			if err != nil {
				panic(fmt.Errorf("Rotate accDPrime (j=%d): %w", j, err))
			}

			if diff := base.Level() - rolled.Level(); diff > 0 {
				eval.DropLevel(base, diff)
			}
			if diff := rolled.Level() - base.Level(); diff > 0 {
				eval.DropLevel(rolled, diff)
			}

			if err := eval.Add(base, rolled, base); err != nil {
				panic(fmt.Errorf("Add rotated accDPrime (j=%d): %w", j, err))
			}
		}

		out[j] = base
	}

	return out
}

// THE MEMORY OPTIMIZED VERSION
// func ThorCCMatMulHE(
// 	eval *ckks.Evaluator,
// 	ctAs, ctBRep []*rlwe.Ciphertext,
// 	masks []EllMask,
// 	d, n, H, s int,
// ) []*rlwe.Ciphertext {

// 	c := s / (n * H)
// 	nH := n * H
// 	mc := d / c

// 	// Keep only the ℓ = 0 base term for final assembly.
// 	baseTerms := make([]*rlwe.Ciphertext, mc)

// 	// Accumulators:
// 	//   accPrime  = C'_j
// 	//   accDPrime = C''_j
// 	accPrime := make([]*rlwe.Ciphertext, mc)
// 	accDPrime := make([]*rlwe.Ciphertext, mc)

// 	// Hoisted rotation shifts for ℓ >= 1
// 	shifts := make([]int, 0, n-1)
// 	for ell := 1; ell < n; ell++ {
// 		shifts = append(shifts, (-n*(ell%c)+ell)*H)
// 	}

// 	// --------------------------------------------------------------------
// 	// Streamed computation:
// 	//
// 	// For each j:
// 	//   1) compute/store baseTerms[j] = C_{j,0}
// 	//   2) hoist rotations of A_j
// 	//   3) for each ell >= 1:
// 	//        v = MulRelin( Rot(A_j, shift_ell), B_ell )
// 	//        Rescale(v)
// 	//        immediately do masking + accumulation
// 	// --------------------------------------------------------------------
// 	for j := 0; j < mc; j++ {

// 		// ℓ = 0 : no rotation, keep only this base term.
// 		base, err := eval.MulRelinNew(ctAs[j], ctBRep[0])
// 		if err != nil {
// 			panic(fmt.Errorf("MulRelin (j=%d, ell=0): %w", j, err))
// 		}
// 		if err := eval.Rescale(base, base); err != nil {
// 			panic(fmt.Errorf("Rescale (j=%d, ell=0): %w", j, err))
// 		}
// 		baseTerms[j] = base

// 		// Hoisted rotations for this ctAs[j]
// 		rots, err := eval.RotateHoistedNew(ctAs[j], shifts)
// 		if err != nil {
// 			panic(fmt.Errorf("RotateHoistedNew (j=%d): %w", j, err))
// 		}

// 		for ell := 1; ell < n; ell++ {
// 			shift := (-n*(ell%c) + ell) * H
// 			m := masks[ell-1]
// 			ellQ := ell / c

// 			jSame := (j + ellQ) % mc
// 			jNext := (j + ellQ + 1) % mc

// 			// v = C_{j,ell}
// 			v, err := eval.MulRelinNew(rots[shift], ctBRep[ell])
// 			if err != nil {
// 				panic(fmt.Errorf("MulRelin (j=%d, ell=%d): %w", j, ell, err))
// 			}
// 			if err := eval.Rescale(v, v); err != nil {
// 				panic(fmt.Errorf("Rescale (j=%d, ell=%d): %w", j, ell, err))
// 			}

// 			// μ_{ell,1} -> C'_{jSame}
// 			v1, err := eval.MulNew(v, m.Mu1)
// 			if err != nil {
// 				panic(fmt.Errorf("MulNew mu1 (j=%d, ell=%d): %w", j, ell, err))
// 			}
// 			if err := eval.Rescale(v1, v1); err != nil {
// 				panic(fmt.Errorf("Rescale mu1 (j=%d, ell=%d): %w", j, ell, err))
// 			}
// 			accumulate(eval, &accPrime[jSame], v1)

// 			// μ_{ell,0} -> C'_{jNext}
// 			var v0 *rlwe.Ciphertext
// 			if m.HasMu0 {
// 				v0, err = eval.MulNew(v, m.Mu0)
// 				if err != nil {
// 					panic(fmt.Errorf("MulNew mu0 (j=%d, ell=%d): %w", j, ell, err))
// 				}
// 				if err := eval.Rescale(v0, v0); err != nil {
// 					panic(fmt.Errorf("Rescale mu0 (j=%d, ell=%d): %w", j, ell, err))
// 				}
// 				accumulate(eval, &accPrime[jNext], v0)
// 			}

// 			// μ_{ell,2} -> C''_{jNext}
// 			var v2 *rlwe.Ciphertext
// 			if m.HasMu2 {
// 				v2, err = eval.MulNew(v, m.Mu2)
// 				if err != nil {
// 					panic(fmt.Errorf("MulNew mu2 (j=%d, ell=%d): %w", j, ell, err))
// 				}
// 				if err := eval.Rescale(v2, v2); err != nil {
// 					panic(fmt.Errorf("Rescale mu2 (j=%d, ell=%d): %w", j, ell, err))
// 				}
// 				accumulate(eval, &accDPrime[jNext], v2)
// 			}

// 			// μ_{ell,3} for free:
// 			//   v3 = v - v0 - v1 - v2  -> C''_{jSame}
// 			v3 := v.CopyNew()
// 			if diff := v3.Level() - v1.Level(); diff > 0 {
// 				eval.DropLevel(v3, diff)
// 			}
// 			if v0 != nil {
// 				if err := eval.Sub(v3, v0, v3); err != nil {
// 					panic(fmt.Errorf("Sub v0 (j=%d, ell=%d): %w", j, ell, err))
// 				}
// 			}
// 			if err := eval.Sub(v3, v1, v3); err != nil {
// 				panic(fmt.Errorf("Sub v1 (j=%d, ell=%d): %w", j, ell, err))
// 			}
// 			if v2 != nil {
// 				if err := eval.Sub(v3, v2, v3); err != nil {
// 					panic(fmt.Errorf("Sub v2 (j=%d, ell=%d): %w", j, ell, err))
// 				}
// 			}
// 			accumulate(eval, &accDPrime[jSame], v3)
// 		}
// 	}

// 	// --------------------------------------------------------------------
// 	// Final assembly:
// 	//   C_j = C_{j,0} + C'_j + Rot(C''_j ; -nH)
// 	// --------------------------------------------------------------------
// 	out := make([]*rlwe.Ciphertext, mc)
// 	for j := 0; j < mc; j++ {
// 		base := baseTerms[j].CopyNew()

// 		targetLevel := -1
// 		if accPrime[j] != nil {
// 			targetLevel = accPrime[j].Level()
// 		}
// 		if accDPrime[j] != nil && (targetLevel < 0 || accDPrime[j].Level() < targetLevel) {
// 			targetLevel = accDPrime[j].Level()
// 		}
// 		if targetLevel >= 0 {
// 			if diff := base.Level() - targetLevel; diff > 0 {
// 				eval.DropLevel(base, diff)
// 			}
// 		}

// 		if accPrime[j] != nil {
// 			if err := eval.Add(base, accPrime[j], base); err != nil {
// 				panic(fmt.Errorf("Add accPrime (j=%d): %w", j, err))
// 			}
// 		}
// 		if accDPrime[j] != nil {
// 			rolled, err := eval.RotateNew(accDPrime[j], -nH)
// 			if err != nil {
// 				panic(fmt.Errorf("Rotate accDPrime (j=%d): %w", j, err))
// 			}
// 			if err := eval.Add(base, rolled, base); err != nil {
// 				panic(fmt.Errorf("Add rotated accDPrime (j=%d): %w", j, err))
// 			}
// 		}
// 		out[j] = base
// 	}

// 	return out
// }

// accumulate lazily initialises an accumulator on first use, then adds ct
// into it. The copy on first use ensures the accumulator is independent of
// the first contribution's ciphertext.
func accumulate(eval *ckks.Evaluator, acc **rlwe.Ciphertext, ct *rlwe.Ciphertext) {
	if *acc == nil {
		*acc = ct.CopyNew()
		return
	}
	if err := eval.Add(*acc, ct, *acc); err != nil {
		panic(fmt.Errorf("accumulator add: %w", err))
	}
}
