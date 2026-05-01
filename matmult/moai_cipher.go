// moai_cipher.go
//
// HE implementation of MOAI Col×Col→Diag (Algorithm 3) and Diag×Col→Col
// (Algorithm 4), built on Lattigo v6. Port of ciphertext/moai_cipher.py.
//
// Scale management (same pattern as THOR, see thor_cipher.go):
//   - First  operand (Q  for col×col, C  for diag×col) is encrypted at the
//     default scale S.
//   - Second operand (K  for col×col, V  for diag×col) is encrypted at
//     scale Q[L], so that the ct × ct product after a single Rescale lands
//     exactly at scale S at level L − 1. All additions in the kernel are
//     therefore at matching scale with no runtime reconciliation.
//
// Hoisted BSGS (moai_col_col_bsgs_he_hoisted in the Python driver) is NOT
// implemented in this port. Lattigo v6 exposes hoisting only through the
// low-level RotateHoistedLazyNew API (which returns unrescaled QP
// polynomials and requires a manual ModDown). Non-hoisted BSGS already
// achieves the paper's asymptotic rotation count; a future patch can add
// hoisting as an optimisation without changing this file's interface.

package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ---------------------------------------------------------------------------
// Rotation set helpers (pass each to ctx.WithRotations before running)
// ---------------------------------------------------------------------------

// RequiredMoaiColColNaiveRotations returns the slot shifts used by
// MoaiColColNaiveHE. All shifts are reduced modulo nHE and zero shifts
// (identity) are filtered out.
func RequiredMoaiColColNaiveRotations(m, rotStride, nHE int) []int {
	set := map[int]struct{}{}
	for j := 1; j < m; j++ {
		if sh := posModInt(j*rotStride, nHE); sh != 0 {
			set[sh] = struct{}{}
		}
	}
	return keysOf(set)
}

// RequiredMoaiColColBSGSRotations returns the slot shifts used by
// MoaiColColBSGSHE: baby-step rotations on K, giant-step rotations on Q,
// and the final rotation per giant step.
func RequiredMoaiColColBSGSRotations(m, rotStride, nHE int) []int {
	b := intCeilSqrt(m)
	g := (m + b - 1) / b

	set := map[int]struct{}{}
	for r := 1; r < b; r++ {
		if sh := posModInt(r*rotStride, nHE); sh != 0 {
			set[sh] = struct{}{}
		}
	}
	for alpha := 0; alpha < g; alpha++ {
		shift := (alpha * b) % m
		if sh := posModInt((m-shift)*rotStride, nHE); sh != 0 {
			set[sh] = struct{}{}
		}
		if sh := posModInt(shift*rotStride, nHE); sh != 0 {
			set[sh] = struct{}{}
		}
	}
	return keysOf(set)
}

// RequiredMoaiDiagColBSGSRotations returns the slot shifts used by
// MoaiDiagColBSGSHE. Structurally identical to the col×col BSGS set.
func RequiredMoaiDiagColBSGSRotations(m, rotStride, nHE int) []int {
	// Same rotation set as col×col BSGS.
	return RequiredMoaiColColBSGSRotations(m, rotStride, nHE)
}

func keysOf(m map[int]struct{}) []int {
	out := make([]int, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// ---------------------------------------------------------------------------
// Algorithm 3 — Col × Col → Diag (HE)
// ---------------------------------------------------------------------------

// MoaiColColNaiveHE is the HE port of moai_col_col_naive:
//
//	diag_j = Σ_i Q[i] · Rot(K[i], j · stride)
//
// ctQ, ctK : d' ciphertexts each (one per column of Q or K).
// Output   : m diag-packed ciphertexts at level inputLevel - 1, scale S.
func MoaiColColNaiveHE(eval *ckks.Evaluator, ctQ, ctK []*rlwe.Ciphertext, m, dPrime, rotStride, nHE int) []*rlwe.Ciphertext {

	out := make([]*rlwe.Ciphertext, m)
	for j := 0; j < m; j++ {
		var acc *rlwe.Ciphertext

		for i := 0; i < dPrime; i++ {
			shift := posModInt(j*rotStride, nHE)
			var rotK *rlwe.Ciphertext
			if shift == 0 {
				rotK = ctK[i] // identity
			} else {
				var err error
				rotK, err = eval.RotateNew(ctK[i], shift)
				if err != nil {
					panic(fmt.Errorf("rot K (j=%d, i=%d): %w", j, i, err))
				}
			}
			prod, err := eval.MulRelinNew(ctQ[i], rotK)
			if err != nil {
				panic(fmt.Errorf("mul Q·rotK (j=%d, i=%d): %w", j, i, err))
			}
			if err := eval.Rescale(prod, prod); err != nil {
				panic(fmt.Errorf("rescale (j=%d, i=%d): %w", j, i, err))
			}

			if acc == nil {
				acc = prod
			} else {
				if err := eval.Add(acc, prod, acc); err != nil {
					panic(err)
				}
			}
		}
		out[j] = acc
	}
	return out
}

// MoaiColColBSGSHE is the HE port of moai_col_col_bsgs:
// j = α · b + r, with baby-step rotations on K done once per i and giant-
// step rotations on Q done per α.
func MoaiColColBSGSHE(eval *ckks.Evaluator, ctQ, ctK []*rlwe.Ciphertext, m, dPrime, rotStride, nHE int) []*rlwe.Ciphertext {
	b := intCeilSqrt(m)
	g := (m + b - 1) / b

	// -------------------------------------------------------------------------
	// Precompute baby-step shifts
	// -------------------------------------------------------------------------
	babyShifts := make([]int, 0, b-1)
	for r := 1; r < b; r++ {
		babyShifts = append(babyShifts, r*rotStride)
	}

	// -------------------------------------------------------------------------
	// Baby-step cache:
	// beta[i][r] = Rot(K[i], r*rotStride), with r=0 as identity.
	// We hoist each ctK[i] once over all baby shifts.
	// -------------------------------------------------------------------------
	beta := make([][]*rlwe.Ciphertext, dPrime)
	for i := 0; i < dPrime; i++ {
		beta[i] = make([]*rlwe.Ciphertext, b)
		beta[i][0] = ctK[i]

		// Hoisting
		rotated, err := eval.RotateHoistedNew(ctK[i], babyShifts)
		if err != nil {
			panic(err)
		}
		for r := 1; r < b; r++ {
			beta[i][r] = rotated[r*rotStride]

			// Non-hoisting part
			// rot, err := eval.RotateNew(ctK[i], r*rotStride)
			// if err != nil {
			// 	panic(fmt.Errorf("baby-step rot (i=%d, r=%d): %w", i, r, err))
			// }
			// beta[i][r] = rot
		}
	}

	// -------------------------------------------------------------------------
	// Precompute giant-step shifts
	// -------------------------------------------------------------------------
	giantShiftByAlpha := make([]int, g) // indexed by alpha; 0 means identity
	distinctSet := map[int]struct{}{}
	for alpha := 0; alpha < g; alpha++ {
		shift := (alpha * b) % m
		s := posModInt((m-shift)*rotStride, nHE)
		giantShiftByAlpha[alpha] = s
		if s != 0 {
			distinctSet[s] = struct{}{}
		}
	}
	giantShifts := make([]int, 0, len(distinctSet))
	for s := range distinctSet {
		giantShifts = append(giantShifts, s)
	}

	// -------------------------------------------------------------------------
	// Hoist every Q[i] once over all distinct giant-step shifts.
	// qRotAll[i][shift] gives Rot(Q[i], shift).
	// -------------------------------------------------------------------------
	qRotAll := make([]map[int]*rlwe.Ciphertext, dPrime)
	if len(giantShifts) > 0 {
		for i := 0; i < dPrime; i++ {
			rotated, err := eval.RotateHoistedNew(ctQ[i], giantShifts)
			if err != nil {
				panic(fmt.Errorf("hoisted giant-step rot (i=%d): %w", i, err))
			}
			qRotAll[i] = rotated
		}
	}

	// ---------------------------------------------------------------------------
	out := make([]*rlwe.Ciphertext, m)
	qRot := make([]*rlwe.Ciphertext, dPrime)

	for alpha := 0; alpha < g; alpha++ {
		qRotShift := giantShiftByAlpha[alpha]
		shift := (alpha * b) % m

		if qRotShift == 0 {
			copy(qRot, ctQ) // identity: alias the originals
		} else {
			for i := 0; i < dPrime; i++ {
				qRot[i] = qRotAll[i][qRotShift]
			}
		}

		for r := 0; r < b; r++ {
			j := alpha*b + r
			if j >= m {
				break
			}
			// partial, err := eval.MulRelinNew(qRot[0], beta[0][r]) // With key-switching
			partial, err := eval.MulNew(qRot[0], beta[0][r])
			if err != nil {
				panic(err)
			}
			for i := 1; i < dPrime; i++ {
				eval.MulThenAdd(qRot[i], beta[i][r], partial)
			}
			eval.Relinearize(partial, partial)
			eval.Rescale(partial, partial)

			// 	prod, err := eval.MulRelinNew(qRot[i], beta[i][r])
			// 	if err != nil {
			// 		panic(err)
			// 	}
			// 	if err := eval.Rescale(prod, prod); err != nil {
			// 		panic(err)
			// 	}
			// 	if err := eval.Add(partial, prod, partial); err != nil {
			// 		panic(err)
			// 	}
			// }

			finalShift := posModInt(shift*rotStride, nHE)
			var rolled *rlwe.Ciphertext
			if finalShift == 0 {
				out[j] = partial
			} else {
				rolled, err = eval.RotateNew(partial, finalShift)
				if err != nil {
					panic(fmt.Errorf("final rot (alpha=%d, r=%d): %w", alpha, r, err))
				}
				out[j] = rolled
			}
			// if out[j] == nil {
			// 	out[j] = rolled
			// } else {
			// 	if err := eval.Add(out[j], rolled, out[j]); err != nil {
			// 		panic(err)
			// 	}
			// }
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Algorithm 4 — Diag × Col → Col (HE, BSGS)
// ---------------------------------------------------------------------------

// MoaiDiagColBSGSHE is the HE port of moai_dig_col_bsgs.
// ctC : m ciphertexts holding the diagonals of C (packed by InterleavedDiagPack).
// ctV : d' ciphertexts holding the columns of V (packed by InterleavedColumnPack).
// Output : d' column-packed ciphertexts at level inputLevel - 1, scale S.
func MoaiDiagColBSGSHE(
	eval *ckks.Evaluator,
	ctC, ctV []*rlwe.Ciphertext,
	m, dPrime, rotStride, nHE int,
) []*rlwe.Ciphertext {
	b := intCeilSqrt(m)
	g := (m + b - 1) / b

	out := make([]*rlwe.Ciphertext, dPrime)
	for j := 0; j < dPrime; j++ {
		// Baby steps on V[j] : beta[r] = Rot(V[j], r · stride). r=0 = identity.
		beta := make([]*rlwe.Ciphertext, b)
		beta[0] = ctV[j]
		for r := 1; r < b; r++ {
			rot, err := eval.RotateNew(ctV[j], r*rotStride)
			if err != nil {
				panic(err)
			}
			beta[r] = rot
		}

		var acc *rlwe.Ciphertext

		for alpha := 0; alpha < g; alpha++ {
			shift := (alpha * b) % m
			cRotShift := posModInt((m-shift)*rotStride, nHE)

			var inner *rlwe.Ciphertext
			for r := 0; r < b; r++ {
				idx := alpha*b + r
				if idx >= m {
					break
				}
				var rotC *rlwe.Ciphertext
				if cRotShift == 0 {
					rotC = ctC[idx]
				} else {
					var err error
					rotC, err = eval.RotateNew(ctC[idx], cRotShift)
					if err != nil {
						panic(err)
					}
				}
				prod, err := eval.MulRelinNew(rotC, beta[r])
				if err != nil {
					panic(err)
				}
				if err := eval.Rescale(prod, prod); err != nil {
					panic(err)
				}
				if inner == nil {
					inner = prod
				} else {
					if err := eval.Add(inner, prod, inner); err != nil {
						panic(err)
					}
				}
			}

			finalShift := posModInt(shift*rotStride, nHE)
			var rolled *rlwe.Ciphertext
			if finalShift == 0 {
				rolled = inner
			} else {
				var err error
				rolled, err = eval.RotateNew(inner, finalShift)
				if err != nil {
					panic(err)
				}
			}
			if acc == nil {
				acc = rolled
			} else {
				if err := eval.Add(acc, rolled, acc); err != nil {
					panic(err)
				}
			}
		}
		out[j] = acc
	}
	return out
}
