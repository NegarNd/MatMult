// bmm1_cipher.go
//
// HE implementation of BMM-I bicyclic matrix multiplication.
// Port of ciphertext/bmm1_cipher.py.
//
// Four hoisting strategies, matching the Python naming:
//
//   "none"          — no hoisting; eval.RotateNew per inner iteration.
//   "per_block"     — hoist A and B inside each single-block call.
//   "prehoisted_a"  — pre-hoist every A block once; hoist B per call.
//   "prehoisted_ab" — pre-hoist every A and every B block (optimal).
//
// Scale management (same trick as THOR and MOAI)
// ----------------------------------------------
//   - A is encrypted at the default scale S.
//   - B is encrypted at scale Q[L], so that ct_A · ct_B after a single
//     Rescale lands at scale S at level L − 1. All additions in the kernel
//     are therefore at matching scale with no runtime reconciliation.
//
// A note on cyclic rotation vs. [:n·p] slicing
// --------------------------------------------
// The Python single-block kernel does `Rot(vec, k)[:n*p]` because the
// rotation is applied to the tiled vector and only the first n·p slots
// hold valid content. In Lattigo, rotations are cyclic over the whole
// n_he slot range — but because `RepeatVector` in bmm1_plain.go already
// tiles the input enough to cover `rotation + n·p` valid slots before any
// zero padding, the product of two such tiled ciphertexts still has the
// correct values in the first n·p slots, which is exactly what
// `BicyclicDecode` reads. The remaining slots are "garbage" but ignored.

package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// Bmm1Hoisting enumerates the four hoisting strategies; use with RunBmm1HE
// or MatMulBmm1HE.
type Bmm1Hoisting string

const (
	HoistNone     Bmm1Hoisting = "none"
	HoistPerBlock Bmm1Hoisting = "per_block"
	HoistPreA     Bmm1Hoisting = "prehoisted_a"
	HoistPreAB    Bmm1Hoisting = "prehoisted_ab"
)

// ---------------------------------------------------------------------------
// Rotation set helper (pass to ctx.WithRotations)
// ---------------------------------------------------------------------------

// RequiredBmm1Rotations returns the distinct shifts the kernel performs.
// All four hoisting strategies use the same underlying shift set, so one
// call covers any strategy the user picks.
func RequiredBmm1Rotations(sN, sM, sP int) []int {
	rotsA, rotsB := blockRotLists(sN, sM, sP)
	set := map[int]struct{}{}
	for _, r := range rotsA {
		if r != 0 {
			set[r] = struct{}{}
		}
	}
	for _, r := range rotsB {
		if r != 0 {
			set[r] = struct{}{}
		}
	}
	out := make([]int, 0, len(set))
	for r := range set {
		out = append(out, r)
	}
	return out
}

// blockRotLists returns the two rotation schedules used per block:
//
//	rotsA[i] = (i · n·p) mod (n·m)
//	rotsB[i] = (i · n·p) mod (m·p)
//
// Port of ciphertext/bmm1_cipher.py → _block_rot_lists.
func blockRotLists(n, m, p int) (rotsA, rotsB []int) {
	step := n * p
	rotsA = make([]int, m)
	rotsB = make([]int, m)
	for i := 0; i < m; i++ {
		rotsA[i] = (i * step) % (n * m)
		rotsB[i] = (i * step) % (m * p)
	}
	return rotsA, rotsB
}

// distinctNonZero filters `shifts` into the set of unique non-zero shifts,
// suitable to pass to eval.RotateHoistedNew.
func distinctNonZero(shifts []int) []int {
	set := map[int]struct{}{}
	for _, s := range shifts {
		if s != 0 {
			set[s] = struct{}{}
		}
	}
	out := make([]int, 0, len(set))
	for s := range set {
		out = append(out, s)
	}
	return out
}

// ---------------------------------------------------------------------------
// Block encoding + encryption
// ---------------------------------------------------------------------------

// EncodeAndEncryptBlocks packs A and B, bicyclic-encodes each block,
// tiles it so rotations stay within valid content, then encrypts.
//
// aCt[i][k] / bCt[k][j] correspond to the Python A_ct / B_ct arrays.
// aScale is the default scale; bScale is Q[inputLevel] (the "scale trick"
// that makes ct_A · ct_B land at the default scale after Rescale).
func EncodeAndEncryptBlocks(
	ctx *HEContext,
	A, B [][]float64,
	sN, sM, sP, inputLevel int,
) (aCt, bCt [][]*rlwe.Ciphertext) {

	N := len(A)
	M := len(A[0])
	P := len(B[0])
	aRows := N / sN
	kBlocks := M / sM
	bCols := P / sP

	aTiles := intCeilDiv(sM+sP, sM)
	bTiles := intCeilDiv(sM+sN, sM)

	aScale := ctx.Params.DefaultScale()
	bScale := rlwe.NewScale(ctx.Params.Q()[inputLevel])

	// A: encrypt each block at default scale.
	aCt = make([][]*rlwe.Ciphertext, aRows)
	for i := 0; i < aRows; i++ {
		aCt[i] = make([]*rlwe.Ciphertext, kBlocks)
		for k := 0; k < kBlocks; k++ {
			block := subMatrix(A, i*sN, (i+1)*sN, k*sM, (k+1)*sM)
			vec := RepeatVector(BicyclicEncode(block), aTiles)
			aCt[i][k] = encryptVecAtScale(ctx, vec, inputLevel, aScale)
		}
	}
	// B: encrypt each block at scale Q[L].
	bCt = make([][]*rlwe.Ciphertext, kBlocks)
	for k := 0; k < kBlocks; k++ {
		bCt[k] = make([]*rlwe.Ciphertext, bCols)
		for j := 0; j < bCols; j++ {
			block := subMatrix(B, k*sM, (k+1)*sM, j*sP, (j+1)*sP)
			vec := RepeatVector(BicyclicEncode(block), bTiles)
			bCt[k][j] = encryptVecAtScale(ctx, vec, inputLevel, bScale)
		}
	}
	return aCt, bCt
}

// encryptVecAtScale is the BMM-I counterpart of encryptVecsAtScale from
// moai_runner.go; kept local to bmm1_cipher.go to avoid entangling the
// two files.
func encryptVecAtScale(
	ctx *HEContext, vec []float64, level int, scale rlwe.Scale,
) *rlwe.Ciphertext {
	pt := ckks.NewPlaintext(ctx.Params, level)
	pt.Scale = scale
	if err := ctx.Encoder.Encode(PadToSlots(vec, ctx.NHE), pt); err != nil {
		panic(err)
	}
	ct, err := ctx.Encryptor.EncryptNew(pt)
	if err != nil {
		panic(err)
	}
	return ct
}

// ---------------------------------------------------------------------------
// Single-block kernels
//
// Each returns the product block's ciphertext at level inputLevel − 1.
// ---------------------------------------------------------------------------

// Bmm1HE is the no-hoisting single-block kernel (Python bmm1_he).
func Bmm1HE(
	eval *ckks.Evaluator,
	aCt, bCt *rlwe.Ciphertext,
	n, m, p, nHE int,
) *rlwe.Ciphertext {
	rotsA, rotsB := blockRotLists(n, m, p)
	return bmm1Accumulate(eval, aCt, bCt, rotsA, rotsB, m, nHE, nil, nil)
}

// Bmm1HEHoisted hoists A and B once each inside this call (Python
// bmm1_he_hoisted).
func Bmm1HEHoisted(
	eval *ckks.Evaluator,
	aCt, bCt *rlwe.Ciphertext,
	n, m, p, nHE int,
) *rlwe.Ciphertext {
	rotsA, rotsB := blockRotLists(n, m, p)
	// beforeHoist := TakeMemSnap()
	aHoist, err := eval.RotateHoistedNew(aCt, distinctNonZero(rotsA))
	if err != nil {
		panic(fmt.Errorf("hoist A: %w", err))
	}
	bHoist, err := eval.RotateHoistedNew(bCt, distinctNonZero(rotsB))
	if err != nil {
		panic(fmt.Errorf("hoist B: %w", err))
	}
	// afterHoist := TakeMemSnap()
	// PrintMemDelta("BMM1 per-block hoist temporary A+B", beforeHoist, afterHoist)
	return bmm1Accumulate(eval, aCt, bCt, rotsA, rotsB, m, nHE, aHoist, bHoist)
}

// Bmm1HEHoistedA is the kernel used when A has already been hoisted by
// the caller (strategy "prehoisted_a"). `aHoist` is the map produced by
// eval.RotateHoistedNew on the A block; `bCt` is the raw B ciphertext.
func Bmm1HEHoistedA(
	eval *ckks.Evaluator,
	aHoist map[int]*rlwe.Ciphertext,
	bCt *rlwe.Ciphertext,
	n, m, p, nHE int,
) *rlwe.Ciphertext {
	rotsA, rotsB := blockRotLists(n, m, p)
	bHoist, err := eval.RotateHoistedNew(bCt, distinctNonZero(rotsB))
	if err != nil {
		panic(fmt.Errorf("hoist B: %w", err))
	}
	return bmm1Accumulate(eval, nil, bCt, rotsA, rotsB, m, nHE, aHoist, bHoist)
}

// Bmm1HEHoistedAB is the kernel used when both A and B are pre-hoisted
// (strategy "prehoisted_ab").
func Bmm1HEHoistedAB(
	eval *ckks.Evaluator,
	aHoist, bHoist map[int]*rlwe.Ciphertext,
	n, m, p, nHE int,
) *rlwe.Ciphertext {
	rotsA, rotsB := blockRotLists(n, m, p)
	return bmm1Accumulate(eval, nil, nil, rotsA, rotsB, m, nHE, aHoist, bHoist)
}

// bmm1Accumulate is the shared inner loop. When aHoist/bHoist is non-nil,
// it is consulted before falling back to a fresh rotation. aCt and bCt may
// be nil when the corresponding hoist map is populated for all required
// shifts (including the identity, which is handled explicitly).
func bmm1Accumulate(
	eval *ckks.Evaluator,
	aCt, bCt *rlwe.Ciphertext, // may be nil if aHoist / bHoist fully cover the shift set
	rotsA, rotsB []int,
	m, nHE int,
	aHoist, bHoist map[int]*rlwe.Ciphertext,
) *rlwe.Ciphertext {

	var acc *rlwe.Ciphertext

	for i := 0; i < m; i++ {
		a := pickRot(eval, aCt, rotsA[i], aHoist)
		b := pickRot(eval, bCt, rotsB[i], bHoist)

		// ct × ct with rescale. Because B was encrypted at scale Q[L],
		// the rescale lands the product at the default scale, level L-1.
		prod, err := eval.MulNew(a, b)
		if err != nil {
			panic(fmt.Errorf("mul (i=%d): %w", i, err))
		}
		if err := eval.Rescale(prod, prod); err != nil {
			panic(fmt.Errorf("rescale (i=%d): %w", i, err))
		}

		if acc == nil {
			acc = prod
		} else {
			if err := eval.Add(acc, prod, acc); err != nil {
				panic(err)
			}
		}
	}
	// if acc != nil {
	// 	if err := eval.Relinearize(acc, acc); err != nil {
	// 		panic(fmt.Errorf("final relin: %w", err))
	// 	}
	// }
	return acc
}

// pickRot returns the rotation of `src` by `shift`. Shift=0 is the
// identity; otherwise the hoist map wins when present, else we do a
// fresh RotateNew.
func pickRot(
	eval *ckks.Evaluator,
	src *rlwe.Ciphertext,
	shift int,
	hoist map[int]*rlwe.Ciphertext,
) *rlwe.Ciphertext {
	if shift == 0 {
		// Identity — prefer the raw input if we have it, else a hoist map
		// that happens to carry the identity (it usually won't).
		if src != nil {
			return src
		}
		if hoist != nil {
			if ct, ok := hoist[0]; ok {
				return ct
			}
		}
		panic("bmm1: identity shift requested but neither src nor hoist[0] available")
	}
	if hoist != nil {
		if ct, ok := hoist[shift]; ok {
			return ct
		}
	}
	if src == nil {
		panic(fmt.Errorf("bmm1: hoist map missing shift %d and no raw src to rotate", shift))
	}
	ct, err := eval.RotateNew(src, shift)
	if err != nil {
		panic(fmt.Errorf("rotate by %d: %w", shift, err))
	}
	return ct
}

// ---------------------------------------------------------------------------
// Block-level strategies
// ---------------------------------------------------------------------------

// MatMulBmm1HE dispatches to the chosen hoisting strategy. Returns a 2D
// slice of ciphertexts indexed as C[i][j] for i ∈ [0, N/s_n), j ∈ [0, P/s_p).
func MatMulBmm1HE(
	eval *ckks.Evaluator,
	aCt, bCt [][]*rlwe.Ciphertext,
	N, M, P, sN, sM, sP, nHE int,
	hoisting Bmm1Hoisting,
) [][]*rlwe.Ciphertext {
	switch hoisting {
	case HoistNone:
		fmt.Println("BMM1 hoisting: none")
		return bmm1MatMulNone(eval, aCt, bCt, N, M, P, sN, sM, sP, nHE)
	case HoistPerBlock:
		fmt.Println("BMM1 hoisting: per_block")
		return bmm1MatMulPerBlock(eval, aCt, bCt, N, M, P, sN, sM, sP, nHE)
	case HoistPreA:
		fmt.Println("BMM1 hoisting: prehoisted_a")
		return bmm1MatMulPreA(eval, aCt, bCt, N, M, P, sN, sM, sP, nHE)
	case HoistPreAB:
		fmt.Println("BMM1 hoisting: prehoisted_ab")
		return bmm1MatMulPreAB(eval, aCt, bCt, N, M, P, sN, sM, sP, nHE)
	default:
		panic(fmt.Errorf("unknown BMM-I hoisting strategy %q", hoisting))
	}
}

func bmm1MatMulNone(
	eval *ckks.Evaluator,
	aCt, bCt [][]*rlwe.Ciphertext,
	N, M, P, sN, sM, sP, nHE int,
) [][]*rlwe.Ciphertext {
	aRows, kBlocks, bCols := N/sN, M/sM, P/sP
	C := make([][]*rlwe.Ciphertext, aRows)
	for i := 0; i < aRows; i++ {
		C[i] = make([]*rlwe.Ciphertext, bCols)
		for j := 0; j < bCols; j++ {
			var acc *rlwe.Ciphertext
			for k := 0; k < kBlocks; k++ {
				block := Bmm1HE(eval, aCt[i][k], bCt[k][j], sN, sM, sP, nHE)
				if acc == nil {
					acc = block
				} else {
					if err := eval.Add(acc, block, acc); err != nil {
						panic(err)
					}
				}
			}
			// [LATE-RELIN] One relin per output block.
			if acc != nil {
				if err := eval.Relinearize(acc, acc); err != nil {
					panic(fmt.Errorf("final relin (i=%d, j=%d): %w", i, j, err))
				}
			}
			C[i][j] = acc
		}
	}
	return C
}

func bmm1MatMulPerBlock(
	eval *ckks.Evaluator,
	aCt, bCt [][]*rlwe.Ciphertext,
	N, M, P, sN, sM, sP, nHE int,
) [][]*rlwe.Ciphertext {
	aRows, kBlocks, bCols := N/sN, M/sM, P/sP
	C := make([][]*rlwe.Ciphertext, aRows)
	for i := 0; i < aRows; i++ {
		C[i] = make([]*rlwe.Ciphertext, bCols)
		for j := 0; j < bCols; j++ {
			var acc *rlwe.Ciphertext
			for k := 0; k < kBlocks; k++ {
				block := Bmm1HEHoisted(eval, aCt[i][k], bCt[k][j], sN, sM, sP, nHE)
				if acc == nil {
					acc = block
				} else {
					if err := eval.Add(acc, block, acc); err != nil {
						panic(err)
					}
				}
			}
			// [LATE-RELIN] One relin per output block.
			if acc != nil {
				if err := eval.Relinearize(acc, acc); err != nil {
					panic(fmt.Errorf("final relin (i=%d, j=%d): %w", i, j, err))
				}
			}
			C[i][j] = acc
		}
	}
	return C
}

func bmm1MatMulPreA(
	eval *ckks.Evaluator,
	aCt, bCt [][]*rlwe.Ciphertext,
	N, M, P, sN, sM, sP, nHE int,
) [][]*rlwe.Ciphertext {
	aRows, kBlocks, bCols := N/sN, M/sM, P/sP

	rotsA, _ := blockRotLists(sN, sM, sP)
	distinctA := distinctNonZero(rotsA)
	beforeHoist := TakeMemSnap()
	// Pre-hoist every A block once.
	aHoist := make([][]map[int]*rlwe.Ciphertext, aRows)
	for i := 0; i < aRows; i++ {
		aHoist[i] = make([]map[int]*rlwe.Ciphertext, kBlocks)
		for k := 0; k < kBlocks; k++ {
			h, err := eval.RotateHoistedNew(aCt[i][k], distinctA)
			if err != nil {
				panic(fmt.Errorf("pre-hoist A[%d][%d]: %w", i, k, err))
			}
			// Inject the identity so pickRot can read hoist[0] without
			// needing the raw ciphertext.
			h[0] = aCt[i][k]
			aHoist[i][k] = h
		}
	}
	afterHoist := TakeMemSnap()
	PrintMemDelta("BMM1 pre-hoisted A ciphertexts", beforeHoist, afterHoist)

	C := make([][]*rlwe.Ciphertext, aRows)
	for i := 0; i < aRows; i++ {
		C[i] = make([]*rlwe.Ciphertext, bCols)
		for j := 0; j < bCols; j++ {
			var acc *rlwe.Ciphertext
			for k := 0; k < kBlocks; k++ {
				block := Bmm1HEHoistedA(eval, aHoist[i][k], bCt[k][j], sN, sM, sP, nHE)
				if acc == nil {
					acc = block
				} else {
					if err := eval.Add(acc, block, acc); err != nil {
						panic(err)
					}
				}
			}
			// [LATE-RELIN] One relin per output block.
			if acc != nil {
				if err := eval.Relinearize(acc, acc); err != nil {
					panic(fmt.Errorf("final relin (i=%d, j=%d): %w", i, j, err))
				}
			}
			C[i][j] = acc
		}
	}
	return C
}

func bmm1MatMulPreAB(
	eval *ckks.Evaluator,
	aCt, bCt [][]*rlwe.Ciphertext,
	N, M, P, sN, sM, sP, nHE int,
) [][]*rlwe.Ciphertext {
	aRows, kBlocks, bCols := N/sN, M/sM, P/sP

	rotsA, rotsB := blockRotLists(sN, sM, sP)
	distinctA := distinctNonZero(rotsA)
	distinctB := distinctNonZero(rotsB)

	// Pre-hoist A and B.
	beforeHoist := TakeMemSnap()
	aHoist := make([][]map[int]*rlwe.Ciphertext, aRows)
	for i := 0; i < aRows; i++ {
		aHoist[i] = make([]map[int]*rlwe.Ciphertext, kBlocks)
		for k := 0; k < kBlocks; k++ {
			h, err := eval.RotateHoistedNew(aCt[i][k], distinctA)
			if err != nil {
				panic(fmt.Errorf("pre-hoist A[%d][%d]: %w", i, k, err))
			}
			h[0] = aCt[i][k]
			aHoist[i][k] = h
		}
	}
	bHoist := make([][]map[int]*rlwe.Ciphertext, kBlocks)
	for k := 0; k < kBlocks; k++ {
		bHoist[k] = make([]map[int]*rlwe.Ciphertext, bCols)
		for j := 0; j < bCols; j++ {
			h, err := eval.RotateHoistedNew(bCt[k][j], distinctB)
			if err != nil {
				panic(fmt.Errorf("pre-hoist B[%d][%d]: %w", k, j, err))
			}
			h[0] = bCt[k][j]
			bHoist[k][j] = h
		}
	}
	afterHoist := TakeMemSnap()
	PrintMemDelta("BMM1 pre-hoisted A and B ciphertexts", beforeHoist, afterHoist)

	C := make([][]*rlwe.Ciphertext, aRows)
	for i := 0; i < aRows; i++ {
		C[i] = make([]*rlwe.Ciphertext, bCols)
		for j := 0; j < bCols; j++ {
			var acc *rlwe.Ciphertext
			for k := 0; k < kBlocks; k++ {
				block := Bmm1HEHoistedAB(eval, aHoist[i][k], bHoist[k][j], sN, sM, sP, nHE)
				if acc == nil {
					acc = block
				} else {
					if err := eval.Add(acc, block, acc); err != nil {
						panic(err)
					}
				}
			}
			// [LATE-RELIN] One relin per output block.
			if acc != nil {
				if err := eval.Relinearize(acc, acc); err != nil {
					panic(fmt.Errorf("final relin (i=%d, j=%d): %w", i, j, err))
				}
			}
			C[i][j] = acc
		}
	}
	return C
}
