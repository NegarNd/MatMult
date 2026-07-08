// thor_runner.go
//
// Plaintext and HE test runners for THOR lower-lower CC-MM (Algorithm 2,
// Moon et al., CCS 2025). These are the Go equivalents of
//
//   plaintext/thor_plain.py  :: run_thor_plaintext
//   ciphertext/thor_cipher.py :: run_thor_he
//
// in the original Python project.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ---------------------------------------------------------------------------
// Suite wrappers — called from the menu in main.go
// ---------------------------------------------------------------------------

// ThorPlaintextSuite mirrors the Python `thor_plaintext(...)` handler.
// Add / remove configurations here to match your experiments.
func ThorPlaintextSuite(verify bool, nTrials int) {
	// The Python driver only runs one tiny sanity check at the plaintext
	// layer (d=n=2, H=2, s=8); this port does the same by default, plus a
	// more realistic configuration so the plaintext algorithm is exercised
	// at production-like dimensions.
	// RunThorPlaintext(2, 2, 2, 8, verify)
	// RunThorPlaintext(128, 128, 1, 4096, verify)
	// RunThorPlaintext(256, 256, 1, 4096, verify)
	RunThorPlaintext(128, 128, 1, 4096, verify)
}

// ThorCiphertextSuite mirrors the Python `thor_ciphertext(...)` handler.
func ThorCiphertextSuite(verify bool, nTrials int) {
	ctx := InitLattigo(DefaultParams)
	const inputLevel = 4

	for _, size := range []int{128, 256, 512, 1024} {
		RunThorHE(ctx, size, size, 1, inputLevel, nTrials, verify)
	}
}

// ---------------------------------------------------------------------------
// Plaintext runner
// ---------------------------------------------------------------------------

// RunThorPlaintext executes Algorithm 2 on plaintext "slot vectors" (plain
// []float64 slices in the THOR layout) and verifies the output against
// MatmulPlain. Useful for debugging the packing and algorithm logic without
// any CKKS noise.
func RunThorPlaintext(d, n, H, s int, verify bool) {
	if s%(n*H) != 0 {
		fmt.Printf("  [SKIP]  n·H = %d does not divide s = %d\n", n*H, s)
		return
	}
	c := s / (n * H)
	if d%c != 0 || n%c != 0 {
		fmt.Printf("  [SKIP]  c = %d does not divide d = %d or n = %d\n", c, d, n)
		return
	}

	rng := rand.New(rand.NewSource(42))
	As := make([][][]float64, H)
	Bs := make([][][]float64, H)
	Cref := make([][][]float64, H)
	for z := 0; z < H; z++ {
		As[z] = RandomMatrix(d, n, rng)
		Bs[z] = RandomMatrix(n, n, rng)
		Cref[z] = MatmulPlain(As[z], Bs[z])
	}

	aPacked := EncodeBatched(As, c, H)
	bPacked := EncodeBatched(Bs, c, H)
	bRep := Replication(bPacked, n, c, H)

	masks := make([]plainEllMask, n-1)
	for ell := 1; ell < n; ell++ {
		mu0, mu1, mu2 := BuildEllMasks(ell, c, n, H, s)
		masks[ell-1] = plainEllMask{
			Mu0: mu0, Mu1: mu1, Mu2: mu2,
			HasMu0: AnyNonZero(mu0),
			HasMu2: AnyNonZero(mu2),
		}
	}

	start := time.Now()
	cVecs, counts := thorCCMatMulPlain(aPacked, bRep, masks, d, n, H, s)
	elapsed := time.Since(start)

	Cout := DecodeBatched(cVecs, d, n, c, H)

	params := fmt.Sprintf("d=%d, n=%d, H=%d, s=%d  →  c=%d, m_c=%d",
		d, n, H, s, c, d/c)
	if verify {
		maxErr := MaxAbsError(Cref, Cout)
		status := "PASS ✓"
		if maxErr > 1e-9 { // plaintext has no CKKS noise; tolerance is tight
			status = "FAIL ✗"
		}
		fmt.Printf("  [%s]  %s  |  %v  |  max_err=%.2e\n  |  %s\n",
			status, params, elapsed, maxErr, counts)
	} else {
		fmt.Printf("  [RUN ]  %s  |  %v\n  |  %s\n",
			params, elapsed, counts)
	}
}

// ---------------------------------------------------------------------------
// Plaintext kernel — a literal port of ThorCCMatMulHE working on []float64
// ---------------------------------------------------------------------------

type plainEllMask struct {
	Mu0, Mu1, Mu2  []float64
	HasMu0, HasMu2 bool
}

// thorCCMatMulPlain is Algorithm 2 on plaintext slot vectors. The
// structure mirrors ThorCCMatMulHE one-to-one, so any change in the HE
// kernel should be replicated here (and vice versa).
//
// Returns the output slot-vectors AND an OpCounts tally, so the caller
// can report how many rotations / CtCt muls / CtPt muls Algorithm 2 would
// perform on ciphertexts at this configuration.
func thorCCMatMulPlain(
	pAs, pBRep [][]float64,
	masks []plainEllMask,
	d, n, H, s int,
) ([][]float64, OpCounts) {

	c := s / (n * H)
	nH := n * H
	mc := d / c

	var counts OpCounts

	// Lines 4-8: intermediate products.
	//   Each multiplication here is ct×ct in the HE version.
	//   Each rotation corresponds to Rot(ct.A_j ; shift_ℓ).
	pCjl := make([][][]float64, mc)
	for j := 0; j < mc; j++ {
		pCjl[j] = make([][]float64, n)

		pCjl[j][0] = mulSlots(pAs[j], pBRep[0])
		counts.CtCtMuls++

		for ell := 1; ell < n; ell++ {
			shift := (-n*(ell%c) + ell) * H

			rot := rotateSlots(pAs[j], shift)
			counts.Rotations++

			pCjl[j][ell] = mulSlots(rot, pBRep[ell])
			counts.CtCtMuls++
		}
	}

	// Lines 9-17: masking and accumulation.
	//   Each multiplication by a mask is ct×pt in the HE version.
	//   μ_{ℓ,3} is computed via subtraction (v − v₀ − v₁ − v₂) and
	//   costs no multiplication — it is intentionally NOT counted.
	accPrime := make([][]float64, mc)
	accDPrime := make([][]float64, mc)

	for ell := 1; ell < n; ell++ {
		m := masks[ell-1]
		ellQ := ell / c
		for j := 0; j < mc; j++ {
			v := pCjl[j][ell]
			jSame := (j + ellQ) % mc
			jNext := (j + ellQ + 1) % mc

			v1 := mulSlots(v, m.Mu1) // μ_{ℓ,1}
			counts.CtPtMuls++
			accPrime[jSame] = addSlots(accPrime[jSame], v1)

			var v0, v2 []float64
			if m.HasMu0 {
				v0 = mulSlots(v, m.Mu0) // μ_{ℓ,0}
				counts.CtPtMuls++
				accPrime[jNext] = addSlots(accPrime[jNext], v0)
			}
			if m.HasMu2 {
				v2 = mulSlots(v, m.Mu2) // μ_{ℓ,2}
				counts.CtPtMuls++
				accDPrime[jNext] = addSlots(accDPrime[jNext], v2)
			}

			// v3 = v - v0 - v1 - v2  (free μ_{ℓ,3} — NOT counted)
			v3 := make([]float64, s)
			copy(v3, v)
			if v0 != nil {
				for k := range v3 {
					v3[k] -= v0[k]
				}
			}
			for k := range v3 {
				v3[k] -= v1[k]
			}
			if v2 != nil {
				for k := range v3 {
					v3[k] -= v2[k]
				}
			}
			accDPrime[jSame] = addSlots(accDPrime[jSame], v3)
		}
	}

	// Line 18: final assembly.
	//   Exactly one Rot(·, -nH) per non-empty accDPrime[j].
	out := make([][]float64, mc)
	for j := 0; j < mc; j++ {
		res := make([]float64, s)
		copy(res, pCjl[j][0])
		if accPrime[j] != nil {
			for k := range res {
				res[k] += accPrime[j][k]
			}
		}
		if accDPrime[j] != nil {
			rolled := rotateSlots(accDPrime[j], -nH)
			counts.Rotations++
			for k := range res {
				res[k] += rolled[k]
			}
		}
		out[j] = res
	}
	return out, counts
}

// ---------------------------------------------------------------------------
// HE runner
// ---------------------------------------------------------------------------

// RunThorHE executes Algorithm 2 at one configuration on ciphertexts.
// This is a straight extraction of the driver that used to live in main.go.
func RunThorHE(
	ctx *HEContext,
	d, n, H, inputLevel, nTrials int,
	verify bool,
) {
	params := ctx.Params
	s := ctx.NHE
	if s%(n*H) != 0 {
		fmt.Printf("  [SKIP]  n·H = %d does not divide s = %d\n", n*H, s)
		return
	}
	c := s / (n * H)
	if d%c != 0 || n%c != 0 {
		fmt.Printf("  [SKIP]  c = %d does not divide d = %d or n = %d\n", c, d, n)
		return
	}
	mc := d / c

	// -------- 1. Build an evaluator with the algorithm's rotation keys --
	eval := ctx.WithRotations(RequiredRotations(d, n, H, s))

	// -------- 2. Generate random operands and the plaintext reference ---
	rng := rand.New(rand.NewSource(42))
	As := make([][][]float64, H)
	Bs := make([][][]float64, H)
	var Cref [][][]float64
	for z := 0; z < H; z++ {
		As[z] = RandomMatrix(d, n, rng)
		Bs[z] = RandomMatrix(n, n, rng)
	}
	if verify {
		Cref = make([][][]float64, H)
		for z := 0; z < H; z++ {
			Cref[z] = MatmulPlain(As[z], Bs[z])
		}
	}

	// -------- 3. Pack A and B into the THOR layout ----------------------
	aPacked := EncodeBatched(As, c, H)
	bPacked := EncodeBatched(Bs, c, H)
	bRep := Replication(bPacked, n, c, H)

	// -------- 4. Encrypt.
	// Scale trick (see thor_cipher.go for the full explanation):
	//   A        is encrypted at scale S        (default scale)
	//   B        is encrypted at scale Q[L]     → product lands at exact S
	//   masks    are encoded at scale Q[L-1]    → product lands at exact S
	// -------------------------------------------------------------------
	ctAs := make([]*rlwe.Ciphertext, mc)
	for j := 0; j < mc; j++ {
		pt := ckks.NewPlaintext(params, inputLevel)
		if err := ctx.Encoder.Encode(PadToSlots(aPacked[j], s), pt); err != nil {
			panic(err)
		}
		var err error
		if ctAs[j], err = ctx.Encryptor.EncryptNew(pt); err != nil {
			panic(err)
		}
	}

	scaleB := rlwe.NewScale(params.Q()[inputLevel])
	ctBRep := make([]*rlwe.Ciphertext, n)
	for ell := 0; ell < n; ell++ {
		pt := ckks.NewPlaintext(params, inputLevel)
		pt.Scale = scaleB
		if err := ctx.Encoder.Encode(PadToSlots(bRep[ell], s), pt); err != nil {
			panic(err)
		}
		var err error
		if ctBRep[ell], err = ctx.Encryptor.EncryptNew(pt); err != nil {
			panic(err)
		}
	}

	scaleMask := rlwe.NewScale(params.Q()[inputLevel-1])
	beforeMasks := TakeMemSnap(true)
	masks := make([]EllMask, n-1)
	for ell := 1; ell < n; ell++ {
		mu0, mu1, mu2 := BuildEllMasks(ell, c, n, H, s)
		m := EllMask{
			HasMu0: AnyNonZero(mu0),
			HasMu2: AnyNonZero(mu2),
			Mu1:    EncodePlaintext(ctx.Encoder, params, PadToSlots(mu1, s), inputLevel-1, scaleMask),
		}
		if m.HasMu0 {
			m.Mu0 = EncodePlaintext(ctx.Encoder, params, PadToSlots(mu0, s), inputLevel-1, scaleMask)
		}
		if m.HasMu2 {
			m.Mu2 = EncodePlaintext(ctx.Encoder, params, PadToSlots(mu2, s), inputLevel-1, scaleMask)
		}
		masks[ell-1] = m
	}
	afterMasks := TakeMemSnap(true)
	PrintMemDelta("THORCCMatMul masks", beforeMasks, afterMasks)
	// -------- 5. Timed kernel (nTrials repetitions) ---------------------
	var ctOut []*rlwe.Ciphertext
	elapseds := make([]time.Duration, 0, nTrials)
	for i := 0; i < nTrials; i++ {
		// beforeAlg := TakeMemSnap(true)
		// mon := StartPeakMemMonitor(10 * time.Millisecond)
		start := time.Now()
		ctOut = ThorCCMatMulHELazyRelin(eval, ctAs, ctBRep, masks, d, n, H, s)
		elapsed := time.Since(start)
		// mon.Stop()
		// afterAlg := TakeMemSnap(true)
		// mon.PrintPeak("THORCCMatMul peak memory usage", beforeAlg)
		// PrintMemDelta("THORCCMatMul total function memory", beforeAlg, afterAlg)
		elapseds = append(elapseds, elapsed)
	}

	// -------- 6. Decrypt and optionally verify --------------------------
	cVecs := make([][]float64, mc)
	for j := 0; j < mc; j++ {
		pt := ctx.Decryptor.DecryptNew(ctOut[j])
		buf := make([]float64, s)
		if err := ctx.Encoder.Decode(pt, buf); err != nil {
			panic(err)
		}
		cVecs[j] = buf
	}
	Cout := DecodeBatched(cVecs, d, n, c, H)

	paramStr := fmt.Sprintf("d=%d, n=%d, H=%d, s=%d  →  c=%d, m_c=%d  level=%d",
		d, n, H, s, c, mc, inputLevel)

	status := "RUN "
	maxErr := 0.0
	if verify {
		maxErr = MaxAbsError(Cref, Cout)
		if maxErr < 1e-3 {
			status = "PASS ✓"
		} else {
			status = "FAIL ✗"
		}
	}

	// Print all runtimes
	fmt.Printf("  [TIMES] Runtimes: [")
	for i, dur := range elapseds {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Print(dur)
	}
	fmt.Println("]")

	mean, stdev := meanStdev(elapseds)
	tail := ""
	if len(elapseds) > 1 {
		tail = fmt.Sprintf("  ±  %v  (n=%d)", stdev, len(elapseds))
	}
	fmt.Printf("  [%s]  %s\n  |  %v%s", status, paramStr, mean, tail)
	if verify {
		fmt.Printf("  |  max_err=%.2e", maxErr)
	}
	fmt.Println()
}

func meanStdev(ds []time.Duration) (mean, stdev time.Duration) {
	if len(ds) == 0 {
		return 0, 0
	}
	var sum int64
	for _, d := range ds {
		sum += int64(d)
	}
	meanInt := sum / int64(len(ds))
	mean = time.Duration(meanInt)
	if len(ds) < 2 {
		return mean, 0
	}
	var sqSum float64
	for _, d := range ds {
		diff := float64(int64(d) - meanInt)
		sqSum += diff * diff
	}
	variance := sqSum / float64(len(ds)-1)
	stdev = time.Duration(int64(math.Sqrt(variance)))
	return mean, stdev
}
