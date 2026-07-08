// bmm1_runner.go
//
// Plaintext and HE test runners for BMM-I bicyclic matrix multiplication.
// Go equivalents of
//
//     plaintext/bmm1_plain.py  :: run_bmm1_plaintext
//     ciphertext/bmm1_cipher.py :: run_bmm1_he
//
// in the original Python project.

package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

// ---------------------------------------------------------------------------
// Suite wrappers — called from the menu in main.go
// ---------------------------------------------------------------------------

// Bmm1PlaintextSuite mirrors the Python `bmm1_plaintext(...)` handler. The
// block dimensions are kept pairwise-coprime (s_n=43, s_m=45, s_p=44) so
// bicyclic decoding is exact, matching the Python driver.
func BMM1PlaintextSuite(verify bool, nTrials int) {
	configs := []struct {
		N, M, P, SN, SM, SP int
	}{
		// {129, 135, 132, 43, 45, 44},
		// {258, 270, 264, 43, 45, 44},
		// {516, 540, 528, 43, 45, 44},
		{1032, 1080, 1056, 43, 45, 44},
		// {2064, 2070, 2068, 43, 45, 44},
	}
	for _, c := range configs {
		RunBmm1Plaintext(c.N, c.M, c.P, c.SN, c.SM, c.SP, verify)
	}
}

// Bmm1CiphertextSuite mirrors the Python `bmm1_ciphertext(...)` handler.
// Runs the "none" strategy at three sizes; swap in any of HoistPerBlock /
// HoistPreA / HoistPreAB to benchmark different hoisting regimes.
func BMM1CiphertextSuite(verify bool, nTrials int) {
	ctx := InitLattigo(DefaultParams)
	const inputLevel = 4

	configs := []struct {
		N, M, P, SN, SM, SP int
	}{
		// {129, 135, 132, 43, 45, 44},
		// {258, 270, 264, 43, 45, 44},
		// {516, 540, 528, 43, 45, 44},
		// {1032, 1080, 1056, 43, 45, 44},
		// {2064, 2070, 2068, 43, 45, 44},
		// {127, 128, 125, 127, 128, 125},
		// {254, 256, 250, 127, 128, 125},
		// {508, 512, 500, 127, 128, 125},
		{1016, 1024, 1000, 127, 128, 125},
		// {1068, 1092, 1080, 89, 91, 90},
	}
	for _, c := range configs {
		// RunBmm1HE(ctx, c.N, c.M, c.P, c.SN, c.SM, c.SP, inputLevel, nTrials, HoistNone, verify) // Change to HoistPreAB for optimal performance
		RunBmm1HE(ctx, c.N, c.M, c.P, c.SN, c.SM, c.SP, inputLevel, nTrials, HoistPerBlock, verify)
		// RunBmm1HE(ctx, c.N, c.M, c.P, c.SN, c.SM, c.SP, inputLevel, nTrials, HoistPreA, verify)
		// RunBmm1HE(ctx, c.N, c.M, c.P, c.SN, c.SM, c.SP, inputLevel, nTrials, HoistPreAB, verify)
	}

	// configs2 := []struct {
	// 	N, M, P, SN, SM, SP int
	// }{
	// 	// {129, 135, 132, 43, 45, 44},
	// 	// {258, 270, 264, 43, 45, 44},
	// 	// {516, 540, 528, 43, 45, 44},
	// 	// {1032, 1080, 1056, 43, 45, 44},
	// 	{2064, 2070, 2068, 43, 45, 44},
	// }
	// for _, c := range configs2 {
	// 	RunBmm1HE(ctx, c.N, c.M, c.P, c.SN, c.SM, c.SP, inputLevel, nTrials, HoistNone, verify) // Change to HoistPreAB for optimal performance
	// 	RunBmm1HE(ctx, c.N, c.M, c.P, c.SN, c.SM, c.SP, inputLevel, nTrials, HoistPerBlock, verify)
	// 	RunBmm1HE(ctx, c.N, c.M, c.P, c.SN, c.SM, c.SP, inputLevel, nTrials, HoistPreA, verify)
	// 	// RunBmm1HE(ctx, c.N, c.M, c.P, c.SN, c.SM, c.SP, inputLevel, nTrials, HoistPreAB, verify)
	// }
	// Uncomment to sweep all four strategies at one size:
	// for _, h := range []Bmm1Hoisting{HoistNone, HoistPerBlock, HoistPreA, HoistPreAB} {
	//     RunBmm1HE(ctx, 129, 135, 132, 43, 45, 44, inputLevel, nTrials, h, verify)
	// }
}

// ---------------------------------------------------------------------------
// Plaintext runner
// ---------------------------------------------------------------------------

// RunBmm1Plaintext runs block BMM-I at plaintext level with a random
// problem, reports timing and op counts, and optionally verifies against
// a reference matmul.
func RunBmm1Plaintext(N, M, P, sN, sM, sP int, verify bool) {
	if N%sN != 0 || M%sM != 0 || P%sP != 0 {
		fmt.Printf("  [SKIP]  block dims do not divide matrix dims\n")
		return
	}

	rng := rand.New(rand.NewSource(42))
	A := RandomMatrix(N, M, rng)
	B := RandomMatrix(M, P, rng)

	aEnc, bEnc := EncodeBlocks(A, B, sN, sM, sP)

	start := time.Now()
	Cout, counts := bmm1MatMulPlain(aEnc, bEnc, N, M, P, sN, sM, sP)
	elapsed := time.Since(start)

	paramStr := fmt.Sprintf("(%d,%d,%d)  block=(%d,%d,%d)", N, M, P, sN, sM, sP)

	status := "RUN "
	maxErr := 0.0
	if verify {
		Cref := MatmulPlain(A, B)
		// Wrap both matrices into the (H=1, m, n) tensor shape MaxAbsError expects.
		maxErr = MaxAbsError([][][]float64{Cref}, [][][]float64{Cout})
		if maxErr < 1e-9 {
			status = "PASS ✓"
		} else {
			status = "FAIL ✗"
		}
	}

	thRot, thMult, thAdd, thKs := TheoreticalBmm1Costs(N, M, P, sN, sM, sP)
	fmt.Printf("  [%s]  %s\n  |  %v\n  |  %s\n", status, paramStr, elapsed, counts)
	if verify && maxErr >= 1e-9 {
		fmt.Printf("  |  max_err=%.2e\n", maxErr)
	}
	fmt.Printf("  |  th_rot=%6d  th_mul=%6d  th_add=%6d  th_ks=%6d\n",
		thRot, thMult, thAdd, thKs)
}

// ---------------------------------------------------------------------------
// HE runner
// ---------------------------------------------------------------------------

// RunBmm1HE runs block BMM-I at one configuration and one hoisting
// strategy on encrypted data, reports mean/stdev timing over nTrials, and
// optionally verifies against a plaintext reference.
func RunBmm1HE(
	ctx *HEContext,
	N, M, P, sN, sM, sP, inputLevel, nTrials int,
	hoisting Bmm1Hoisting,
	verify bool,
) {
	nHE := ctx.NHE
	if N%sN != 0 || M%sM != 0 || P%sP != 0 {
		fmt.Printf("  [SKIP]  block dims do not divide matrix dims\n")
		return
	}
	required := Bmm1RequiredSlots(sN, sM, sP)
	if nHE < required {
		fmt.Printf("  [SKIP]  n_he=%d < required=%d\n", nHE, required)
		return
	}

	// -------- 1. Galois keys -----------------------------------------------
	eval := ctx.WithRotations(RequiredBmm1Rotations(sN, sM, sP))

	// -------- 2. Random data + plaintext reference -------------------------
	rng := rand.New(rand.NewSource(42))
	A := RandomMatrix(N, M, rng)
	B := RandomMatrix(M, P, rng)

	var Cref [][]float64
	if verify {
		Cref = MatmulPlain(A, B)
	}

	// -------- 3. Encrypt blocks (once, outside the timed region) -----------
	aCt, bCt := EncodeAndEncryptBlocks(ctx, A, B, sN, sM, sP, inputLevel)

	// -------- 4. Timed kernel (nTrials repetitions) ------------------------
	var cCt [][]*rlwe.Ciphertext
	elapseds := make([]time.Duration, 0, nTrials)
	for t := 0; t < nTrials; t++ {
		// beforeAlg := TakeMemSnap(true)
		// mon := StartPeakMemMonitor(10 * time.Millisecond)
		start := time.Now()
		cCt = MatMulBmm1HE(eval, aCt, bCt, N, M, P, sN, sM, sP, nHE, hoisting)
		elapsed := time.Since(start)
		// mon.Stop()
		// afterAlg := TakeMemSnap(true)
		// PrintMemDelta("MatMulBmm1HE total function memory", beforeAlg, afterAlg)
		// mon.PrintPeak("MatMulBmm1HE peak memory usage", beforeAlg)
		elapseds = append(elapseds, elapsed)
	}

	// -------- 5. Decrypt and assemble --------------------------------------
	Cout := make([][]float64, N)
	for i := range Cout {
		Cout[i] = make([]float64, P)
	}
	for i := 0; i < N/sN; i++ {
		for j := 0; j < P/sP; j++ {
			pt := ctx.Decryptor.DecryptNew(cCt[i][j])
			buf := make([]float64, nHE)
			if err := ctx.Encoder.Decode(pt, buf); err != nil {
				panic(err)
			}
			dec := BicyclicDecode(buf[:sN*sP], sN, sP)
			for ii := 0; ii < sN; ii++ {
				copy(Cout[i*sN+ii][j*sP:(j+1)*sP], dec[ii])
			}
		}
	}

	// -------- 6. Report ----------------------------------------------------
	paramStr := fmt.Sprintf("(%d,%d,%d)  block=(%d,%d,%d)  level=%d  hoisting=%s",
		N, M, P, sN, sM, sP, inputLevel, hoisting)

	status := "RUN "
	maxErr := 0.0
	if verify {
		maxErr = MaxAbsError([][][]float64{Cref}, [][][]float64{Cout})
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
	thRot, thMult, thAdd, thKs := TheoreticalBmm1Costs(N, M, P, sN, sM, sP)

	fmt.Printf("  [%s]  %s\n  |  %v%s", status, paramStr, mean, tail)
	if verify {
		fmt.Printf("  |  max_err=%.2e", maxErr)
	}
	fmt.Printf("\n  |  th_rot=%6d  th_mul=%6d  th_add=%6d  th_ks=%6d\n",
		thRot, thMult, thAdd, thKs)
}
