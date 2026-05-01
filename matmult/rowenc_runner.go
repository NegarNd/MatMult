// row_runner.go
//
// Plaintext and HE test runners for row-packing matrix multiplication.
// Go equivalents of
//
//     plaintext/rowEnc_plain.py  :: run_row_plaintext
//     ciphertext/rowEnc_cipher.py :: run_row_he

package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// ---------------------------------------------------------------------------
// Suite wrappers — called from the menu in main.go
// ---------------------------------------------------------------------------

// RowPlaintextSuite mirrors the Python `row_plaintext(...)` handler. It
// sweeps n ∈ {128, 256, 512, 1024, 2048}; trim or extend as you like.
func RowPlaintextSuite(verify bool, nTrials int) {
	for _, n := range []int{128, 256, 512, 1024, 2048} {
		RunRowPlaintext(n, verify)
	}
}

// RowCiphertextSuite mirrors the Python `row_ciphertext(...)` handler.
// The Python driver runs HE only at small sizes (n ∈ {4, 8, 16, 32})
// because s = n² has to fit inside n_he slots.
func RowCiphertextSuite(verify bool, nTrials int) {
	ctx := InitLattigo(DefaultParams)
	const inputLevel = 4

	for _, n := range []int{4, 8, 16, 32} {
		RunRowHE(ctx, n, inputLevel, nTrials, verify)
	}
}

// ---------------------------------------------------------------------------
// Plaintext runner
// ---------------------------------------------------------------------------

// RunRowPlaintext exercises the plaintext kernel and reports timing,
// op counts, and (optionally) the max error against a reference matmul.
func RunRowPlaintext(n int, verify bool) {
	if n == 0 || n&(n-1) != 0 {
		fmt.Printf("  [SKIP]  n=%d is not a power of 2\n", n)
		return
	}

	rng := rand.New(rand.NewSource(42))
	A := RandomMatrix(n, n, rng)
	B := RandomMatrix(n, n, rng)
	Af := RowPack(A)
	Bf := RowPack(B)

	start := time.Now()
	Cflat, counts := rowPackMatMulPlain(Af, Bf, n)
	elapsed := time.Since(start)

	status := "RUN "
	maxErr := 0.0
	if verify {
		Cref := MatmulPlain(A, B)
		Cout := RowUnpack(Cflat, n)
		maxErr = MaxAbsError([][][]float64{Cref}, [][][]float64{Cout})
		if maxErr < 1e-9 {
			status = "PASS ✓"
		} else {
			status = "FAIL ✗"
		}
	}

	thRot, thPMult, thMult, thAdd, thKs := TheoreticalRowCosts(n)
	fmt.Printf("  [%s]  n=%d\n  |  %v\n  |  %s\n",
		status, n, elapsed, counts)
	if verify && maxErr >= 1e-9 {
		fmt.Printf("  |  max_err=%.2e\n", maxErr)
	}
	fmt.Printf("  |  th_rot=%6d  th_pmult=%6d  th_mult=%6d  th_add=%6d  th_ks=%6d\n",
		thRot, thPMult, thMult, thAdd, thKs)
}

// ---------------------------------------------------------------------------
// HE runner
// ---------------------------------------------------------------------------

// RunRowHE exercises the HE kernel at one (n, inputLevel) configuration,
// repeats the timed kernel nTrials times, decrypts, and verifies.
func RunRowHE(
	ctx *HEContext,
	n, inputLevel, nTrials int,
	verify bool,
) {
	nHE := ctx.NHE
	if n == 0 || n&(n-1) != 0 {
		fmt.Printf("  [SKIP]  n=%d is not a power of 2\n", n)
		return
	}
	if nHE < n*n {
		fmt.Printf("  [SKIP]  n_he=%d < n²=%d\n", nHE, n*n)
		return
	}

	// -------- 1. Galois keys ------------------------------------------------
	eval := ctx.WithRotations(RequiredRowRotations(n, nHE))

	// -------- 2. Random data + plaintext reference --------------------------
	rng := rand.New(rand.NewSource(42))
	A := RandomMatrix(n, n, rng)
	B := RandomMatrix(n, n, rng)
	var Cref [][]float64
	if verify {
		Cref = MatmulPlain(A, B)
	}

	// -------- 3. Encrypt inputs (default scale, level L) --------------------
	pt := ckks.NewPlaintext(ctx.Params, inputLevel)
	if err := ctx.Encoder.Encode(PadToSlots(RowPack(A), nHE), pt); err != nil {
		panic(err)
	}
	ctA, err := ctx.Encryptor.EncryptNew(pt)
	if err != nil {
		panic(err)
	}
	if err := ctx.Encoder.Encode(PadToSlots(RowPack(B), nHE), pt); err != nil {
		panic(err)
	}
	ctB, err := ctx.Encryptor.EncryptNew(pt)
	if err != nil {
		panic(err)
	}

	// -------- 4. Build the 2n masks once, outside the timed region ----------
	masks := BuildRowMasks(ctx, n, inputLevel)

	// -------- 5. Timed kernel -----------------------------------------------
	var ctOut *rlwe.Ciphertext
	elapseds := make([]time.Duration, 0, nTrials)
	for t := 0; t < nTrials; t++ {
		start := time.Now()
		ctOut = RowPackMatMulHE(eval, ctA, ctB, masks, n, nHE)
		elapseds = append(elapseds, time.Since(start))
	}

	// -------- 6. Decrypt and verify -----------------------------------------
	ptOut := ctx.Decryptor.DecryptNew(ctOut)
	buf := make([]float64, nHE)
	if err := ctx.Encoder.Decode(ptOut, buf); err != nil {
		panic(err)
	}
	Cout := RowUnpack(buf[:n*n], n)

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

	mean, stdev := meanStdev(elapseds)
	tail := ""
	if len(elapseds) > 1 {
		tail = fmt.Sprintf("  ±  %v  (n=%d)", stdev, len(elapseds))
	}
	thRot, thPMult, thMult, thAdd, thKs := TheoreticalRowCosts(n)

	fmt.Printf("  [%s]  n=%d  level=%d\n  |  %v%s",
		status, n, inputLevel, mean, tail)
	if verify {
		fmt.Printf("  |  max_err=%.2e", maxErr)
	}
	fmt.Printf("\n  |  th_rot=%6d  th_pmult=%6d  th_mult=%6d  th_add=%6d  th_ks=%6d\n",
		thRot, thPMult, thMult, thAdd, thKs)
}
