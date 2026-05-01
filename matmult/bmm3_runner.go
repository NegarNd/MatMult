// bmm3_runner.go
//
// HE test runner for BMM-III bicyclic matrix multiplication, Go port of
// ciphertext/bmm3_cipher.py → run_bmm3_he. There is no plaintext runner
// for BMM-III in this port (the algorithm is interesting specifically at
// problem sizes where the bicyclic encoding exceeds a single ciphertext,
// which makes a fast pure-Go plaintext reference less valuable).
//
// Menu wiring: option 12 (BMM-III — HE). Option 11 (BMM-III — plaintext)
// remains a stub; see runner_stubs.go.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// BMM3CiphertextSuite mirrors the Python `bmm3_ciphertext(...)` handler.
// Pairwise-coprime dimensions are required by the algorithm; the tuples
// below mirror the Python driver's defaults.
func BMM3CiphertextSuite(verify bool, nTrials int) {
	ctx := InitLattigo(DefaultParams)
	const inputLevel = 4
	const hoistBlockSize = 16

	configs := []struct {
		N, M, P int
	}{
		{128, 131, 129},
		{256, 259, 257},
		{512, 515, 513},
		{1024, 1027, 1025},
		// {2048, 2051, 2049},
	}

	// Default to the hoisted mode — the most interesting for the paper.
	// Swap in Bmm3ModeNaive / Bmm3ModeCached to benchmark the ladder.
	for _, c := range configs {
		RunBmm3HE(ctx, c.N, c.M, c.P, inputLevel, nTrials,
			Bmm3ModeHoisted, hoistBlockSize, verify)
	}
	// Uncomment to sweep all three modes at one size:
	// for _, mode := range []Bmm3Mode{Bmm3ModeNaive, Bmm3ModeCached, Bmm3ModeHoisted} {
	//     RunBmm3HE(ctx, 43, 45, 44, inputLevel, nTrials, mode, hoistBlockSize, verify)
	// }
}

// RunBmm3HE runs one BMM-III configuration at one mode, repeats the
// kernel nTrials times, decrypts, and optionally verifies against a
// plaintext reference.
func RunBmm3HE(
	ctx *HEContext,
	n, m, p, inputLevel, nTrials int,
	mode Bmm3Mode,
	hoistBlockSize int,
	verify bool,
) {
	nHE := ctx.NHE

	// -------- 1. Galois keys ------------------------------------------------
	eval := ctx.WithRotations(RequiredBmm3Rotations(n, m, p, nHE))

	// -------- 2. Random data + reference -----------------------------------
	rng := rand.New(rand.NewSource(42))
	A := RandomMatrix(n, m, rng)
	B := RandomMatrix(m, p, rng)
	var Cref [][]float64
	if verify {
		Cref = MatmulPlain(A, B)
	}

	// -------- 3. Bicyclic-encode + chunk + encrypt --------------------------
	aEnc := BicyclicEncode(A)
	bEnc := BicyclicEncode(B)
	aChunks := breakIntoChunks(aEnc, n*m, n*p, nHE)
	bChunks := breakIntoChunks(bEnc, m*p, n*p, nHE)

	aCts := encryptChunks(ctx, aChunks, inputLevel)
	bCts := encryptChunks(ctx, bChunks, inputLevel)

	// -------- 4. Timed kernel ----------------------------------------------
	var cChunks []*rlwe.Ciphertext
	elapseds := make([]time.Duration, 0, nTrials)
	for t := 0; t < nTrials; t++ {
		beforeAlg := TakeMemSnap()
		start := time.Now()
		cChunks = MatMulBmm3HE(eval, ctx, aCts, bCts, n, m, p, nHE, inputLevel,
			mode, hoistBlockSize)
		elapsed := time.Since(start)
		afterAlg := TakeMemSnap()
		PrintMemDelta("MatMulBmm3HE total function memory", beforeAlg, afterAlg)
		elapseds = append(elapseds, elapsed)
	}

	// -------- 5. Decrypt & decode ------------------------------------------
	stop := int(math.Ceil(float64(n*p) / float64(nHE)))
	raw := make([]float64, 0, stop*nHE)
	for s := 0; s < stop; s++ {
		pt := ctx.Decryptor.DecryptNew(cChunks[s])
		buf := make([]float64, nHE)
		if err := ctx.Encoder.Decode(pt, buf); err != nil {
			panic(err)
		}
		raw = append(raw, buf...)
	}
	Cout := BicyclicDecode(raw[:n*p], n, p)

	// -------- 6. Report ----------------------------------------------------
	paramStr := fmt.Sprintf("(%d,%d,%d)  r=%d  level=%d  mode=%s",
		n, m, p, smallestR(n, m, p), inputLevel, mode)
	if mode == Bmm3ModeHoisted {
		paramStr += fmt.Sprintf("  block=%d", hoistBlockSize)
	}

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
	fmt.Printf("  [%s]  %s\n  |  %v%s", status, paramStr, mean, tail)
	if verify {
		fmt.Printf("  |  max_err=%.2e", maxErr)
	}
	fmt.Println()
}

// encryptChunks encodes + encrypts every chunk at `inputLevel`, default
// scale. Helper local to BMM-III to keep the runner readable.
func encryptChunks(ctx *HEContext, chunks [][]float64, inputLevel int) []*rlwe.Ciphertext {
	cts := make([]*rlwe.Ciphertext, len(chunks))
	for i, vec := range chunks {
		pt := ckks.NewPlaintext(ctx.Params, inputLevel)
		if err := ctx.Encoder.Encode(PadToSlots(vec, ctx.NHE), pt); err != nil {
			panic(err)
		}
		ct, err := ctx.Encryptor.EncryptNew(pt)
		if err != nil {
			panic(err)
		}
		cts[i] = ct
	}
	return cts
}
