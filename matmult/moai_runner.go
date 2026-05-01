// moai_runner.go
//
// Plaintext and HE test runners for MOAI Col×Col→Diag (Algorithm 3) and
// Diag×Col→Col (Algorithm 4). Go equivalents of
//
//     plaintext/moai_plain.py  :: run_moai_plaintext
//     ciphertext/moai_cipher.py :: run_moai_he
//
// in the original Python project.

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

// MoaiPlaintextSuite mirrors moai_plaintext() in run.py: for each size,
// run both the naive and the BSGS variant of Algorithm 3.
func MoaiPlaintextSuite(verify bool, nTrials int) {
	const nHE = 1 << 12 // matches the Python driver's 2^12 plaintext slots

	for _, size := range []int{128, 256, 512, 1024, 2048} {
		RunMoaiPlaintext(size, size, nHE, false, true, verify) // naive col×col
		RunMoaiPlaintext(size, size, nHE, true, true, verify)  // BSGS  col×col
	}
	// Optionally also exercise the diag×col (BSGS only) variant:
	// RunMoaiPlaintext(128, 128, nHE, true, false, verify)
}

// MoaiCiphertextSuite mirrors moai_ciphertext() in run.py.
func MoaiCiphertextSuite(verify bool, nTrials int) {
	ctx := InitLattigo(DefaultParams)
	const inputLevel = 4

	for _, size := range []int{2048} {
		RunMoaiHE(ctx, size, size, true /*bsgs*/, true /*colCol*/, inputLevel, nTrials, verify)
	}
}

// ---------------------------------------------------------------------------
// Plaintext runner
// ---------------------------------------------------------------------------

// RunMoaiPlaintext runs MOAI Algorithm 3 (col×col) or Algorithm 4
// (diag×col) at plaintext level and verifies against a NumPy-style
// reference matmul.
func RunMoaiPlaintext(m, dPrime, nHE int, bsgs, colCol, verify bool) {
	if nHE%m != 0 {
		fmt.Printf("  [SKIP]  m = %d does not divide n_he = %d\n", m, nHE)
		return
	}
	nBatch := nHE / m

	rng := rand.New(rand.NewSource(0))
	Qs := make([][][]float64, nBatch)
	Ks := make([][][]float64, nBatch)
	Vs := make([][][]float64, nBatch)
	for s := 0; s < nBatch; s++ {
		Qs[s] = RandomMatrix(m, dPrime, rng)
		Ks[s] = RandomMatrix(m, dPrime, rng)
		Vs[s] = RandomMatrix(m, dPrime, rng)
	}

	// Reference: QKT[s] = Qs[s] · Ks[s]^T  ,  Attn[s] = QKT[s] · Vs[s].
	QKTs := make([][][]float64, nBatch)
	Attns := make([][][]float64, nBatch)
	for s := 0; s < nBatch; s++ {
		QKTs[s] = matmulWithTranspose(Qs[s], Ks[s])
		Attns[s] = MatmulPlain(QKTs[s], Vs[s])
	}

	encQ := InterleavedColumnPack(Qs, nHE)
	encK := InterleavedColumnPack(Ks, nHE)
	encV := InterleavedColumnPack(Vs, nHE)

	start := time.Now()
	var (
		encOut [][]float64
		counts OpCounts
	)
	switch {
	case !colCol:
		encC := InterleavedDiagPack(QKTs, nHE)
		encOut, counts = moaiDiagColBSGS(encC, encV, m, dPrime, nBatch)
	case bsgs:
		encOut, counts = moaiColColBSGS(encQ, encK, m, dPrime, nBatch)
	default:
		encOut, counts = moaiColColNaive(encQ, encK, m, dPrime, nBatch)
	}
	elapsed := time.Since(start)

	params := fmt.Sprintf("m=%d, d'=%d, n_batch=%d, bsgs=%v, col_col=%v",
		m, dPrime, nBatch, bsgs, colCol)

	status := "RUN "
	maxErr := 0.0
	if verify {
		var got, ref [][][]float64
		if colCol {
			got = InterleavedDiagUnpack(encOut, m, nBatch)
			ref = QKTs
		} else {
			got = InterleavedColumnUnpack(encOut, m, nBatch)
			ref = Attns
		}
		maxErr = MaxAbsError(ref, got)
		if maxErr < 1e-9 {
			status = "PASS ✓"
		} else {
			status = "FAIL ✗"
		}
	}

	thMulN, thRotN, thKsN := TheoreticalNaive(m, dPrime)
	thMulB, thRotB, thKsB := TheoreticalBSGS(m, dPrime)

	fmt.Printf("  [%s]  %s\n  |  %v\n  |  %s\n", status, params, elapsed, counts)
	if verify && maxErr >= 1e-9 {
		fmt.Printf("  |  max_err=%.2e\n", maxErr)
	}
	fmt.Printf("  |  th_rot_n=%8d  th_mul_n=%8d  th_ks_n=%8d\n", thRotN, thMulN, thKsN)
	fmt.Printf("  |  th_rot_b=%8d  th_mul_b=%8d  th_ks_b=%8d\n", thRotB, thMulB, thKsB)
}

// matmulWithTranspose returns A · B^T.
func matmulWithTranspose(A, B [][]float64) [][]float64 {
	m, d := len(A), len(A[0])
	n := len(B) // B has shape (n, d) so B^T is (d, n)
	C := make([][]float64, m)
	for i := 0; i < m; i++ {
		C[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			var sum float64
			for k := 0; k < d; k++ {
				sum += A[i][k] * B[j][k]
			}
			C[i][j] = sum
		}
	}
	return C
}

// ---------------------------------------------------------------------------
// HE runner
// ---------------------------------------------------------------------------

// RunMoaiHE runs the HE version of MOAI at one configuration.
//
//	m, dPrime : problem size  (matrices are m × dPrime)
//	bsgs      : if true, use BSGS variant; otherwise naive.
//	colCol    : if true, run Col×Col → Diag (Alg. 3);
//	            if false, run Diag×Col → Col (Alg. 4, always BSGS).
//	inputLevel: starting level of the input ciphertexts.
//	nTrials   : number of timed repetitions of the kernel.
func RunMoaiHE(
	ctx *HEContext,
	m, dPrime int,
	bsgs, colCol bool,
	inputLevel, nTrials int,
	verify bool,
) {
	params := ctx.Params
	nHE := ctx.NHE
	if nHE%m != 0 {
		fmt.Printf("  [SKIP]  m = %d does not divide n_he = %d\n", m, nHE)
		return
	}
	nBatch := nHE / m

	// -------- 1. Galois keys for this algorithm's rotation set ----------
	var rots []int
	switch {
	case !colCol:
		rots = RequiredMoaiDiagColBSGSRotations(m, nBatch, nHE)
	case bsgs:
		rots = RequiredMoaiColColBSGSRotations(m, nBatch, nHE)
	default:
		rots = RequiredMoaiColColNaiveRotations(m, nBatch, nHE)
	}
	eval := ctx.WithRotations(rots)

	// -------- 2. Random data + plaintext reference ----------------------
	rng := rand.New(rand.NewSource(0))
	Qs := make([][][]float64, nBatch)
	Ks := make([][][]float64, nBatch)
	Vs := make([][][]float64, nBatch)
	for s := 0; s < nBatch; s++ {
		Qs[s] = RandomMatrix(m, dPrime, rng)
		Ks[s] = RandomMatrix(m, dPrime, rng)
		Vs[s] = RandomMatrix(m, dPrime, rng)
	}
	var (
		QKTs  [][][]float64
		Attns [][][]float64
	)
	if verify {
		QKTs = make([][][]float64, nBatch)
		Attns = make([][][]float64, nBatch)
		for s := 0; s < nBatch; s++ {
			QKTs[s] = matmulWithTranspose(Qs[s], Ks[s])
			Attns[s] = MatmulPlain(QKTs[s], Vs[s])
		}
	}

	// -------- 3. Pack and encrypt.
	// Q (or C) at default scale; K, V at scale Q[L] — see file header.
	packedQ := InterleavedColumnPack(Qs, nHE)
	packedK := InterleavedColumnPack(Ks, nHE)
	packedV := InterleavedColumnPack(Vs, nHE)

	ctQ := encryptVecsAtScale(ctx, packedQ, inputLevel, params.DefaultScale())
	scaleSecond := rlwe.NewScale(params.Q()[inputLevel])
	ctK := encryptVecsAtScale(ctx, packedK, inputLevel, scaleSecond)
	ctV := encryptVecsAtScale(ctx, packedV, inputLevel, scaleSecond)

	var ctC []*rlwe.Ciphertext
	if !colCol {
		// For diag×col we need QKT in diag-packed form.
		if QKTs == nil {
			// Generated above only when verify is true; create now.
			QKTs = make([][][]float64, nBatch)
			for s := 0; s < nBatch; s++ {
				QKTs[s] = matmulWithTranspose(Qs[s], Ks[s])
			}
		}
		packedC := InterleavedDiagPack(QKTs, nHE)
		// C plays the role of the "first" operand in diag×col.
		ctC = encryptVecsAtScale(ctx, packedC, inputLevel, params.DefaultScale())
	}

	// -------- 4. Timed kernel (nTrials repetitions) ---------------------
	var ctOut []*rlwe.Ciphertext
	elapseds := make([]time.Duration, 0, nTrials)
	for t := 0; t < nTrials; t++ {
		beforeAlg := TakeMemSnap()
		start := time.Now()
		switch {
		case !colCol:
			ctOut = MoaiDiagColBSGSHE(eval, ctC, ctV, m, dPrime, nBatch, nHE)
		case bsgs:
			ctOut = MoaiColColBSGSHE(eval, ctQ, ctK, m, dPrime, nBatch, nHE)
		default:
			ctOut = MoaiColColNaiveHE(eval, ctQ, ctK, m, dPrime, nBatch, nHE)
		}
		elapsed := time.Since(start)
		afterAlg := TakeMemSnap()
		PrintMemDelta("MoaiDiagColBSGSHE total function memory", beforeAlg, afterAlg)
		elapseds = append(elapseds, elapsed)
	}

	// -------- 5. Decrypt and optionally verify --------------------------
	decVecs := make([][]float64, len(ctOut))
	for i, ct := range ctOut {
		pt := ctx.Decryptor.DecryptNew(ct)
		buf := make([]float64, nHE)
		if err := ctx.Encoder.Decode(pt, buf); err != nil {
			panic(err)
		}
		decVecs[i] = buf
	}

	paramStr := fmt.Sprintf("m=%d, d'=%d, n_batch=%d, bsgs=%v, col_col=%v  level=%d",
		m, dPrime, nBatch, bsgs, colCol, inputLevel)

	status := "RUN "
	maxErr := 0.0
	if verify {
		var got, ref [][][]float64
		if colCol {
			got = InterleavedDiagUnpack(decVecs, m, nBatch)
			ref = QKTs
		} else {
			got = InterleavedColumnUnpack(decVecs, m, nBatch)
			ref = Attns
		}
		maxErr = MaxAbsError(ref, got)
		if maxErr < 1e-3 {
			status = "PASS ✓"
		} else {
			status = "FAIL ✗"
		}
	}

	// Print all runtimes in elapseds
	fmt.Print("  [runtimes]:")
	for i, elapsed := range elapseds {
		fmt.Printf("  %v", elapsed)
		if i < len(elapseds)-1 {
			fmt.Print(",")
		}
	}
	fmt.Println()
	mean, stdev := meanStdev(elapseds)
	tail := ""
	if len(elapseds) > 1 {
		tail = fmt.Sprintf("  ±  %v  (n=%d)", stdev, len(elapseds))
	}

	thMulN, thRotN, thKsN := TheoreticalNaive(m, dPrime)
	thMulB, thRotB, thKsB := TheoreticalBSGS(m, dPrime)

	fmt.Printf("  [%s]  %s\n  |  %v%s", status, paramStr, mean, tail)
	if verify {
		fmt.Printf("  |  max_err=%.2e", maxErr)
	}
	fmt.Printf("\n  |  th_rot_n=%8d  th_mul_n=%8d  th_ks_n=%8d\n", thRotN, thMulN, thKsN)
	fmt.Printf("  |  th_rot_b=%8d  th_mul_b=%8d  th_ks_b=%8d\n", thRotB, thMulB, thKsB)
}

// encryptVecsAtScale encodes each slot vector at the given level and scale,
// then encrypts. The scale parameter lets the caller implement the
// "encrypt second operand at Q[L]" trick described in the file header.
func encryptVecsAtScale(
	ctx *HEContext,
	vecs [][]float64,
	level int,
	scale rlwe.Scale,
) []*rlwe.Ciphertext {
	out := make([]*rlwe.Ciphertext, len(vecs))
	for i, v := range vecs {
		pt := ckks.NewPlaintext(ctx.Params, level)
		pt.Scale = scale
		if err := ctx.Encoder.Encode(PadToSlots(v, ctx.NHE), pt); err != nil {
			panic(err)
		}
		var err error
		if out[i], err = ctx.Encryptor.EncryptNew(pt); err != nil {
			panic(err)
		}
	}
	return out
}
