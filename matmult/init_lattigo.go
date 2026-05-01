// lattigo_init.go
//
// Lattigo-side scheme initialisation. This is the direct equivalent of
// `init_orion()` in the Python runner: it builds the CKKS parameters,
// generates the secret / public / relinearization keys, and returns a
// context bundle that every HE runner can reuse.
//
// Galois (rotation) keys are intentionally NOT generated here — the rotation
// set depends on the algorithm. Each algorithm builds its own evaluator via
// ctx.WithRotations(...) so we do not pay for rotations we do not need.

package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/ckks"
)

// DefaultParams is the CKKS parameter set used by every HE benchmark unless
// an individual runner overrides it. It is the analogue of
// `config/matmult_conf.yml` in the Python runner.
//
// The chain is one 55-bit prime followed by four 45-bit primes, giving a
// maximum multiplicative level of 4 — which is exactly what the Python
// driver passes (input_level=4) to run_thor_he.
var DefaultParams = ckks.ParametersLiteral{
	LogN:            13,                        // ring degree 2^13  →  2^12 = 4096 slots
	LogQ:            []int{55, 45, 45, 45, 45}, // 1 × 55-bit  +  4 × 45-bit
	LogP:            []int{61},
	LogDefaultScale: 45,
}

// HEContext bundles everything a runner needs to encode, encrypt, evaluate,
// and decrypt. It is created once by InitLattigo and shared between runs.
type HEContext struct {
	Params    ckks.Parameters
	SK        *rlwe.SecretKey
	PK        *rlwe.PublicKey
	RLK       *rlwe.RelinearizationKey
	KeyGen    *rlwe.KeyGenerator
	Encoder   *ckks.Encoder
	Encryptor *rlwe.Encryptor
	Decryptor *rlwe.Decryptor
	Evaluator *ckks.Evaluator // carries the relin key; Galois keys are added per-algorithm
	NHE       int             // number of usable slots (2^(LogN-1))
}

// InitLattigo instantiates CKKS with the given parameter literal and
// generates the scheme-wide keys. This is the Lattigo equivalent of the
// Python `orion.init_scheme(CONFIG)` call.
func InitLattigo(paramsLit ckks.ParametersLiteral) *HEContext {
	params, err := ckks.NewParametersFromLiteral(paramsLit)
	if err != nil {
		panic(fmt.Errorf("CKKS parameter instantiation: %w", err))
	}

	kgen := rlwe.NewKeyGenerator(params)

	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	m0 := TakeMemSnap()
	rlk := kgen.GenRelinearizationKeyNew(sk)
	m1 := TakeMemSnap()
	PrintMemDelta("Relinearization key", m0, m1)

	evk := rlwe.NewMemEvaluationKeySet(rlk) // Galois keys added per-algorithm

	nHE := params.MaxSlots()
	fmt.Printf("  [Lattigo] scheme initialised  logN=%d  n_he=%d\n",
		params.LogN(), nHE)

	return &HEContext{
		Params:    params,
		SK:        sk,
		PK:        pk,
		RLK:       rlk,
		KeyGen:    kgen,
		Encoder:   ckks.NewEncoder(params),
		Encryptor: rlwe.NewEncryptor(params, pk),
		Decryptor: rlwe.NewDecryptor(params, sk),
		Evaluator: ckks.NewEvaluator(params, evk),
		NHE:       nHE,
	}
}

// WithRotations returns a fresh Evaluator carrying Galois keys for the
// requested rotation set, in addition to the shared relinearization key.
// Call once per algorithm with its algorithm-specific rotation list.
func (ctx *HEContext) WithRotations(rotations []int) *ckks.Evaluator {
	galEls := make([]uint64, len(rotations))
	for i, k := range rotations {
		galEls[i] = ctx.Params.GaloisElement(k)
	}
	m0 := TakeMemSnap()
	gks := ctx.KeyGen.GenGaloisKeysNew(galEls, ctx.SK)
	m1 := TakeMemSnap()
	PrintMemDelta("Galois keys", m0, m1)

	m2 := TakeMemSnap()
	evk := rlwe.NewMemEvaluationKeySet(ctx.RLK, gks...)
	eval := ctx.Evaluator.WithKey(evk)
	m3 := TakeMemSnap()
	PrintMemDelta("Evaluation keyset + evaluator only", m2, m3)
	return eval
}
