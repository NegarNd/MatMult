// bmm3_plain.go
//
// BMM-III helpers used by the HE kernel in bmm3_cipher.go. These mirror
// the public helpers exposed by plaintext/bmm3_plain.py that the Python
// cipher code imports:
//
//     from plaintext.bmm3_plain import (
//         bicyclic_encode, bicyclic_decode, smallest_r, break_into_chunks,
//     )
//
// The first two (BicyclicEncode, BicyclicDecode) already live in
// bmm1_plain.go and are reused here — no duplication. This file adds only
// what BMM-III needs on top.
//
// There is no plaintext runner for BMM-III in this port (HE-only). If you
// want a plaintext reference later, add a long_rot simulator and kernel
// here and a RunBmm3Plaintext entry in bmm3_runner.go.

package main

import "math"

// smallestR finds the smallest r ≥ 1 such that (r·m − n) is divisible by
// p AND ≥ 0. Port of plaintext/bmm3_plain.py → smallest_r.
//
// This is the constant the BMM-III algorithm uses to compute the B-side
// rotation amount: rot_b = ((r·m − n) · i) mod (m·p).
func smallestR(n, m, p int) int {
	r, s := 1, m-n
	for (s%p != 0) || s < 0 {
		r++
		s += m
	}
	return r
}

// breakIntoChunks cyclically tiles `enc` to a length that supports the
// worst-case rotation window LongRot may need, then splits into
// ceil(len/nHE) chunks of nHE slots each. The last chunk is zero-padded
// on the right if it falls short.
//
// Port of plaintext/bmm3_plain.py → break_into_chunks. Returns the raw
// float64 chunks; encryption is the caller's responsibility.
//
// The cyclic extension rule matches the Python:
//
//	needed = outputLen + nHE          // worst-case window
//	if enc_len < needed:
//	    reps = ceil(needed / enc_len) + 1
//	    enc  = tile(enc[:enc_len], reps)
//
// Returns `w = ceil(len(enc) / nHE)` chunks of length nHE each.
func breakIntoChunks(enc []float64, encLen, outputLen, nHE int) [][]float64 {
	// Extend cyclically if the raw encoding is shorter than the window
	// LongRot may read.
	needed := outputLen + nHE
	work := enc[:encLen]
	if encLen < needed {
		reps := int(math.Ceil(float64(needed)/float64(encLen))) + 1
		tiled := make([]float64, 0, reps*encLen)
		for r := 0; r < reps; r++ {
			tiled = append(tiled, work...)
		}
		work = tiled
	}

	w := intCeilDiv(len(work), nHE)
	chunks := make([][]float64, w)
	for i := 0; i < w; i++ {
		chunk := make([]float64, nHE)
		start := i * nHE
		end := start + nHE
		if end > len(work) {
			end = len(work)
		}
		copy(chunk, work[start:end])
		chunks[i] = chunk
	}
	return chunks
}
