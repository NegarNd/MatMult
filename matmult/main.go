package main

import (
	"flag"
	"fmt"
	"strings"
)

func main() {
	// ------------------------------------------------------------------
	// CLI flags
	// ------------------------------------------------------------------
	var (
		verify bool
		trials int
	)

	flag.BoolVar(&verify, "verify", false, "Check results against a plaintext reference.")
	flag.BoolVar(&verify, "v", false, "Alias for -verify.")
	flag.IntVar(&trials, "trials", 1, "Number of timed repetitions per configuration.")
	flag.IntVar(&trials, "t", 1, "Alias for -trials.")
	flag.Parse()

	var tags []string
	if verify {
		tags = append(tags, "verify ON")
	}
	if trials > 1 {
		tags = append(tags, fmt.Sprintf("%d trials", trials))
	}

	if len(tags) == 0 {
		fmt.Println("  [timing only]")
	} else {
		fmt.Printf("  [%s]\n", strings.Join(tags, ", "))
	}

	// ------------------------------------------------------------------
	// Run THOR ciphertext suite directly
	// ------------------------------------------------------------------
	fmt.Println()
	fmt.Println("Running THOR — HE suite...")
	fmt.Println()

	// Change this with:
	// ThorCiphertextSuite, MoaiCiphertextSuite, BMM1CiphertextSuite, BMM3CiphertextSuite, RowCiphertextSuite
	MoaiCiphertextSuite(verify, trials)

	fmt.Println()
	fmt.Println("Done.")
}
