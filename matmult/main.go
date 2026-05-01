// main.go
//
// Matrix-Multiplication Test Runner (Go port of run.py).
//
// Usage:
//   go run .                         # timing only, 1 trial
//   go run . -verify                 # check correctness
//   go run . -trials 5               # mean ± std over 5 runs
//   go run . -verify -trials 5       # both
//
// The CKKS scheme is initialised once per HE suite (see InitLattigo in
// lattigo_init.go) and the resulting HEContext is reused across all
// configurations within that suite — this mirrors the role of
// init_orion() in run.py.

package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
)

// Runner is the shared signature of every menu handler. `verify` toggles
// correctness checks against a plaintext reference; `nTrials` is the
// number of timed repetitions per configuration.
type Runner func(verify bool, nTrials int)

// options maps menu entries to their handlers. Keep the keys in sync with
// the strings printed in `menu` below.
var options = map[string]Runner{
	"1": ThorPlaintextSuite,
	"2": ThorCiphertextSuite,
	"3": MoaiPlaintextSuite,
	"4": MoaiCiphertextSuite,
	"5": BMM1PlaintextSuite,
	"6": BMM1CiphertextSuite,
	// "7":  BMM3PlaintextSuite,
	"8":  BMM3CiphertextSuite,
	"9":  RowPlaintextSuite,
	"10": RowCiphertextSuite,
	// "5":  ArionPlaintextSuite,
	// "6":  ArionCiphertextSuite,
}

const menu = `
============================================================
  Matrix Multiplication — Diagonal / Bicyclic Encoding
============================================================
  --- THOR (Moon et al., CCS 2025) ---
  1.  THOR — plaintext
  2.  THOR — HE

  --- MOAI ---
  3.  MOAI — plaintext
  4.  MOAI — HE

  --- ARION ---
  5.  ARION — plaintext                        [TODO]
  6.  ARION — HE                               [TODO]

  --- Row Packing ---
  7.  Row packing — plaintext                  [TODO]
  8.  Row packing — HE                         [TODO]

  --- BMM-I (Zheng et al., IEEE TIFS 2024) ---
  9.  BMM-I — plaintext                        [TODO]
  10. BMM-I — HE                               [TODO]

  --- Bicycle BMM-III (Zheng et al., IEEE TIFS 2024) ---
  11. BMM-III — plaintext                      [TODO]
  12. BMM-III — HE                             [TODO]

  0.  Exit
============================================================`

func main() {
	// ------------------------------------------------------------------
	// CLI flags (mirror run.py's argparse setup)
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
	// Interactive menu loop
	// ------------------------------------------------------------------
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Println(menu)
		fmt.Print("Select option: ")

		line, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println("\nExiting.")
			return
		}
		choice := strings.TrimSpace(line)

		if choice == "0" {
			fmt.Println("Exiting.")
			return
		}

		handler, ok := options[choice]
		if !ok {
			fmt.Printf("  Invalid option '%s'. Please try again.\n", choice)
			continue
		}

		fmt.Println()
		handler(verify, trials)
	}
}
