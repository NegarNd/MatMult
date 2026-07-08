// package main

// import (
// 	"fmt"
// 	"runtime"
// )

// type MemSnap struct {
// 	Alloc      uint64
// 	TotalAlloc uint64
// 	Sys        uint64
// 	HeapAlloc  uint64
// 	HeapInuse  uint64
// 	NumGC      uint32
// }

// func TakeMemSnap() MemSnap {
// 	runtime.GC()
// 	var m runtime.MemStats
// 	runtime.ReadMemStats(&m)
// 	return MemSnap{
// 		Alloc:      m.Alloc,
// 		TotalAlloc: m.TotalAlloc,
// 		Sys:        m.Sys,
// 		HeapAlloc:  m.HeapAlloc,
// 		HeapInuse:  m.HeapInuse,
// 		NumGC:      m.NumGC,
// 	}
// }

// func PrintMemDelta(name string, before, after MemSnap) {
// 	fmt.Printf("\n[MEM] %s\n", name)
// 	fmt.Printf("  Live Alloc delta:  %.3f MB\n", float64(after.Alloc-before.Alloc)/1024/1024)
// 	fmt.Printf("  HeapAlloc delta:   %.3f MB\n", float64(after.HeapAlloc-before.HeapAlloc)/1024/1024)
// 	fmt.Printf("  HeapInuse delta:   %.3f MB\n", float64(after.HeapInuse-before.HeapInuse)/1024/1024)
// 	fmt.Printf("  TotalAlloc delta:  %.3f MB\n", float64(after.TotalAlloc-before.TotalAlloc)/1024/1024)
// 	fmt.Printf("  Sys delta:         %.3f MB\n", float64(after.Sys-before.Sys)/1024/1024)
// }

package main

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

type MemSnap struct {
	Alloc      uint64
	TotalAlloc uint64
	Sys        uint64
	HeapAlloc  uint64
	HeapInuse  uint64
	NumGC      uint32
}

func TakeMemSnap(runGC bool) MemSnap {
	if runGC {
		runtime.GC()
		runtime.GC()
	}

	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return MemSnap{
		Alloc:      m.Alloc,
		TotalAlloc: m.TotalAlloc,
		Sys:        m.Sys,
		HeapAlloc:  m.HeapAlloc,
		HeapInuse:  m.HeapInuse,
		NumGC:      m.NumGC,
	}
}

func PrintMemDelta(name string, before, after MemSnap) {
	fmt.Printf("\n[MEM] %s\n", name)
	fmt.Printf("  Live Alloc delta:  %.3f MB\n", float64(after.Alloc-before.Alloc)/1024/1024)
	fmt.Printf("  HeapAlloc delta:   %.3f MB\n", float64(after.HeapAlloc-before.HeapAlloc)/1024/1024)
	fmt.Printf("  HeapInuse delta:   %.3f MB\n", float64(after.HeapInuse-before.HeapInuse)/1024/1024)
	fmt.Printf("  TotalAlloc delta:  %.3f MB\n", float64(after.TotalAlloc-before.TotalAlloc)/1024/1024)
	fmt.Printf("  Sys delta:         %.3f MB\n", float64(after.Sys-before.Sys)/1024/1024)
	fmt.Printf("  NumGC delta:       %d\n", after.NumGC-before.NumGC)
}

type PeakMemMonitor struct {
	stop chan struct{}
	done chan struct{}

	startHeapAlloc uint64
	peakHeapAlloc  uint64
	peakHeapInuse  uint64
	peakSys        uint64

	mu sync.Mutex
}

func StartPeakMemMonitor(interval time.Duration) *PeakMemMonitor {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	mon := &PeakMemMonitor{
		stop:           make(chan struct{}),
		done:           make(chan struct{}),
		startHeapAlloc: m.HeapAlloc,
		peakHeapAlloc:  m.HeapAlloc,
		peakHeapInuse:  m.HeapInuse,
		peakSys:        m.Sys,
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		defer close(mon.done)

		for {
			select {
			case <-ticker.C:
				runtime.ReadMemStats(&m)

				mon.mu.Lock()
				if m.HeapAlloc > mon.peakHeapAlloc {
					mon.peakHeapAlloc = m.HeapAlloc
				}
				if m.HeapInuse > mon.peakHeapInuse {
					mon.peakHeapInuse = m.HeapInuse
				}
				if m.Sys > mon.peakSys {
					mon.peakSys = m.Sys
				}
				mon.mu.Unlock()

			case <-mon.stop:
				// final sample
				runtime.ReadMemStats(&m)

				mon.mu.Lock()
				if m.HeapAlloc > mon.peakHeapAlloc {
					mon.peakHeapAlloc = m.HeapAlloc
				}
				if m.HeapInuse > mon.peakHeapInuse {
					mon.peakHeapInuse = m.HeapInuse
				}
				if m.Sys > mon.peakSys {
					mon.peakSys = m.Sys
				}
				mon.mu.Unlock()

				return
			}
		}
	}()

	return mon
}

func (mon *PeakMemMonitor) Stop() {
	close(mon.stop)
	<-mon.done
}

func (mon *PeakMemMonitor) PrintPeak(name string, before MemSnap) {
	mon.mu.Lock()
	defer mon.mu.Unlock()

	fmt.Printf("\n[PEAK MEM] %s\n", name)
	fmt.Printf("  Peak HeapAlloc:       %.3f MB\n", float64(mon.peakHeapAlloc)/1024/1024)
	fmt.Printf("  Peak HeapAlloc delta: %.3f MB\n", float64(mon.peakHeapAlloc-before.HeapAlloc)/1024/1024)
	fmt.Printf("  Peak HeapInuse delta: %.3f MB\n", float64(mon.peakHeapInuse-before.HeapInuse)/1024/1024)
	fmt.Printf("  Peak Sys delta:       %.3f MB\n", float64(mon.peakSys-before.Sys)/1024/1024)
}
