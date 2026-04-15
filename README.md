# Brazil Weather Analysis Tool
### Parallel Computing — OpenCL 

A parallelised weather data analysis tool built with OpenCL for the Parallel Computing. Analyses over 21 years of air temperature data from five Brazilian weather stations using GPU-accelerated parallel algorithms.

---

## Overview

This tool loads historical temperature records from Brazil and performs statistical analysis entirely on parallel hardware using OpenCL kernels. All heavy computation runs on the GPU — the CPU only orchestrates the work.

**Stations covered:** Bernardo, Guarulhos, Santos, Cacoal, Tarauaca  
**Dataset size:** ~1 million readings (full) / ~18,000 readings (short)  
**Language:** C++ (host) + OpenCL C (kernels)

---

## Features

| Feature | Description | 
|---|---|
| Min / Max / Average | Parallel reduction across all readings |
| Standard deviation | Parallel sum of squared deviations |
| Histogram | Auto bin-width histogram with ASCII bar chart |
| Seasonal statistics | Per-month breakdown (Jan–Dec) with min/max/avg/stddev | 
| Kernel profiling | Execution time printed for every kernel launch |
| Menu interface | Interactive console menu matching the assignment brief |

---

## Project Structure

```
/
├── Assignment.cpp          # Main host program — OpenCL setup, menu, orchestration
├── my_kernels.cl           # All OpenCL GPU kernels
├── Weather.h               # Weather data class declaration
├── Weather.cpp             # Loads Brazil dataset into vectors
├── Utils.h                 # Lecturer-provided: platform listing, context, profiling
├── SerialStatistics.h      # Lecturer-provided: serial stats (median, quartiles)
├── SerialStatistics.cpp    # Lecturer-provided: serial stats implementation
├── brazil_temp_short.txt   # Short dataset (~18k readings) for development
└── README.md               # This file
```

> `Utils.h`, `SerialStatistics.h`, and `SerialStatistics.cpp` should not be modified.

---

## Kernels

| Kernel | File | Purpose |
|---|---|---|
| `reduce_add_4` | `my_kernels.cl` | Parallel tree reduction — computes the total sum |
| `reduce_STD_4` | `my_kernels.cl` | Parallel sum of `(x − mean)²` for standard deviation |
| `ParallelSelection` | `my_kernels.cl` | Parallel selection sort — gives min and max for free |
| `histogram_local` | `my_kernels.cl` | Histogram using local memory atomics |
| `atomicAddFloat` | `my_kernels.cl` | Helper: CAS-loop float atomic add (OpenCL 1.2 workaround) |

---

## Getting Started

### Prerequisites

- Visual Studio 2019 or later
- Intel OpenCL SDK **or** NVIDIA CUDA Toolkit (for GPU support)
- OpenCL-compatible device (Intel HD/UHD GPU, NVIDIA GPU, or CPU fallback)

### Setup in Visual Studio

1. Clone or download this repository
2. Open `Assignment.sln` in Visual Studio
3. Add all `.cpp`, `.h`, and `.cl` files to the project
4. Configure include and library paths (see note below)
5. Place `brazil_temp_short.txt` in the same directory as the compiled `.exe`
6. Build and run

### Switching to the full dataset

In `Assignment.cpp` line ~78, change:

```cpp
Data.Load("brazil_temp_short.txt");   // development
```
to:
```cpp
Data.Load("Brazil_air_temp.txt");     // full 1M readings
```

---

## Usage

```
============================================
  Brazil Weather Analysis Tool
============================================
  1 : List platforms and devices
  2 : Run statistical analysis (parallel)
  3 : Exit
============================================
>
```

**Option 1** lists every OpenCL platform and device found on your machine.  
**Option 2** runs the full analysis pipeline:
- Sorts the dataset in parallel
- Computes min, max, mean, standard deviation, median, Q1, Q3
- Prints kernel execution times in nanoseconds
- Prompts for number of histogram bins, then prints an ASCII bar chart
- Prints seasonal statistics for all 12 months

**Command-line flags** (optional):

```bash
Assignment.exe -p 0 -d 0    # select platform 0, device 0
Assignment.exe -l            # list all platforms and devices
Assignment.exe -h            # print help
```

---

## Sample Output

```
Running on: Intel(R) OpenCL HD Graphics | Intel(R) UHD Graphics

Loading: brazil_temp_short.txt
Loaded 18732 readings from brazil_temp_short.txt

============================================
  Brazil Weather Analysis Tool
============================================
> 2

Dataset size: 18732 readings

--- Sort ---
Sort write [ns]: 48200
Sort read  [ns]: 31100
Sort kernel time [ns]: 245000

--- Sum / Average ---
Sum kernel time [ns]: 18400

========== OVERALL PARALLEL RESULTS ==========
  Total readings : 18732
  Min            : 1.00 C
  Max            : 40.00 C
  Mean (avg)     : 19.43 C
  Std deviation  :  8.21 C
  Median         : 20.00 C
  Q1             : 12.00 C
  Q3             : 27.00 C
  Total kernel time [ns]: 312800
==============================================

Enter number of histogram bins: 10

Histogram: 10 bins, bin width = 3.90 C
--------------------------------------------------------------
Bin range (C)         Count     Bar
--------------------------------------------------------------
[  1.0 to   4.9]       823  ##########
[  4.9 to   8.8]      1204  ###############
...

========== SEASONAL STATISTICS (by month) ==========
Month        Count    Min (C)   Max (C)   Avg (C)    StdDev
------------------------------------------------------------
January       1821    2.00      39.00     21.34       7.80
February      1654    1.00      40.00     22.01       8.12
...
```

---

## Parallel Patterns Used

### Tree Reduction (sum, std dev)
Each work-group loads data into local memory, then halves the active work-items each round until one holds the group total. Groups atomically accumulate into a single output value.

```
Round 0:  [a][b][c][d][e][f][g][h]   (8 work-items load)
Round 1:  [a+b]   [c+d]   [e+f]   [g+h]
Round 2:  [a+b+c+d]       [e+f+g+h]
Round 3:  [a+b+c+d+e+f+g+h]          → atomicAdd to B[0]
```

### Parallel Selection Sort
Every work-item counts how many elements are smaller than it — that count is its correct sorted position. All work-items do this simultaneously, sorting the entire array in a single kernel launch.

### Local Memory Histogram
Each work-group maintains a private histogram in fast local memory. After all work-items have voted, the local histogram is merged into global memory once per group — minimising slow global atomic operations.

---

## Notes

> **If your `<CL/cl.hpp>` says "Could not open source file"**, follow these steps:
>
> 1. Right-click your project → **Properties**
> 2. Go to **C/C++ → General → Additional Include Directories**
> 3. Add: `$(INTELOCLSDKROOT)include;%(AdditionalIncludeDirectories)`

> **Linker error `LNK2019 unresolved external symbol`?**
>
> 1. Right-click project → **Properties**
> 2. **Linker → General → Additional Library Directories** → add `$(INTELOCLSDKROOT)lib\x64` or replace entirely with `$(INTELOCLSDKROOT)lib\x64;%(AdditionalLibraryDirectories)`
> 3. **Linker → Input → Additional Dependencies** → add `OpenCL.lib` or replace entirely with `OpenCL.lib;%(AdditionalDependencies)`

---

## Acknowledgements

Sample code structure and `Utils.h`, `SerialStatistics.h/.cpp` are provided as workshop material. All parallel kernel implementations and the Brazil-specific host program are original work for this assignment.

---

## License

This project is submitted as assessed coursework. Do not copy or reuse without permission — academic misconduct policies apply.
