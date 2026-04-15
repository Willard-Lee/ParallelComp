// ============================================================
// CAT3024N Parallel Computing – Brazil Weather Analysis Tool
// File: my_kernels.cl
//
// Kernels in this file:
//   atomicAddFloat()    – helper: atomic add for floats
//   reduce_add_4()      – parallel sum (tree reduction)
//   reduce_STD_4()      – parallel sum of squared deviations
//   ParallelSelection() – parallel selection sort
//   histogram_local()   – histogram using local memory atomics
// ============================================================


// ============================================================
// atomicAddFloat – atomic add for floats
//
// WHY THIS EXISTS:
//   OpenCL 1.2 only provides atomic_add for integers.
//   For floats we use a compare-and-swap (CAS) loop:
//   read the current value, compute the new value, try to
//   write it atomically. If another work-item changed it
//   in between, retry. This is the standard GPU trick.
//
// DEMO TIP: explain that without this, multiple work-groups
//   writing to B[0] simultaneously would produce a race
//   condition and a wrong answer.
// ============================================================
inline void atomicAddFloat(volatile __global float* addr, float val)
{
    union {
        unsigned int u32;
        float        f32;
    } next, expected, current;

    current.f32 = *addr;

    do {
        expected.f32 = current.f32;
        next.f32     = expected.f32 + val; // the value we want to write
        // atomic_cmpxchg: only writes next if *addr still equals expected
        current.u32  = atomic_cmpxchg(
            (volatile __global unsigned int*)addr,
            expected.u32,
            next.u32
        );
    } while (current.u32 != expected.u32); // retry if another thread beat us
}


// ============================================================
// reduce_add_4 – parallel sum via tree reduction
//
// HOW IT WORKS (for a group of N work-items):
//   Round 1: items 0,2,4,… add their right neighbour → N/2 sums
//   Round 2: items 0,4,8,… add their right neighbour → N/4 sums
//   …continues until item 0 holds the group total.
//   Item 0 then atomically adds that total to B[0].
//   All groups do this simultaneously; B[0] ends up as the
//   grand total across the entire array.
//
// ARGS:
//   A       – input temperature array (global memory)
//   B       – output accumulator; B[0] = final sum (global)
//   scratch – fast per-group workspace   (local memory)
// ============================================================
__kernel void reduce_add_4(__global const float* A,
                           __global       float* B,
                           __local        float* scratch)
{
    int id  = get_global_id(0);  // this work-item's global index
    int lid = get_local_id(0);   // this work-item's index within its group
    int N   = get_local_size(0); // number of work-items in this group

    // Phase 1: copy from slow global memory -> fast local memory
    scratch[lid] = A[id];
    barrier(CLK_LOCAL_MEM_FENCE); // WAIT – all items must finish loading
                                  // before anyone reads a neighbour

    // Phase 2: tree reduction – active items halve each round
    for (int i = 1; i < N; i *= 2)
    {
        // Only items at positions 0, 2i, 4i, … are active this round
        if (!(lid % (i * 2)) && ((lid + i) < N))
            scratch[lid] += scratch[lid + i]; // accumulate neighbour

        barrier(CLK_LOCAL_MEM_FENCE); // WAIT after each round
    }

    // Phase 3: item 0 writes this group's partial sum to global B[0]
    // atomic needed because multiple groups all write to the same B[0]
    if (!lid)
        atomicAddFloat(&B[0], scratch[lid]);
}


// ============================================================
// reduce_STD_4 – parallel sum of (x - mean)^2
//
// Same tree reduction pattern as reduce_add_4, but each
// element is transformed to (x - mean)^2 before accumulating.
// The host then divides by N and takes the square root.
//
// ARGS:
//   A       – input temperature array (global)
//   B       – output: sum of squared deviations at B[0] (global)
//   scratch – per-group workspace (local)
//   Mean    – the mean temperature (pre-computed on host)
//   Padd    – number of padding elements to ignore
// ============================================================
__kernel void reduce_STD_4(__global const float* A,
                           __global       float* B,
                           __local        float* scratch,
                           float Mean,
                           int   Padd)
{
    int id  = get_global_id(0);
    int lid = get_local_id(0);
    int N   = get_local_size(0);
    int NG  = get_global_size(0); // total work-items launched

    // Load: ignore padded elements (treat them as 0 contribution)
    if (id < (NG - Padd))
    {
        scratch[lid] = A[id];
        // Transform: (x - mean)^2 is what we want to sum
        scratch[lid] = (scratch[lid] - Mean) * (scratch[lid] - Mean);
    }
    else
    {
        scratch[lid] = 0.0f; // padded element contributes nothing
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Tree reduction (identical structure to reduce_add_4)
    for (int i = 1; i < N; i *= 2)
    {
        if (!(lid % (i * 2)) && ((lid + i) < N))
            scratch[lid] += scratch[lid + i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!lid)
        atomicAddFloat(&B[0], scratch[lid]);
}


// ============================================================
// ParallelSelection – parallel selection sort
//
// HOW IT WORKS:
//   Each work-item i counts how many elements are smaller
//   than A[i]. That count is exactly the correct sorted
//   position for A[i]. All work-items do this simultaneously,
//   so the entire array is sorted in one kernel launch.
//
// COMPLEXITY:
//   Work = O(n^2)  – each of n items scans all n items
//   Span = O(n)    – all items scan in parallel
//   This is fine for this dataset size. For a 1M dataset you
//   would want bitonic sort (O(n log^2 n) work) – worth
//   mentioning in your report as a potential optimisation.
//
// ARGS:
//   A – input array (global, read-only)
//   B – sorted output array (global, write-only)
// ============================================================
__kernel void ParallelSelection(__global const float* A,
                                __global       float* B)
{
    int   i    = get_global_id(0);  // I am element i
    int   n    = get_global_size(0);
    float iKey = A[i];              // my value
    int   pos  = 0;                 // my rank (sorted position)

    // Count elements strictly smaller than me, or equal with lower index
    // The tie-break (j < i) ensures equal elements get distinct positions
    for (int j = 0; j < n; j++)
    {
        float jKey = A[j];
        bool smaller = (jKey < iKey) || (jKey == iKey && j < i);
        pos += smaller ? 1 : 0;
    }

    B[pos] = A[i]; // place myself at the correct sorted position
}


// ============================================================
// histogram_local – parallel histogram using local atomics
//
// HOW IT WORKS:
//   Each work-group maintains its own private copy of the
//   histogram in fast local memory. Every work-item atomically
//   increments the correct local bin for its temperature value.
//   After all items have voted, work-item 0..numBins-1 merge
//   the local histogram into the global histogram using
//   global atomics. This is much faster than each work-item
//   hitting global memory directly.
//
// ARGS:
//   A          – input temperature array (global)
//   globalHist – output bin counts (global, accumulated)
//   localHist  – per-group bin counts (local scratch)
//   minVal     – minimum temperature (= Values[0] after sort)
//   binWidth   – (max - min) / numBins  (computed on host)
//   numBins    – number of histogram bins
// ============================================================
__kernel void histogram_local(__global const float* A,
                              __global       int*   globalHist,
                              __local        int*   localHist,
                              float minVal,
                              float binWidth,
                              int   numBins)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int gs  = get_global_size(0); // may be padded beyond actual data

    // Step 1: zero out this group's local histogram
    // Each of the first numBins work-items zeros one bin
    if (lid < numBins)
        localHist[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE); // wait until all bins are zeroed

    // Step 2: each work-item increments the correct local bin
    // Guard: only process real elements (not padding)
    if (gid < gs)
    {
        // Compute which bin this temperature falls into
        int bin = (int)((A[gid] - minVal) / binWidth);

        // Clamp to valid range – the last temperature may land on MAX exactly
        if (bin < 0)       bin = 0;
        if (bin >= numBins) bin = numBins - 1;

        // atomic_add on local memory – fast, no cross-group contention
        atomic_add(&localHist[bin], 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE); // wait until all items have voted

    // Step 3: merge local histogram into global histogram
    // Only the first numBins work-items participate
    if (lid < numBins)
        atomic_add(&globalHist[lid], localHist[lid]);
}
