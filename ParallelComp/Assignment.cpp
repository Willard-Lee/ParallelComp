// ============================================================
// CAT3024N Parallel Computing – Brazil Weather Analysis Tool
// File: Assignment.cpp
//
// What this file does:
//   main()          – OpenCL setup, menu loop
//   Parallel()      – orchestrates all parallel computations
//   SumVec()        – parallel sum (-> used for average)
//   STDVec()        – parallel standard deviation
//   Sort()          – parallel selection sort (min/max come free)
//   Histogram()     – parallel histogram  [2nd class feature]
//   SeasonalStats() – per-month breakdown [1st class feature]
//   AddPadding()    – pads vector to multiple of local_size
//   KernelExec()    – launches a kernel, overwrites input vector
//   KernelExecRet() – launches a kernel, returns B[0]
// ============================================================

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cstdlib>

// Lecturer-provided helpers – do not modify these files
#include "Utils.h"
#include "Weather.h"
#include "SerialStatistics.h"

typedef float myType; // change to double here if you want higher precision

// ============================================================
// Forward declarations
// ============================================================
void Parallel(std::vector<float>& Values,
    std::vector<int>& Months,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event);

myType SumVec(std::vector<myType>& temp,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event);

myType STDVec(std::vector<myType>& temp, myType Mean,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event);

void Sort(std::vector<myType>& temp,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event);

void Histogram(std::vector<myType>& sortedTemp,
    float MIN, float MAX, int numBins,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event);

void SeasonalStats(std::vector<float>& Values,
    std::vector<int>& Months,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event);

int AddPadding(std::vector<myType>& temp, size_t LocalSize, float PadVal);

void KernelExec(cl::Kernel kernel, std::vector<myType>& temp,
    size_t Local_Size, cl::Context context,
    cl::CommandQueue queue,
    bool Two, bool Three, bool Four,
    float FThree, int IFour,
    cl::Event& prof_event, std::string Name);

float KernelExecRet(cl::Kernel kernel, std::vector<myType>& temp,
    size_t Local_Size, cl::Context context,
    cl::CommandQueue queue,
    bool Two, bool Three, bool Four,
    float FThree, int IFour,
    cl::Event& prof_event, std::string Name);

// ============================================================
// print_help
// ============================================================
void print_help()
{
    std::cout << "Brazil Weather Analysis Tool – usage:" << std::endl;
    std::cout << "  -p <id> : select platform (default 0)" << std::endl;
    std::cout << "  -d <id> : select device   (default 0)" << std::endl;
    std::cout << "  -l      : list all platforms and devices" << std::endl;
    std::cout << "  -h      : print this message" << std::endl;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv)
{
    // ----------------------------------------------------------
    // 1. Load the Brazil dataset
    //    Weather class reads: StationName Year Month Day Time Temp
    // ----------------------------------------------------------
    Weather Data = Weather();

    // Use the short dataset for development/testing.
    // Swap to "Brazil_air_temp.txt" for the full 1M-reading run.
    Data.Load("brazil_temp_short.txt");

    // Pull the vectors we need out of the Weather object
    std::vector<float>& temp = Data.GetTemp();   // temperature values
    std::vector<int>& months = Data.GetMonth();  // month of each reading

    // ----------------------------------------------------------
    // 2. Handle command-line flags (-p, -d, -l, -h)
    // ----------------------------------------------------------
    int platform_id = 0;
    int device_id = 0;

    for (int i = 1; i < argc; i++)
    {
        if ((strcmp(argv[i], "-p") == 0) && (i < argc - 1)) { platform_id = atoi(argv[++i]); }
        else if ((strcmp(argv[i], "-d") == 0) && (i < argc - 1)) { device_id = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
        else if (strcmp(argv[i], "-h") == 0) { print_help(); }
    }

    // ----------------------------------------------------------
    // 3. Initialise OpenCL: context -> queue -> program
    // ----------------------------------------------------------
    try
    {
        // GetContext() (from Utils.h) finds the chosen platform/device
        cl::Context context = GetContext(platform_id, device_id);

        std::cout << "Running on: " << GetPlatformName(platform_id)
            << " | " << GetDeviceName(platform_id, device_id)
            << std::endl << std::endl;

        // CommandQueue with profiling so we can measure kernel times
        cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

        // Load and compile the kernel source file at runtime
        cl::Program::Sources sources;
        AddSources(sources, "my_kernels.cl");
        cl::Program program(context, sources);

        try
        {
            program.build();
        }
        catch (const cl::Error& err)
        {
            std::cout << "Build Status: "
                << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(
                    context.getInfo<CL_CONTEXT_DEVICES>()[0])
                << std::endl;
            std::cout << "Build Log:\n"
                << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                    context.getInfo<CL_CONTEXT_DEVICES>()[0])
                << std::endl;
            throw err;
        }

        cl::Event prof_event;

        // ----------------------------------------------------------
        // 4. Menu loop (matches assignment brief Figure 1)
        // ----------------------------------------------------------
        int choice = 0;
        do
        {
            std::cout << "\n============================================" << std::endl;
            std::cout << "  Brazil Weather Analysis Tool" << std::endl;
            std::cout << "============================================" << std::endl;
            std::cout << "  1 : List platforms and devices" << std::endl;
            std::cout << "  2 : Run statistical analysis (parallel)" << std::endl;
            std::cout << "  3 : Exit" << std::endl;
            std::cout << "============================================" << std::endl;
            std::cout << "> ";
            std::cin >> choice;
            std::cout << std::endl;

            switch (choice)
            {
            case 1:
                std::cout << ListPlatformsDevices() << std::endl;
                break;

            case 2:
                // Pass a COPY of temp – Sort() will rearrange it,
                // and we still need the original ordering for
                // the Months pairing in SeasonalStats.
            {
                std::vector<float> tempCopy = temp;
                Parallel(tempCopy, months, context, queue, program, prof_event);
            }
            break;

            case 3:
                std::cout << "Exiting..." << std::endl;
                break;

            default:
                std::cout << "Invalid option. Please enter 1, 2 or 3." << std::endl;
                break;
            }

        } while (choice != 3);
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: " << err.what() << ", "
            << getErrorString(err.err()) << std::endl;
    }

    std::cout << "\nPress Enter to terminate...";
    std::cin.ignore();
    std::cin.get();
    return 0;
}

// ============================================================
// Parallel – orchestrates all parallel analyses
// ============================================================
void Parallel(std::vector<float>& Values,
    std::vector<int>& Months,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event)
{
    int originalSize = (int)Values.size();

    std::cout << "Dataset size: " << originalSize << " readings" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    // ----------------------------------------------------------
    // Step 1: Sort ascending – gives min and max for free
    // ----------------------------------------------------------
    std::cout << "\n--- Selection Sort ---" << std::endl;
    Sort(Values, context, queue, program, prof_event);

    unsigned long sortTime = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>()
        - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    std::cout << "Sort kernel time [ns]: " << sortTime << std::endl;
    std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << "\n";

    float MIN = Values[0];
    float MAX = Values[Values.size() - 1];

    // ----------------------------------------------------------
    // Step 2: Parallel sum -> average
    // ----------------------------------------------------------
    std::cout << "\n--- Sum / Average ---" << std::endl;
    float Sum = SumVec(Values, context, queue, program, prof_event);
    float Mean = Sum / (float)originalSize;

    unsigned long sumTime = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>()
        - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    std::cout << "Sum kernel time [ns]: " << sumTime << std::endl;
    std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << "\n";

    // ----------------------------------------------------------
    // Step 3: Parallel standard deviation
    // ----------------------------------------------------------
    std::cout << "\n--- Standard Deviation ---" << std::endl;
    float SD = STDVec(Values, Mean, context, queue, program, prof_event);

    unsigned long stdTime = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>()
        - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    std::cout << "StdDev kernel time [ns]: " << stdTime << std::endl;
    std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << "\n";

    // Median and quartiles from the sorted array (host-side, cheap)
    SerialStatistics SStats;
    float Median = SStats.GetMedianValue(Values);
    float Q1 = SStats.FirstQuartile(Values);
    float Q3 = SStats.ThirdQuartile(Values);

    // ----------------------------------------------------------
    // Step 4: Print overall results
    // ----------------------------------------------------------
    std::cout << "\n========== OVERALL PARALLEL RESULTS ==========" << std::endl;
    std::cout << "  Total readings : " << originalSize << std::endl;
    std::cout << "  Min            : " << MIN << " C" << std::endl;
    std::cout << "  Max            : " << MAX << " C" << std::endl;
    std::cout << "  Mean (avg)     : " << Mean << " C" << std::endl;
    std::cout << "  Std deviation  : " << SD << " C" << std::endl;
    std::cout << "  Median         : " << Median << " C" << std::endl;
    std::cout << "  Q1             : " << Q1 << " C" << std::endl;
    std::cout << "  Q3             : " << Q3 << " C" << std::endl;
    std::cout << "  Total kernel time [ns]: "
        << sortTime + sumTime + stdTime << std::endl;
    std::cout << "==============================================" << std::endl;

    // ----------------------------------------------------------
    // Step 5: Histogram  
    //   Bin width is auto-computed: (MAX - MIN) / numBins
    // ----------------------------------------------------------
    std::cout << "\n--- Histogram ---" << std::endl;
    std::cout << "Enter number of histogram bins: ";
    int numBins;
    std::cin >> numBins;
    if (numBins < 1) numBins = 10;

    Histogram(Values, MIN, MAX, numBins, context, queue, program, prof_event);

    // ----------------------------------------------------------
    // Step 6: Seasonal statistics 
    // ----------------------------------------------------------
    SeasonalStats(Values, Months, context, queue, program, prof_event);
}

// ============================================================
// SumVec – parallel tree reduction sum
//   Works on an internal copy so the caller's vector is
//   never modified by padding.
// ============================================================
myType SumVec(std::vector<myType>& temp,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event)
{
    std::vector<myType> work = temp; // copy – never touch the original
    size_t local_size = 256;
    AddPadding(work, local_size, 0.0f); // pad with 0 – neutral for addition

    cl::Kernel kernel = cl::Kernel(program, "reduce_add_4");
    return KernelExecRet(kernel, work, local_size,
        context, queue,
        true, false, false, 0.0f, 0,
        prof_event, "Sum");
}

// ============================================================
// ============================================================
// STDVec – parallel standard deviation
//   Works on an internal copy so the caller's vector is
//   never modified by padding.
// ============================================================
myType STDVec(std::vector<myType>& temp, myType Mean,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event)
{
    std::vector<myType> work = temp; // copy – never touch the original
    size_t local_size = 256;
    int true_size = (int)work.size(); // save before padding grows it
    int padding_size = AddPadding(work, local_size, 0.0f);

    cl::Kernel kernel = cl::Kernel(program, "reduce_STD_4");
    // arg3 = Mean, arg4 = padding_size (kernel uses this to skip padded elements)
    float result = KernelExecRet(kernel, work, local_size,
        context, queue,
        true, true, true,
        Mean, padding_size,
        prof_event, "StdDev");

    // result = sum of (x-mean)^2 — divide by N then sqrt
    result = result / (float)true_size;
    return sqrt(result);
}

// ============================================================
// Sort – parallel selection sort
//   Works on an internal padded copy, then resizes temp to
//   exactly the sorted real elements (no padding leftover).
//   After return: temp[0]=MIN, temp[last]=MAX
// ============================================================
void Sort(std::vector<myType>& temp,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event)
{
    int originalSize = (int)temp.size(); // save true count before padding
    size_t local_size = 32;
    int padding_size = AddPadding(temp, local_size, -1000000.0f);

    cl::Kernel kernel = cl::Kernel(program, "ParallelSelection");
    KernelExec(kernel, temp, local_size,
        context, queue,
        false, false, false, 0.0f, 0,
        prof_event, "Sort");

    // Remove padding elements (they sort to the front as -1000000)
    // then trim to exactly originalSize so no ghost elements remain
    if (padding_size)
        temp.erase(temp.begin(), temp.begin() + (local_size - padding_size));

    // Safety trim – guarantee vector is exactly originalSize
    if ((int)temp.size() > originalSize)
        temp.resize(originalSize);
}

// ============================================================
// Histogram – parallel histogram using local memory atomics
// ============================================================
void Histogram(std::vector<myType>& sortedTemp,
    float MIN, float MAX, int numBins,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event)
{
    // Auto-compute bin width from data range
    float binWidth = (MAX - MIN) / (float)numBins;

    std::cout << "\nHistogram: " << numBins << " bins, "
        << "bin width = " << std::fixed << std::setprecision(2)
        << binWidth << " C" << std::endl;

    size_t input_elements = sortedTemp.size();
    size_t input_size = input_elements * sizeof(myType);

    std::vector<int> h_hist(numBins, 0);
    size_t hist_size = numBins * sizeof(int);

    // Device buffers
    cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
    cl::Buffer buffer_Hist(context, CL_MEM_READ_WRITE, hist_size);

    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size,
        &sortedTemp[0], NULL, &prof_event);
    queue.enqueueFillBuffer(buffer_Hist, 0, 0, hist_size);

    cl::Kernel kernel = cl::Kernel(program, "histogram_local");
    kernel.setArg(0, buffer_A);
    kernel.setArg(1, buffer_Hist);
    kernel.setArg(2, cl::Local(numBins * sizeof(int))); // local scratch bins
    kernel.setArg(3, MIN);
    kernel.setArg(4, binWidth);
    kernel.setArg(5, numBins);

    // Pad global size to a multiple of local_size
    size_t local_size = 64;
    size_t padded_size = ((input_elements + local_size - 1) / local_size) * local_size;

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
        cl::NDRange(padded_size),
        cl::NDRange(local_size),
        NULL, &prof_event);
    queue.finish();

    queue.enqueueReadBuffer(buffer_Hist, CL_TRUE, 0, hist_size, &h_hist[0]);

    unsigned long histTime = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>()
        - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    std::cout << "Histogram kernel time [ns]: " << histTime << std::endl;
    std::cout << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << "\n";

    // ── Print ASCII bar chart to console (quick reference) ───────
    int maxCount = *std::max_element(h_hist.begin(), h_hist.end());

    std::cout << "\n" << std::left
        << std::setw(22) << "Bin range (C)"
        << std::setw(10) << "Count"
        << "Bar" << std::endl;
    std::cout << std::string(62, '-') << std::endl;

    for (int i = 0; i < numBins; i++)
    {
        float binLow = MIN + i * binWidth;
        float binHigh = binLow + binWidth;
        int barLen = (maxCount > 0) ? (int)(30.0f * h_hist[i] / maxCount) : 0;
        std::string bar(barLen, '#');

        std::cout << std::fixed << std::setprecision(1)
            << "[" << std::setw(6) << binLow
            << " to " << std::setw(6) << binHigh << "]  "
            << std::setw(8) << h_hist[i] << "  "
            << bar << std::endl;
    }
    std::cout << std::string(62, '-') << std::endl;

    // ── Export histogram data to text file ─────────────────────────
    // Format per line: bin_low  bin_high  count
    // Read by plot_histogram.py to produce a matplotlib bar chart.
    std::ofstream outFile("histogram_data.txt");
    if (!outFile.is_open())
    {
        std::cerr << "Warning: could not write histogram_data.txt" << std::endl;
        return;
    }

    // Header comment – stores parameters for the Python script
    outFile << "# Brazil Weather Histogram\n";
    outFile << "# bins=" << numBins
        << " min=" << std::fixed << std::setprecision(2) << MIN
        << " max=" << MAX
        << " binwidth=" << binWidth << "\n";

    // Data rows: low_edge  high_edge  count
    outFile << std::fixed << std::setprecision(4);
    for (int i = 0; i < numBins; i++)
    {
        float binLow = MIN + i * binWidth;
        float binHigh = binLow + binWidth;
        outFile << binLow << " " << binHigh << " " << h_hist[i] << "\n";
    }
    outFile.close();
    std::cout << "\nHistogram data saved to histogram_data.txt" << std::endl;

    // ── Launch Python plot automatically ───────────────────────────
    // system() runs plot_histogram.py in the same folder as the .exe
    std::cout << "Launching plot_histogram.py..." << std::endl;
    int ret = system("python plot_histogram.py");
    if (ret != 0)
        std::cerr << "Note: Python plot failed. "
        << "Run plot_histogram.py manually." << std::endl;
}

// ============================================================
// SeasonalStats – per-month parallel statistics
//   Filters by month on the host, reuses Sort + SumVec + STDVec.
//   No new kernels needed.
// ============================================================
void SeasonalStats(std::vector<float>& Values,
    std::vector<int>& Months,
    cl::Context context, cl::CommandQueue queue,
    cl::Program program, cl::Event& prof_event)
{
    std::cout << "\n========== SEASONAL STATISTICS (by month) ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    const std::string monthNames[] = {
        "", "January", "February", "March",    "April",
        "May",         "June",     "July",      "August",
        "September",   "October",  "November",  "December"
    };

    std::cout << std::left
        << std::setw(12) << "Month"
        << std::setw(8) << "Count"
        << std::setw(10) << "Min (C)"
        << std::setw(10) << "Max (C)"
        << std::setw(10) << "Avg (C)"
        << std::setw(10) << "StdDev"
        << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    for (int m = 1; m <= 12; m++)
    {
        // Collect all temperatures that belong to month m
        // Months[i] and Values[i] are paired – same reading index
        std::vector<float> monthTemps;
        for (int i = 0; i < (int)Values.size(); i++)
        {
            if (Months[i] == m)
                monthTemps.push_back(Values[i]);
        }

        if (monthTemps.empty())
        {
            std::cout << std::setw(12) << monthNames[m] << "  no data" << std::endl;
            continue;
        }

        int count = (int)monthTemps.size();

        // Sort this month's subset -> gives min and max
        Sort(monthTemps, context, queue, program, prof_event);
        float monthMin = monthTemps[0];
        float monthMax = monthTemps[monthTemps.size() - 1];

        // Parallel sum -> average
        float sum = SumVec(monthTemps, context, queue, program, prof_event);
        float mean = sum / (float)count;

        // Parallel standard deviation
        float sd = STDVec(monthTemps, mean, context, queue, program, prof_event);

        std::cout << std::setw(12) << monthNames[m]
            << std::setw(8) << count
            << std::setw(10) << monthMin
            << std::setw(10) << monthMax
            << std::setw(10) << mean
            << std::setw(10) << sd
            << std::endl;
    }

    std::cout << std::string(60, '-') << std::endl;
    std::cout << "=====================================================" << std::endl;
}

// ============================================================
// AddPadding – extend vector to a multiple of LocalSize
//   Returns number of elements added (needed to undo padding later)
// ============================================================
int AddPadding(std::vector<myType>& temp, size_t LocalSize, float PadVal)
{
    int padding_size = (int)(temp.size() % LocalSize);
    if (padding_size)
    {
        std::vector<float> ext(LocalSize - padding_size, PadVal);
        temp.insert(temp.end(), ext.begin(), ext.end());
    }
    return padding_size;
}

// ============================================================
// KernelExec – launches kernel, writes result back into temp
// ============================================================
void KernelExec(cl::Kernel kernel, std::vector<myType>& temp,
    size_t Local_Size, cl::Context context,
    cl::CommandQueue queue,
    bool Two, bool Three, bool Four,
    float FThree, int IFour,
    cl::Event& prof_event, std::string Name)
{
    size_t input_elements = temp.size();
    size_t input_size = input_elements * sizeof(myType);

    std::vector<myType> B(input_elements);
    size_t output_size = B.size() * sizeof(myType);

    cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size,
        &temp[0], NULL, &prof_event);
    std::cout << Name << " write [ns]: "
        << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>()
        - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
        << std::endl;

    queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

    kernel.setArg(0, buffer_A);
    kernel.setArg(1, buffer_B);
    if (Two)   kernel.setArg(2, cl::Local(Local_Size * sizeof(myType)));
    if (Three) kernel.setArg(3, FThree);
    if (Four)  kernel.setArg(4, IFour);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
        cl::NDRange(input_elements),
        cl::NDRange(Local_Size),
        NULL, &prof_event);

    cl::Event read_event;
    queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size,
        &temp[0], NULL, &read_event);
    std::cout << Name << " read  [ns]: "
        << read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>()
        - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
        << std::endl;
}

// ============================================================
// KernelExecRet – launches kernel, returns B[0]
//   Does NOT overwrite temp. Used for reduction kernels where
//   the answer accumulates into B[0] via atomic operations.
// ============================================================
float KernelExecRet(cl::Kernel kernel, std::vector<myType>& temp,
    size_t Local_Size, cl::Context context,
    cl::CommandQueue queue,
    bool Two, bool Three, bool Four,
    float FThree, int IFour,
    cl::Event& prof_event, std::string Name)
{
    size_t input_elements = temp.size();
    size_t input_size = input_elements * sizeof(myType);

    std::vector<myType> B(input_elements, 0.0f);
    size_t output_size = B.size() * sizeof(myType);

    cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
    cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

    queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size,
        &temp[0], NULL, &prof_event);
    std::cout << Name << " write [ns]: "
        << prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>()
        - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
        << std::endl;

    queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);

    kernel.setArg(0, buffer_A);
    kernel.setArg(1, buffer_B);
    if (Two)   kernel.setArg(2, cl::Local(Local_Size * sizeof(myType)));
    if (Three) kernel.setArg(3, FThree);
    if (Four)  kernel.setArg(4, IFour);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
        cl::NDRange(input_elements),
        cl::NDRange(Local_Size),
        NULL, &prof_event);

    cl::Event read_event;
    queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size,
        &B[0], NULL, &read_event);
    std::cout << Name << " read  [ns]: "
        << read_event.getProfilingInfo<CL_PROFILING_COMMAND_END>()
        - read_event.getProfilingInfo<CL_PROFILING_COMMAND_START>()
        << std::endl;

    return B[0]; // the accumulated result lives at index 0
}