#pragma once
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>
#include <unordered_set>

// 计时类
class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

// 内存监控类
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}

// 内存监控类
// #if defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
// #include <unistd.h>
// #include <sys/resource.h>

// static size_t getCurrentRSS() {
//     struct rusage rusage;
//     getrusage(RUSAGE_SELF, &rusage);
// #if defined(__APPLE__) && defined(__MACH__)
//     return (size_t)rusage.ru_maxrss;
// #else
//     return (size_t)(rusage.ru_maxrss * 1024L);
// #endif
// }

// static size_t getPeakRSS() {
//     struct rusage rusage;
//     getrusage(RUSAGE_SELF, &rusage);
// #if defined(__APPLE__) && defined(__MACH__)
//     return (size_t)rusage.ru_maxrss;
// #else
//     return (size_t)(rusage.ru_maxrss * 1024L);
// #endif
// }
// #else
// static size_t getCurrentRSS() { return 0; }
// static size_t getPeakRSS() { return 0; }
// #endif



// float recall_at_k(int k, unsigned qury_len, unsigned* ground_data, unsigned* results) {
//     if (k <= 0 || ground_data == nullptr || results == nullptr) {
//         return 0.0f;
//     }
//     std::unordered_set<unsigned> ground_set;
//     for (int i = 0; ground_data[i] != UINT_MAX; i++) {
//         ground_set.insert(ground_data[i]);
//     }

//     if (ground_set.empty()) {
//         return 0.0f;
//     }
//     int hits = 0;
//     for (int i = 0; i < k && results[i] != UINT_MAX; i++) {
//         if (ground_set.count(results[i])) {
//             hits++;
//         }
//     }

//     return static_cast<float>(hits) / static_cast<float>(ground_set.size());
// }

float recall_at_k(unsigned k, unsigned query_len, unsigned ground_dim,
                  unsigned* ground_data, unsigned* results) {
    if (k == 0 || ground_data == nullptr || results == nullptr) {
        return 0.0f;
    }

    float total_recall = 0.0f;

    for (unsigned i = 0; i < query_len; i++) {
        // 只取 ground truth 的前 k 个作为标准答案
        std::unordered_set<unsigned> ground_set;
        for (unsigned j = 0; j < k; j++) {
            ground_set.insert(ground_data[i * ground_dim + j]);
        }

        // 计算 results 前 k 个中有多少命中
        int hits = 0;
        for (unsigned j = 0; j < k; j++) {
            if (ground_set.count(results[i * k + j])) {
                hits++;
            }
        }

        total_recall += static_cast<float>(hits) / static_cast<float>(k);
    }

    return total_recall / static_cast<float>(query_len);
}