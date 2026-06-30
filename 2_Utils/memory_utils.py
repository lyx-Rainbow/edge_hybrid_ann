"""
Memory measurement utilities — Python port of utils.h getPeakRSS / getCurrentRSS.

Returns bytes on all platforms. Returns 0 if unsupported.
"""

import os
import resource
import sys
import platform


def getPeakRSS() -> int:
    """Peak (maximum so far) resident set size in bytes.

    Linux/macOS: uses getrusage(RUSAGE_SELF).ru_maxrss.
    Windows:     uses GetProcessMemoryInfo().PeakWorkingSetSize.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(pmc),
                pmc.cb,
            )
            return pmc.PeakWorkingSetSize

        else:
            # Linux, macOS, BSD
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                # macOS: ru_maxrss is in bytes
                return rusage.ru_maxrss
            else:
                # Linux: ru_maxrss is in KB
                return rusage.ru_maxrss * 1024

    except Exception:
        return 0


def getCurrentRSS() -> int:
    """Current resident set size in bytes.

    Linux:       reads /proc/self/statm.
    macOS:       uses task_info().
    Windows:     uses GetProcessMemoryInfo().WorkingSetSize.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.wintypes.DWORD),
                    ("PageFaultCount", ctypes.wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(pmc),
                pmc.cb,
            )
            return pmc.WorkingSetSize

        elif sys.platform == "linux":
            with open("/proc/self/statm", "r") as fp:
                fields = fp.read().split()
            # statm[1] is RSS in pages
            return int(fields[1]) * os.sysconf("SC_PAGESIZE")

        elif sys.platform == "darwin":
            import ctypes
            import ctypes.util

            libc = ctypes.CDLL(ctypes.util.find_library("c"))

            class MachTaskBasicInfo(ctypes.Structure):
                _fields_ = [
                    ("virtual_size", ctypes.c_uint64),
                    ("resident_size", ctypes.c_uint64),
                    ("resident_size_max", ctypes.c_uint64),
                    ("user_time", ctypes.c_uint64),
                    ("system_time", ctypes.c_uint64),
                    ("policy", ctypes.c_int32),
                    ("suspend_count", ctypes.c_int32),
                ]

            MACH_TASK_BASIC_INFO_COUNT = ctypes.sizeof(MachTaskBasicInfo) // 4
            info = MachTaskBasicInfo()
            kr = libc.task_info(
                libc.mach_task_self(),
                5,  # MACH_TASK_BASIC_INFO
                ctypes.byref(info),
                ctypes.byref(ctypes.c_uint32(MACH_TASK_BASIC_INFO_COUNT)),
            )
            if kr == 0:
                return info.resident_size
            return 0

        else:
            return 0

    except Exception:
        return 0


class QueryRSSSampler:
    """Track max RSS observed between calls to sample().

    Usage:
        sampler = QueryRSSSampler()
        for ...:
            do_query()
            sampler.sample()
        peak_mb = sampler.peak_mb
    """
    def __init__(self):
        self._peak = getCurrentRSS()

    def sample(self):
        cur = getCurrentRSS()
        if cur > self._peak:
            self._peak = cur

    @property
    def peak_mb(self) -> float:
        return self._peak / (1024 * 1024)
