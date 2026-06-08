import os
import platform
import socket
import sys
import psutil
import subprocess


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return None


def collect_system_info():
    memory = psutil.virtual_memory()

    return {
        "hostname": socket.gethostname(),

        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },

        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "frequency_mhz": (
                psutil.cpu_freq().current
                if psutil.cpu_freq()
                else None
            ),
        },

        "memory": {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
        },

        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },

        "git_commit": get_git_commit(),

        "environment": {
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS"),
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        }
    }