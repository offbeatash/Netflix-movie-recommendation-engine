import os
import psutil
from functools import wraps

def log_memory(stage_name=""):
    """Logs current RAM usage of the Python process in GB."""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3)
    print(f"[Memory] {stage_name} | RAM in use: {mem_gb:.2f} GB")

def memory_tracker(func):
    """Decorator to track memory before and after a pipeline function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        log_memory(f"Starting {func.__name__}")
        result = func(*args, **kwargs)
        log_memory(f"Finished {func.__name__}")
        return result
    return wrapper