import time
from functools import wraps

from utils.time import format_time


def time_taken(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        formatted_time = format_time(elapsed_time)
        print(f"Function {func.__name__} took {formatted_time}")
        return result

    return wrapper
