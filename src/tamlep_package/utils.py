"""
Utililty functions for tamlep_package
"""

import functools
import os
import random
import time
from typing import Callable

import numpy as np


def set_seed(seed: int = 100):
    """
    Set random seed for random, numpy, os.environ['PYTHONHASHSEED']

    Parameters
    ----------
    seed : int, optional
        An interger to initialize random seed. Default: 100
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def attach_debugger():
    """Attach VS Code debugger"""
    import debugpy

    debugpy.listen(5678)  # Default port for debugpy.
    print("Waiting for debugger!")
    debugpy.wait_for_client()
    print("Attached!")


def timer(func: Callable) -> Callable:
    """
    Decorator to compute execution time

    Parameters
    ----------
    func : Callable
        The function to decorate with a timer.

    Returns
    -------
    Callable
        The decorated function / callable.
    """

    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        run_time = time.perf_counter() - start_time
        print(f"Finished {func.__name__!r} in {run_time:.2f} secs")
        return value

    return _wrapper
