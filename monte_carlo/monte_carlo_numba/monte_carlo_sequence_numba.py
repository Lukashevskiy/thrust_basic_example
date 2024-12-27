import random
from time import perf_counter_ns
from numba import njit, prange
import numba

@njit(parallel=True)
def upper(num_samples):
    
    num_threads = numba.get_num_threads()
    chunk_size = num_samples // num_threads + 1
    
   
    for _ in prange(num_threads):
        x = random.uniform(a, b)
        max_f = max(max_f, f(x))

    return max_f

@njit(parallel=True)
def monte_carlo_area(f, a, b, num_samples):
    
    max_f = upper(num_samples)
    
    under_curve = 0
    for _ in prange(num_samples):
        x = random.uniform(a, b)
        y = random.uniform(0, max_f)
        if y <= f(x):
            under_curve += 1

    rect_area = (b - a) * max_f

    area = rect_area * under_curve / num_samples

    return area

@njit(parallel=True)
def f(x):
    return x**2 / (x + 1) + 1 / x

a, b = list(map(int, input().split()))
arr_num_samples = [100, 1_000, 10_000, 100_000, 1_000_000]

for num_samples in arr_num_samples:
    time_begin = perf_counter_ns()
    area = monte_carlo_area(f, a, b, num_samples)
    time_end = perf_counter_ns()
    
    duration = time_end - time_begin
    
    print('-'*20)
    print(f"Num samples \t- {num_samples}")
    print(f'Trapesoid \t- {area}')
    print(f'TIME DURATION \t- {duration // 1_000} milliseconds')
    print('-'*20)