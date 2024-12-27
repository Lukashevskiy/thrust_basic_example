import numpy as np
from numba import jit
from time import perf_counter_ns

@jit(nopython=True, parallel=True)
def monte_carlo_area(f, a, b, num_samples):

    x = np.random.uniform(a, b, num_samples)
    y = np.random.uniform(0, 1, num_samples)

    f_values = f(x)

    max_f = np.max(f_values)

    y *= max_f

    under_curve = y <= f_values

    rect_area = (b - a) * max_f

    area = rect_area * np.sum(under_curve) / num_samples

    return area

@jit(nopython=True, parallel=True)
def f(x):
    return x**2 / (x + 1) + 1 / x

if __name__ == '__main__':
    a, b = list(map(int, input().split()))
    arr_num_samples = [100, 1_000, 10_000, 100_000, 1_000_000]
    
    monte_carlo_area(f, a, b, 1)
    
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