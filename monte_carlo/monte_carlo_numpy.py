import numpy as np

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

def f(x):
    return x**2 / (x + 1) + 1 / x

a, b = 1, 5 
num_samples = 100_000  

area = monte_carlo_area(f, a, b, num_samples)
print(f"Оценка площади криволинейной трапеции: {area}")
