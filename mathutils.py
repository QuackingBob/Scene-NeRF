import random
import numpy as np

def sample_gaussian(mean, stddev):
    return random.gauss(mean, stddev)

def sample_uniform(low, high):
    return random.uniform(low, high)

def stratified_quadrature_sampling(t_n, t_f, N, density_func, color_func, ray_func, d):
    T_i_sum = 0
    C_r = np.zeros((3))
    t_i = 0
    for i in range(N):
        t_i_next = sample_uniform(t_n + (i - 1) / N * (t_f - t_n), t_n + i / N * (t_f - t_n))
        delta_i = t_i_next - t_i
        t_i = t_i_next 
        r_i = ray_func(t_i)
        density_sample = density_func(r_i)
        color_sample = color_func(r_i, d)
        T_i_sum += density_sample * delta_i
        T_i = np.exp(-1 * T_i_sum)
        C_r += T_i * (1 - np.exp(-1 * density_sample * delta_i)) * color_sample
    return C_r


