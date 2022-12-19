import random
import numpy as np

def sample_gaussian(mean, stddev):
    return random.gauss(mean, stddev)

def sample_uniform(low, high):
    return random.uniform(low, high)

def stratified_quadrature_sampling(t_i, t_f, N, density_func, color_func):
    #TODO finish
    pass