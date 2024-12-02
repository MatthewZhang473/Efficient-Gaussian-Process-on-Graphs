import math

def diffusion_modulator(length, beta):
    numerator = (-beta)**length
    denominator = 2**length * math.factorial(length)
    return numerator / denominator