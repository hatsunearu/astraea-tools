# implements the truncation calculation from 
# An Economical Class of Digital Filters for Decimation and Interpolation (E. Hogenauer 1981)
# section IV B.

# also referenced are:
# https://www.so-logic.net/documents/trainings/03_so_implementation_of_filters.pdf slide 29-30
# https://www.dsprelated.com/showcode/269.php

import numpy as np
from scipy.special import comb as ncr

N = 3
R = 16
M = 1
bits_to_lose = 12

# implements equation (9b)
def get_h_j_kk(j):
    if j <= N:
        kk = np.arange(0, (R*M - 1)*N + j)

        # whatever's inside the sigma_l
        def integrator_inner_loop(k):
            ll = np.arange(0, np.floor(k / R*M) + 1)
            return np.sum(
                np.power(-1, ll) * ncr(N, ll) * \
                ncr(N - j + k - R*M*ll, k - R*M*ll)
            )

        # sum over the sigma arguments
        h_j_kk = np.fromiter(map(integrator_inner_loop, kk), np.double)        

    elif j >= N+1:
        kk = np.arange(0, 2*N + 1 - j + 1)

        h_j_kk = np.power(-1, kk) * ncr(2*N+1-j, kk)
    
    return h_j_kk

print("#### h_j_k #####")
for j in range(1, 2*N+1):
    h_j_kk = get_h_j_kk(j)
    print(f"\nh_{j}_k: {h_j_kk}")

# 16b
print("\n#### F_j_sq ####")
def get_F_j_sq(j):
    if 1 <= j <= 2*N:
        return np.sum(np.square(get_h_j_kk(j)))
    elif j == 2*N+1:
        return 1
    else:
        raise ValueError("j out of range on F_j_sq")

F_jj = np.fromiter(map( lambda j: np.sqrt(get_F_j_sq(j)) , \
    range(1, 2*N+2)), np.double)
print(f"F_jj: {F_jj}")

total_error_variance = 2**(2*bits_to_lose) / 12
print(f"sigma_T_2N+1_sq (total error variance) = {total_error_variance}")

B_ii = np.floor(-np.log2(F_jj) + np.log2(np.sqrt(total_error_variance)) + 0.5 * np.log2(6/N))
B_ii[2*N] = bits_to_lose
print("\n------------------------tear here------------------------")
print("#########################################################")
print(f"Using parameters R={R}, M={M}, N={N}, Total Bits to lose={bits_to_lose}")
print(f"B_i (bits to lose after each stage): \n{B_ii}")
print("#########################################################")
    
