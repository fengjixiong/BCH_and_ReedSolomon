# -*- coding: utf-8 -*-
# 2024.12.6 fengjx

import galois
import numpy as np

VERBOSE = True

def is_bit_set(x, n):
    return (x & (1 << n))

def get_cyclotomic_cosets(m):
    """
    Fills a list of cyclotomic cosets. For GF(16) = GF(2⁴) cyclotomic cosets are:
    :param m: the power in size of the field GF(2ᵖᵒʷᵉʳ).
    :returns: a list of cyclotomic cosets.
    [[0], 
    [a**1, a**2, a**4, a**8], 
    [a**3, a**6, a**9, a**12], 
    [a**5, a**10], 
    [a**7, a**14, a**13, a**11]]
    """
    cyclotomic_cosets = []
    all_cyclotomic_members = 1
    i = 0

    while all_cyclotomic_members < 2 ** (2 ** m - 1) - 1:
        cyclotomic_cosets.append(0)
        k = 0
        while True:
            if not is_bit_set(all_cyclotomic_members, k):
                break
            k += 1
        while True:
            k = k % (2 ** m - 1)
            if is_bit_set(cyclotomic_cosets[i], k):
                break
            cyclotomic_cosets[i] ^= 1 << k
            k *= 2
        all_cyclotomic_members ^= cyclotomic_cosets[i]
        i += 1

    cyclotomic_cosets.append(0) # {a0}
    return cyclotomic_cosets

def get_cyclotomic_cosets_GF(cyclotomic_cosets, m, GF):
    N = len(cyclotomic_cosets)
    cyclotomic_cosets_GF = [None for i in range(N)]
    a = GF.primitive_element

    for j in range(N):
        power_list = []
        for i in range(2 ** m):
            if is_bit_set(cyclotomic_cosets[j], i):
                power_list.append(i)
        if len(power_list) == 0:
            power_list = [0]

        cyclotomic_cosets_GF[j] = GF(a ** np.array(power_list, dtype=np.uint32))

    return cyclotomic_cosets_GF


def Euclid(GF, a, b, max_vx=None, max_rx=None):
    '''
    a, b support int or list
    get ux, vx，for GCD(ax, bx) = dx
    and u(x) * a(x) + v(x) * b(x) = d(x)
    ax, gx are type of galois.Poly
    '''
    if type(a) == type(123):
        a = [int(x) for x in bin(a)[2:]]
    if type(b) == type(123):
        b = [int(x) for x in bin(b)[2:]]
    ax, bx = galois.Poly(a, field=GF), galois.Poly(b, field=GF)
    u0, v0, r0 = galois.Poly([1], field=GF), galois.Poly([0], field=GF), ax
    u1, v1, r1 = galois.Poly([0], field=GF), galois.Poly([1], field=GF), bx
    i = 1
    if VERBOSE:
	    print(f'input:  a(x)={ax}, b(x)={bx}, (u,v)=({max_vx}, {max_rx})')
	    print('--------------------------------------------------------------')
	    print('i |       u         |      v          |       r         |  q')
	    print('--------------------------------------------------------------')
    while True:
        if max_vx is not None and max_rx is not None:
            if v1.degree <= max_vx and r1.degree <= max_rx:
                break
        qx, rx = r0//r1, r0%r1
        if rx == 0:
            break
        ux = u0 + qx * u1
        vx = v0 + qx * v1
        u0, v0, r0 = u1, v1, r1   
        u1, v1, r1 = ux, vx, rx
        if VERBOSE:
        	print(f'{i} | {ux} | {vx} | {rx} | {qx}')
        i += 1
    if VERBOSE:
	    print('--------------------------------------------------------------')
	    print(f'result: u(x)={u1}, v(x)={v1}, d(x)={r1}')

    return u1, v1, r1

class BCHCoder():
    def __init__(self, n:int, t:int):
        self.n = n
        self.t = t
        self.m = int(np.round(np.log2(n+1)))
        self.GF = galois.GF(2 ** self.m)
        self.GF.repr('power')
        
        # cyclotomic cosets
        self.a = self.GF.primitive_element
        self.cosets_bin = get_cyclotomic_cosets(self.m)
        self.cosets_GF = get_cyclotomic_cosets_GF(self.cosets_bin, self.m, self.GF)
        # generative poly
        A = []
        for i in range(t):
            A.append(self.cosets_GF[i])

        self.roots = np.hstack(A)
        self.gx = galois.Poly.Roots(self.roots, field=self.GF)
        self.k = self.n - self.gx.degree

        if VERBOSE:
            print('---- BCH info ----')
            print(f'n={self.n}, k={self.k}, m={self.m}, t={self.t}')
            print(f'roots: {self.roots}')
            print(f'gx: {self.gx}')

    def Encode(self, message:list):
        '''
        C' = [C(x), C(x) % g(x)]
        '''
        if len(message) != self.k:
            error(f'error：message bit {len(message)} not equal to k {self.k}')
            return None

        message_x = galois.Poly(message, field=self.GF)
        C = message_x * galois.Poly.Degrees([self.n - self.k], coeffs=[1], field=self.GF)
        residue_x = C % self.gx
        residue = residue_x.coeffs.tolist()
        message_encoded = message + [0] * (self.n - self.k - len(residue)) + residue
        if VERBOSE:
            print('---- BCH Encode ----')
            print(f'message: {message}')
            print(f'message(x): {message_x}')
            print(f'C: {C}')
            print(f'residue(x): {residue_x}')
            print(f'message_encoded: {message_encoded}')
        return message_encoded


    def Decode(self, message_received:list):
        '''
        时域解码
        '''
        if len(message_received) != self.n:
            error(f'error：message bit {len(message_received)} not equal to n {self.n}')
            return None
        R_x = self.GF(message_received)
        if VERBOSE:
        	print(f'message_received: {message_received}')
        	print(f'R(x)={R_x}')
        # Syndrome
        S = [None] * (2 * self.t)
        for j in range(1, 2*self.t+1, 1):
            Sj = self.GF(0)
            for i in range(self.n):
                Sj = Sj + R_x[self.n - 1 - i] * (self.a ** (i * j))
            S[2 * self.t - j] = Sj
        S_x = galois.Poly(S, field=self.GF)

        if VERBOSE:
        	print(f'S(x): {S_x}')

        if S_x == 0:
            print('info: no error')
            return message_received[0:self.k], None

        x_2t = galois.Poly.Degrees([2*self.t], coeffs=[1], field=self.GF)
        u_x, v_x, r_x = Euclid(self.GF, x_2t.coeffs.tolist(), S_x.coeffs.tolist(), self.t, self.t-1)
        if VERBOSE:
        	print(f'u(x)={u_x}, v(x)={v_x}, r(x)={r_x}')

        if v_x(0) == 0:
        	error('Error: decode error with v(0)=0!')
        	return None

        sigma_x = v_x // v_x(0)
        if VERBOSE:
        	print(f'sigma(x)={sigma_x}')

        E = [0] * (self.n)
        for i in range(self.n):
            if sigma_x(self.a ** (-i)) == 0:
                E[self.n - 1 - i] = 1
                print(f'fnd error in {i} bit!')
        C = (np.array(message_received) ^ np.array(E)).tolist()
        if VERBOSE:
        	print(f'message_decoded = {C[0:self.k]}')
        	print(f'noise = {E}')
        return C[0:self.k], E

if __name__ == '__main__':
    bchHelper = BCHCoder(n=15, t=3)

    # --- coding ---
    # len(message) = k = 5, MSB first
    message = [1, 0, 1, 0, 1]
    # len(message_encoded) = n = 15
    message_encoded = bchHelper.Encode(message)

    # --- adding noise ---
    # len(noise) = n = 15
    noise = [0] * 15
    noise[14-2] = 1
    noise[14-7] = 1
    message_received = (np.array(message_encoded) ^ np.array(noise)).tolist()

    # --- decoding ---
    message_decoded, noise = bchHelper.Decode(message_received)
    
    