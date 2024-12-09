# -*- coding: utf-8 -*-
# 2024.12.9 fengjx

import galois
import numpy as np

VERBOSE = True

from BCH import Euclid


class ReedSolomonCoder():
    def __init__(self, n, t):
        self.n = n
        self.t = t
        self.k = n - 2 * t
        self.m = int(np.round(np.log2(n+1)))
        self.GF = galois.GF(2 ** self.m)
        self.GF.repr('power')
        
        self.a = self.GF.primitive_element
        power_list = np.arange(1, 2 * t + 1)
        self.roots = np.array(self.a ** power_list)
        self.gx = galois.Poly.Roots(self.roots, field=self.GF)
        if VERBOSE:
            print('---- Reed Solomon info ----')
            print(f'n={self.n}, k={self.k}, m={self.m}, t={self.t}')
            print(f'roots: {self.roots}')
            print(f'gx: {self.gx}')

    def Encode(self, message):
        '''
        C' = [C(x), C(x) % g(x)]
        '''
        if len(message) != self.k:
            error(f'error：message bit {len(message)} not equal to k {self.k}')
            return None

        message_x = galois.Poly(message, field=self.GF)
        C_x = message_x * galois.Poly.Degrees([self.n - self.k], coeffs=[1], field=self.GF)
        residue_x = C_x % self.gx
        message_encoded = (C_x + residue_x).coeffs
        if VERBOSE:
            print('---- Reed Solomon Encode ----')
            print(f'message: {message}')
            print(f'message(x): {message_x}')
            print(f'C(x): {C_x}')
            print(f'residue(x): {residue_x}')
            print(f'message_encoded: {message_encoded}')
        return message_encoded

    def Decode(self, message_received):
        '''
        time field method
        '''
        if len(message_received) != self.n:
            error(f'error：message bit {len(message_received)} not equal to n {self.n}')
            return None
        #R_x = self.GF(message_received)
        R_x = message_received
        if VERBOSE:
            print(f'message_received: {message_received}')
            print(f'R(x)={R_x}')
        # syndrome
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
        omega_x = r_x // v_x(0)
        sigma_dx = sigma_x.derivative()
        if VERBOSE:
            print(f"sigma(x)={sigma_x}, omega(x)={omega_x}, sigma'(x)={sigma_dx}")

        E = self.GF([0] * (self.n))
        for i in range(self.n):
            x = self.a ** (-i)
            if sigma_x(x) == 0:
                e = -omega_x(x) // sigma_dx(x)
                E[self.n - 1 - i] = e
                print(f'fnd error in {i} bit!, value is {e}')
        C = message_received + E
        if VERBOSE:
            print(f'message_decoded = {C[0:self.k]}')
            print(f'noise = {E}')
        return C[0:self.k], E

if __name__ == '__main__':
    rsHelper = ReedSolomonCoder(n=7, t=2)

    # --- coding ---
    # len(message) = k = 3， MSB first
    GF = rsHelper.GF
    a = rsHelper.a
    message = GF([1, a**3, 0])
    message_encoded = rsHelper.Encode(message)

    # --- adding noise, n = 7 ---
    noise = GF([0, 0, 0, a**6, a**3, 0, 0])
    message_received = message_encoded + noise

    # --- decoding ---
    message_decoded, noise = rsHelper.Decode(message_received)
    
