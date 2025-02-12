# -*- coding: utf-8 -*-
# 2024.12.9 fengjx

import galois
import numpy as np
import sys

VERBOSE = True

from BCH import Euclid


class ReedSolomonCoder():
    def __init__(self, n, t):
        self.n = n
        self.t = t
        self.k = n - 2 * t
        self.m = int(np.round(np.log2(n+1)))
        self.GF = galois.GF(2 ** self.m)
        #self.GF.repr('int') # power
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

    def Decode_with_erase(self, message_received, erase=[]):
        '''
        time field method
        '''
        if len(message_received) != self.n:
            print(f'error：message bit {len(message_received)} not equal to n {self.n}')
            sys.exit(-1)
        #R_x = self.GF(message_received)
        R_x = message_received
        e0 = len(erase)
            
        if VERBOSE:
            print(f'message_received: {message_received}')
            print(f'R(x)={R_x}')
            print(f'e0={e0}, erase={erase}')

        # syndrome
        def getSyndrome(R, n, t, GF, a):
            S = [None] * (2 * t)
            for j in range(1, 2*t+1, 1):
                Sj = GF(0)
                for i in range(n):
                    Sj = Sj + R[n - 1 - i] * (a ** (i * j))
                S[2 * t - j] = Sj
            S_x = galois.Poly(S, field=GF)
            if VERBOSE:
                print(f'S(x): {S_x}')
            return S_x
        
        S_x = getSyndrome(R_x, self.n, self.t, self.GF, self.a)
        
        if S_x == 0:
            print('info: no error')
            return message_received[0:self.k], None
        x_2t = galois.Poly.Degrees([2*self.t], coeffs=[1], field=self.GF)
        # erase poly
        sigma_x_0 = galois.Poly([1], field=self.GF)
        for i in erase:
            sigma_x_0 = sigma_x_0 * galois.Poly([self.a**i, 1], field=self.GF)
        if VERBOSE:
            print(f'sigma0(x) = {sigma_x_0}')

        S_x_0 = (S_x * sigma_x_0) % (x_2t)
        if VERBOSE:
            print(f'S0(x) = {S_x_0}')

        
        ui = np.floor(self.t - e0 / 2.0)
        vi = np.ceil(self.t + e0 / 2.0) - 1
        u_x, v_x, r_x = Euclid(self.GF, x_2t.coeffs.tolist(), S_x_0.coeffs.tolist(), self.t, self.t-1)
        if VERBOSE:
            print(f'u(x)={u_x}, v(x)={v_x}, r(x)={r_x}')

        if v_x(0) == 0:
            print('Error: decode error with v(0)=0!')
            #sys.exit(-1)

        sigma_x_1 = v_x // v_x(0)
        if VERBOSE:
            print(f'sigma1(x) = {sigma_x_1}')
        sigma_x = sigma_x_0 * sigma_x_1
        if VERBOSE:
            print(f'sigma(x) = {sigma_x}')
        
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
        C_x = message_received + E
        if VERBOSE:
            print(f'message_decoded = {C_x[0:self.k]}')
            print(f'noise = {E}')
            
        # Verify decoded code
        S_x_new = getSyndrome(C_x, self.n, self.t, self.GF, self.a)
        
        if VERBOSE:
            print(f'S_new(x): {S_x_new}')

        if S_x_new == 0:
            print('info: decode OK')
        else:
            print(f'Error: decode Error! More than {self.t} errors in code')
        
        return C_x[0:self.k], E


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
        def getSyndrome(R, n, t, GF, a):
            S = [None] * (2 * t)
            for j in range(1, 2*t+1, 1):
                Sj = GF(0)
                for i in range(n):
                    Sj = Sj + R[n - 1 - i] * (a ** (i * j))
                S[2 * t - j] = Sj
            S_x = galois.Poly(S, field=GF)
            if VERBOSE:
                print(f'S(x): {S_x}')
            return S_x
        
        S_x = getSyndrome(R_x, self.n, self.t, self.GF, self.a)
        
        if S_x == 0:
            print('info: no error')
            return message_received[0:self.k], None

        x_2t = galois.Poly.Degrees([2*self.t], coeffs=[1], field=self.GF)
        u_x, v_x, r_x = Euclid(self.GF, x_2t.coeffs.tolist(), S_x.coeffs.tolist(), self.t, self.t-1)
        if VERBOSE:
            print(f'u(x)={u_x}, v(x)={v_x}, r(x)={r_x}')

        if v_x(0) == 0:
            print('Error: decode error with v(0)=0!')
            #sys.exit(-1)

        sigma_x = v_x // v_x(0)
        print(f'sigma(x) = {sigma_x}')
        
        if len(sigma_x.coeffs) < 3:
            print("Error: sigma(x)'s x2 coeff is 0")
            sys.exit(-1)
            
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
        C_x = message_received + E
        if VERBOSE:
            print(f'message_decoded = {C_x[0:self.k]}')
            print(f'noise = {E}')
            
        # Verify decoded code
        S_x_new = getSyndrome(C_x, self.n, self.t, self.GF, self.a)
        
        if VERBOSE:
            print(f'S_new(x): {S_x_new}')

        if S_x_new == 0:
            print('info: decode OK')
        else:
            print(f'Error: decode Error! More than {self.t} errors in code')
        
        return C_x[0:self.k], E

def Test1():
    rsHelper = ReedSolomonCoder(n=7, t=2)
    # --- coding ---
    # len(message) = k = 3， MSB first
    GF = rsHelper.GF
    a = rsHelper.a
    #GF.repr('int')
    message = GF([1, a**3, 0])
    message_encoded = rsHelper.Encode(message)

    # # --- adding noise, n = 7 ---
    # noise = GF([0, 0, 0, a**6, a**3, 0, 0])
    # message_received = message_encoded + noise

    # # --- decoding ---
    # message_decoded, noise = rsHelper.Decode(message_received)
    
    # --- adding noise, more than t=2, n = 7 ---
    noise = GF([0, 0, 0, a**6, a**3, 0, 0])
    print(f'noise = {noise}')
    message_received = message_encoded + noise

    # --- decoding ---
    message_decoded, noise = rsHelper.Decode(message_received)
    print('')
    
    # --- adding noise, more than t=2, n = 7 ---
    for i in range(7):
        for j in range(8):
            noise = GF([0, 0, 0, a**6, a**3, 0, 0])
            noise[i] = a**j
            print(f'message = {message}')
            print(f'encode = {message_encoded}')
            print(f'noise = {noise}')
            message_received = message_encoded + noise

            # --- decoding ---
            message_decoded, noise = rsHelper.Decode(message_received)
            print('')
            print('')
   
def Test2():
    rsHelper = ReedSolomonCoder(n=31, t=3)
    GF = rsHelper.GF
    a = rsHelper.a
    # len(message) = k = n-2t = 25， MSB first

    if True:
        # --- coding ---
        message = GF([1, a**10, a**9, a**24, a**16, a**24, a**21] + [0] * 18)
        message_encoded = rsHelper.Encode(message)
        # --- adding noise, more than t=2, n = 7 ---
        noise = GF([a**17, 0, 0, a**24, 0, a**24] + [0] * 25)
        print(f'noise = {noise}')
        message_received = message_encoded + noise
    else:
        message_received = GF([a**30, a**10, a**9, 0, a**16, 0, a**21] + [0] * 24)
    # --- decoding ---
    message_decoded, noise = rsHelper.Decode(message_received)
    print('')

def Test3():
    rsHelper = ReedSolomonCoder(n=31, t=3)
    GF = rsHelper.GF
    a = rsHelper.a
    # len(message) = k = n-2t = 25， MSB first

    message_received = GF([a**30, a**10, a**9, 0, a**16, 0, a**21] + [0] * 24)
    message_decoded, noise = rsHelper.Decode_with_erase(message_received, erase=[27, 25])
    print('')
    print('----')
    print('')

    message_received = GF([0, 0, 1, a**22, a**9, 0, a**16, a**24, 0] + [0] * 22)
    message_decoded, noise = rsHelper.Decode_with_erase(message_received, erase=[22, 25])
    print('')
    print('----')
    print('')

    message_received = GF([0, 0, 0, a**24, a**16, a**24, a**21] + [0] * 24)
    message_decoded, noise = rsHelper.Decode_with_erase(message_received, erase=[28, 29, 30])
    print('')
    print('----')
    print('')

    message_received = GF([0] * 18 + [0, a**21, 0, a**24, a**16, a**24, a**21] + [0] * 6)
    message_decoded, noise = rsHelper.Decode_with_erase(message_received, erase=[10, 12])
    print('')
    print('----')
    print('')

if __name__ == '__main__':
    #Test1()
    #Test2()
    Test3()

