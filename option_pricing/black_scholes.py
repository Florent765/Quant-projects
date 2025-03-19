import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
from plotly.subplots import make_subplots

class BS_formula:

    def __init__(self, S, K, T, sigma, r):
        self.S = S
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T

    def d1(self, S, K, T, sigma, r):
        if sigma <= 0 or T <= 0:
            return float('nan')
        return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    def d2(self, S, K, T, sigma, r):
        if sigma <= 0 or T <= 0:
            return float('nan')
        return self.d1(S, K, T, sigma, r) - sigma * np.sqrt(T)

    def call_price(self, S, K, T, sigma, r):
        if T <= 0:
            return float('nan')
        d1 = self.d1(S, K, T, sigma, r)
        d2 = self.d2(S, K, T, sigma, r)
        if np.isnan(d1) or np.isnan(d2):
            return 0.0
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def put_price(self, S, K, T, sigma, r):
        if T <= 0:
            return float('nan')
        d1 = self.d1(S, K, T, sigma, r)
        d2 = self.d2(S, K, T, sigma, r)
        if np.isnan(d1) or np.isnan(d2):
            return 0.0
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Test script for BlackScholes calculator
calculator = BS_formula(S=100, K=100, T=1, sigma=0.2, r=0.05)

# Test standard case
call_price = calculator.call_price(100, 100, 1, 0.2, 0.05)
put_price = calculator.put_price(100, 100, 1, 0.2, 0.05)
print(f"At-the-money options (S=K=100): Call = {call_price:.4f}, Put = {put_price:.4f}")

# Test in-the-money call / out-of-the-money put
call_price = calculator.call_price(120, 100, 1, 0.2, 0.05)
put_price = calculator.put_price(120, 100, 1, 0.2, 0.05)
print(f"In-the-money call (S=120, K=100): Call = {call_price:.4f}, Put = {put_price:.4f}")

# Test edge case: expiration (T=0)
call_price = calculator.call_price(100, 100, 0, 0.2, 0.05)
put_price = calculator.put_price(100, 100, 0, 0.2, 0.05)
print(f"At expiration (T=0): Call = {call_price:.4f}, Put = {put_price:.4f}")