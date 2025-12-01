# Bond Pricing & Vasicek Model

This project implements the Vasicek short-rate model.

## 1. Vasicek Model

The short rate follows:

$$
dr_t = a(b - r_t)dt + \sigma dW_t
$$

## 2. Calibration

We estimate parameters using OLS:

$$
r_{t+1} = \alpha + \beta r_t + \varepsilon_t
$$

...
# Pricing-derivatives-on-Zero-Coupon-Bonds-Vasicek-Model
