
# Pricing-derivatives-on-Zero-Coupon-Bonds-Vasicek-Model

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


## Vasicek Short-Rate Model

$$
dr_t = a(b - r_t)\,dt + \sigma\,dW_t
$$

## Discretized Vasicek (Euler)

$$
r_{t+\Delta t} = r_t + a(b-r_t)\Delta t + \sigma\sqrt{\Delta t}\,\varepsilon
$$

## OLS Calibration Regression

$$
r_{t+1} = \alpha + \beta r_t + \varepsilon_t
$$

## Mapping OLS to Continuous Parameters

$$
\beta = e^{-a\Delta t}
$$

$$
a = -\frac{\ln(\beta)}{\Delta t}
$$

$$
b = \frac{\alpha}{a(1-\beta)}
$$

## Sigma From Residual Variance

$$
\sigma = \sqrt{ \frac{2a\,\text{Var}(\varepsilon)}{1 - e^{-2a\Delta t}} }
$$

## Zero-Coupon Bond Pricing (Closed Form)

$$
P(0,T) = A(0,T)\,e^{-B(0,T)r_0}
$$

Where:

$$
B(0,T) = \frac{1 - e^{-aT}}{a}
$$

$$
A(0,T) = \exp\left[\left(B - T\right)\left(\frac{a^2 b - \sigma^2/2}{a^2}\right)
       - \frac{\sigma^2 B^2}{4a}\right]
$$


## Monte Carlo Pricing

We compute:

$$
P(0,T) = \mathbb{E}\left[ e^{-\int_0^T r(t) dt } \right]
$$

Using:

$$
r_{t+\Delta t} = r_t + a(b-r_t)\Delta t + \sigma\sqrt{\Delta t}\,\varepsilon
$$


## Fair Interest Rate Swap Fixed Rate

$$
K = \frac{1 - P(0,T_N)}{\sum_{i=1}^{N} P(0,T_i)}
$$

## European Bond Call Option

$$
C = e^{-r_0 T}\,\mathbb{E}\left[\max(P(T,T_b) - K, 0)\right]
$$

## Forward Swap Rate

$$
F = \frac{P(0,T_0) - P(0,T_N)}{\sum_{i=1}^N P(0,T_i)}
$$


## Black's Formula (Payer Swaption)

$$
C = A \left[ F N(d_1) - K N(d_2) \right]
$$

where

$$
d_1 = \frac{\ln(F/K) + \frac12\sigma^2 T}{\sigma\sqrt{T}}
$$

$$
d_2 = d_1 - \sigma\sqrt{T}
$$

pip install -r requirements.txt

python examples/run_pipeline.py

python examples/run_swaption.py


MIT License

Copyright (c) 2025
Pravir Pandao

