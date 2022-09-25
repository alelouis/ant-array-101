"""
This script demonstrate one way of calibrating antenna arrays responses from covariance matrices.
MUSIC estimation is performed for analytical, calibrated and estimated responses.
Author : Alexis LOUIS
"""

import numpy as np
import matplotlib.pyplot as plt
from array import *

N, π, λ, L, n_θ, nc = 4, np.pi, 1, 200, 1000, 0.3
θ = np.linspace(-π/2, π/2, n_θ)
d = λ/2
k = get_wave_vector(λ, θ)
r = get_ula_positions(N, d)
v = get_steering_vector(k, r)
rp = get_radiation_pattern(θ)
phases = generate_random_phase(N)
s_analytical = v * rp
s = s_analytical * phases
m = generate_signal(L)

# Estimate responses from covariance matrices
s_est = np.zeros_like(s)
for idx in range(n_θ):
    x = apply_response(s[:, [idx]], m)
    x = add_noise(x, nc)
    rxx = get_covariance(x)
    s_est[:, idx] = rxx[:, 0]

# Recalibrate responses from estimated phases from cov matrices
phases_estimated = s_est[:, 0] / s_analytical[:, 0]
s_calibrated = s_analytical * phases_estimated[:, None]

# Compute signals for a given doa
doa_idx = 400
x = apply_response(s[:, [doa_idx]], m)
x = add_noise(x, nc)

# Compute music spectrum for analytical, calibrated and estimated responses
rxx = get_covariance(x)
spectrum_s_analytical = music(rxx, s_analytical)
spectrum_s_calibrated = music(rxx, s_calibrated)
spectrum_s_estimated = music(rxx, s_est)

# Plot spectrums
plt.figure(figsize = (10, 5), dpi = 100)
plt.semilogy(θ, spectrum_s_analytical, label = "analytical", c = 'k', linewidth = 1)
plt.semilogy(θ, spectrum_s_calibrated, label = "calibrated", c = 'blue', linewidth = 1)
plt.semilogy(θ, spectrum_s_estimated, label = "estimated", c = 'red', linewidth = 1)
plt.grid(which="minor", linestyle = ':')
plt.grid(which="major", linestyle = '-')
plt.axvline(x=θ[doa_idx], c = "k", linestyle = "--", label = "True DOA")
plt.legend()
plt.xlim([θ.min(), θ.max()])
plt.savefig('fig.png', dpi = 150, transparent=True)
plt.show()