import numpy as np

π = np.pi


def get_wave_vector(λ, θ):
    """Gets wave vector for doa and wavelength"""
    k = 2 * π / λ * np.sin(θ)
    return k


def get_steering_vector(k, r):
    """Gets steering vector of antenna array"""
    v = np.exp(-1j * k * r)
    return v


def get_ula_positions(N, d):
    """Gets uniform linear array positions"""
    r = (np.arange(N) * d)[:, None]
    return r


def get_radiation_pattern(θ):
    """Gets isotropic radiation pattern"""
    rp = np.ones_like(θ)
    return rp


def generate_signal(L):
    """Generates complex random gaussian signal"""
    m = np.random.randn(L) + 1j * np.random.randn(L)
    m /= np.sqrt(2)
    return m


def apply_response(s, x):
    """Applies antenna response to signal"""
    feeds = s * x
    return feeds


def add_noise(x, coef=1):
    """Adds noise on all feeds with coef"""
    N, L = x.shape
    x += coef * (np.random.randn(N, L) + 1j * np.random.randn(N, L))
    return x


def get_covariance(x):
    """Computes complex covariance matrix"""
    _, L = x.shape
    rxx = (x @ x.conj().T) / L
    return rxx


def get_array_factor(s, w=None):
    """Gets array factor of antenna array"""
    if w is None:
        w = np.ones(s.shape[0])
    af = np.abs(np.sum(w * s, axis=0))
    return af


def generate_random_phase(N):
    """Generates random phases to emulate DUT response"""
    phases = np.exp(2j * np.pi * np.random.uniform(size=(N, 1)))
    phases[0] = 1
    return phases


def music(rxx, s):
    """Compute MUSIC spectrum"""
    w, v = np.linalg.eig(rxx)
    v = v[:, np.flip(np.argsort(w))]
    Un = v[:, 1:]
    Pn = Un @ Un.conj().T
    spectrum = np.reciprocal(np.linalg.norm(Pn @ s, axis=0))
    return spectrum
