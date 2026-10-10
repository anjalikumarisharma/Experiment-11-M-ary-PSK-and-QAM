"""
===============================================================================
Digital Communication Laboratory - Experiment 11
M-ary PSK and M-ary QAM: Constellation Normalization, Gray Mapping & Efficiency
===============================================================================
Programme: B.Tech in Electronics and Communication Engineering
Platform: Python 3 (NumPy, SciPy, Matplotlib, Pandas)
Laboratory Mode: Self-contained first-principles simulation without toolboxes

Description:
  Implements multi-level digital modulation transceivers:
  1. Constellation synthesis for QPSK, 8-PSK, 16-QAM, and 64-QAM.
  2. Average symbol energy normalization to Es = 1.0.
  3. Minimum Euclidean distance detection / decision slicing.
  4. Constellation clouds at SNR levels: 0 dB, 5 dB, 10 dB, and 20 dB.
  5. Gray mapping vs Natural (Binary) mapping performance comparison.
  6. Theoretical vs Monte Carlo SER and BER performance curves.
  7. Spectral efficiency trade-offs (bits/s/Hz) and Shannon capacity bound.
  8. Mandatory syllabus validation: Compute and report numerical average
     constellation energy before and after normalization.
===============================================================================
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import pandas as pd

# Set plot styling for publication-quality visuals
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.5,
    "grid.alpha": 0.5,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

# Setup Output Directory
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. GRAY CODE UTILITIES & MAPPING TABLES
# =============================================================================

def binary_to_gray(n: int) -> int:
    """Convert an integer to its binary-reflected Gray code equivalent."""
    return n ^ (n >> 1)


def generate_gray_code_table(num_bits: int) -> List[str]:
    """Generate ordered list of bit-strings for standard Gray code."""
    n_states = 1 << num_bits
    return [format(binary_to_gray(i), f"0{num_bits}b") for i in range(n_states)]


# =============================================================================
# 2. CONSTELLATION GENERATION & NORMALIZATION (FIRST PRINCIPLES)
# =============================================================================

def get_mpsk_constellation(m_order: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, complex], float, float]:
    """
    Generate normalized M-PSK constellation:
      Points: s_m = exp(j * (2*pi*m / M + pi/M))
      Gray coded around circle.
    Returns:
      (unnormalized_symbols, normalized_symbols, bit_to_symbol_map, unnorm_energy, norm_energy)
    """
    k = int(np.log2(m_order))
    gray_codes = generate_gray_code_table(k)

    angles = np.array([(2.0 * m + 1.0) * np.pi / m_order for m in range(m_order)])
    unnorm_syms = np.exp(1j * angles)
    unnorm_energy = float(np.mean(np.abs(unnorm_syms) ** 2))

    norm_factor = 1.0 / np.sqrt(unnorm_energy)
    norm_syms = unnorm_syms * norm_factor
    norm_energy = float(np.mean(np.abs(norm_syms) ** 2))

    mapping = {gray_codes[i]: norm_syms[i] for i in range(m_order)}
    return unnorm_syms, norm_syms, mapping, unnorm_energy, norm_energy


def get_mqam_constellation(m_order: int, use_gray: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict[str, complex], float, float]:
    """
    Generate square M-QAM constellation (e.g. 16-QAM, 64-QAM):
      Levels per dimension: 2*i - 1 - sqrt(M) for i in 1..sqrt(M).
      Unnormalized average energy: E_unnorm = 2 * (M - 1) / 3.
      Normalization factor: 1 / sqrt(E_unnorm).
    Returns:
      (unnormalized_symbols, normalized_symbols, bit_to_symbol_map, unnorm_energy, norm_energy)
    """
    sqrt_m = int(np.sqrt(m_order))
    k = int(np.log2(m_order))
    k_per_axis = k // 2

    # 1D coordinate levels: e.g. for 16-QAM [-3, -1, +1, +3]
    levels_1d = 2 * np.arange(sqrt_m) - (sqrt_m - 1)

    if use_gray:
        code_1d = generate_gray_code_table(k_per_axis)
    else:
        code_1d = [format(i, f"0{k_per_axis}b") for i in range(sqrt_m)]

    # Form 2D grid
    unnorm_list = []
    mapping = {}

    for i, i_val in enumerate(levels_1d):
        for q, q_val in enumerate(levels_1d):
            unnorm_pt = complex(i_val, q_val)
            unnorm_list.append(unnorm_pt)

    unnorm_syms = np.array(unnorm_list)
    unnorm_energy = float(np.mean(np.abs(unnorm_syms) ** 2))

    norm_factor = 1.0 / np.sqrt(unnorm_energy)
    norm_syms = unnorm_syms * norm_factor
    norm_energy = float(np.mean(np.abs(norm_syms) ** 2))

    # Construct mapping dictionary
    idx = 0
    for i, i_val in enumerate(levels_1d):
        for q, q_val in enumerate(levels_1d):
            bit_str = code_1d[i] + code_1d[q]
            mapping[bit_str] = norm_syms[idx]
            idx += 1

    return unnorm_syms, norm_syms, mapping, unnorm_energy, norm_energy


# =============================================================================
# 3. MINIMUM EUCLIDEAN DISTANCE DETECTOR (SLICER)
# =============================================================================

def demodulate_minimum_distance(rx_symbols: np.ndarray,
                                mapping: Dict[str, complex]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimum Euclidean distance maximum-likelihood slicer:
      m_hat = argmin_m |r - s_m|^2
    Returns:
      (detected_bits, detected_symbols)
    """
    bit_strings = list(mapping.keys())
    constellation = np.array(list(mapping.values()))
    k = len(bit_strings[0])

    # Compute Euclidean distance squared matrix: shape (N_rx, M)
    # |r - s|^2 = (Re(r) - Re(s))^2 + (Im(r) - Im(s))^2
    rx_col = rx_symbols[:, np.newaxis]
    const_row = constellation[np.newaxis, :]
    dist_sq = np.abs(rx_col - const_row) ** 2

    best_indices = np.argmin(dist_sq, axis=1)
    detected_symbols = constellation[best_indices]

    # Map back to bits
    detected_bits = np.empty(len(rx_symbols) * k, dtype=int)
    for i, idx in enumerate(best_indices):
        b_str = bit_strings[idx]
        for b_pos in range(k):
            detected_bits[i * k + b_pos] = int(b_str[b_pos])

    return detected_bits, detected_symbols


# =============================================================================
# 4. AWGN CHANNEL
# =============================================================================

def add_awgn_complex(symbols: np.ndarray, snr_db: float, bits_per_symbol: int,
                     seed: Optional[int] = None) -> np.ndarray:
    """
    Add complex AWGN to symbols with average energy Es = 1.0.
    Eb = Es / bits_per_symbol.
    Eb/N0 (linear) = 10^(snr_db / 10).
    Noise power spectral density N0 = Eb / (Eb/N0).
    Noise variance sigma^2 = N0 / 2 per real dimension.
    """
    eb = 1.0 / bits_per_symbol
    snr_lin = 10.0 ** (snr_db / 10.0)
    n0 = eb / snr_lin
    sigma = np.sqrt(n0 / 2.0)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=len(symbols)) + 1j * rng.normal(0.0, sigma, size=len(symbols))
    return symbols + noise


# =============================================================================
# 5. MANDATORY SYLLABUS VALIDATION: CONSTELLATION ENERGY
# =============================================================================

def validate_constellation_energies() -> pd.DataFrame:
    """
    Mandatory Syllabus Validation:
    'Report numerical average constellation energy before and after normalization.'
    """
    configs = [
        ("QPSK (4-PSK)", 4, "PSK", 2),
        ("8-PSK", 8, "PSK", 3),
        ("16-QAM", 16, "QAM", 4),
        ("64-QAM", 64, "QAM", 6)
    ]

    records = []
    for name, order, mod_type, bits in configs:
        if mod_type == "PSK":
            unnorm_s, norm_s, _, e_unnorm, e_norm = get_mpsk_constellation(order)
            th_unnorm = 1.0
        else:
            unnorm_s, norm_s, _, e_unnorm, e_norm = get_mqam_constellation(order, use_gray=True)
            th_unnorm = 2.0 * (order - 1.0) / 3.0

        # Numerical confirmation
        num_unnorm = float(np.mean(np.abs(unnorm_s) ** 2))
        num_norm = float(np.mean(np.abs(norm_s) ** 2))
        scale_factor = 1.0 / np.sqrt(num_unnorm)

        records.append({
            "Modulation": name,
            "Order M": order,
            "Bits/Sym (k)": bits,
            "Theoretical Unnorm E": th_unnorm,
            "Numerical Unnorm E": num_unnorm,
            "Scale Factor": scale_factor,
            "Normalized E (Es)": num_norm,
            "Status": "PASSED" if np.isclose(num_norm, 1.0, atol=1e-8) else "FAILED"
        })

    return pd.DataFrame(records)


# =============================================================================
# 6. MONTE CARLO SER & BER SIMULATION
# =============================================================================

def simulate_ser_ber(m_order: int, mod_type: str, eb_no_db_range: np.ndarray,
                     use_gray: bool = True, n_symbols: int = 40000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Symbol Error Rate (SER) and Bit Error Rate (BER) across Eb/N0.
    """
    k = int(np.log2(m_order))

    if mod_type == "PSK":
        _, _, mapping, _, _ = get_mpsk_constellation(m_order)
    else:
        _, _, mapping, _, _ = get_mqam_constellation(m_order, use_gray=use_gray)

    bit_strings = list(mapping.keys())
    constellation = np.array(list(mapping.values()))

    # Generate random symbols
    rng = np.random.default_rng(42)
    sym_indices = rng.integers(0, m_order, size=n_symbols)
    tx_symbols = constellation[sym_indices]

    # Transmitted bit stream
    tx_bits = np.empty(n_symbols * k, dtype=int)
    for i, idx in enumerate(sym_indices):
        b_str = bit_strings[idx]
        for b_pos in range(k):
            tx_bits[i * k + b_pos] = int(b_str[b_pos])

    ser_arr = np.zeros(len(eb_no_db_range))
    ber_arr = np.zeros(len(eb_no_db_range))

    for s_idx, snr_db in enumerate(eb_no_db_range):
        rx = add_awgn_complex(tx_symbols, snr_db, bits_per_symbol=k, seed=1000 + s_idx)
        rx_bits, rx_syms = demodulate_minimum_distance(rx, mapping)

        ser_arr[s_idx] = np.mean(rx_syms != tx_symbols)
        ber_arr[s_idx] = np.mean(rx_bits != tx_bits)

    return ser_arr, ber_arr


def theoretical_ser_ber(m_order: int, mod_type: str, eb_no_db_range: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Theoretical SER and BER approximations from digital communications theory:
      - M-PSK:
        P_s approx 2 * Q(sqrt(2 * k * Eb/N0) * sin(pi / M))
        P_b approx P_s / k (with Gray coding)
      - Square M-QAM:
        P_sqrtM = 2 * (1 - 1/sqrt(M)) * Q(sqrt(3 * k * Eb/N0 / (M - 1)))
        P_s = 1 - (1 - P_sqrtM)^2
        P_b approx P_s / k (with Gray coding)
    """
    eb_no_lin = 10.0 ** (eb_no_db_range / 10.0)
    k = int(np.log2(m_order))

    def q_func(x):
        return 0.5 * sp.erfc(x / np.sqrt(2.0))

    if mod_type == "PSK":
        arg = np.sqrt(2.0 * k * eb_no_lin) * np.sin(np.pi / m_order)
        ser = 2.0 * q_func(arg)
        ber = ser / k
    else:
        sqrt_m = int(np.sqrt(m_order))
        arg = np.sqrt((3.0 * k * eb_no_lin) / (m_order - 1.0))
        p_sub = 2.0 * (1.0 - 1.0 / sqrt_m) * q_func(arg)
        ser = 1.0 - (1.0 - p_sub) ** 2
        ber = ser / k

    return ser, ber


# =============================================================================
# 7. PUBLICATION VISUALIZATIONS (SAVED TO OUTPUTS/)
# =============================================================================

def plot_ideal_constellations(filename: str = "1_ideal_constellations_all_orders.png"):
    """Figure 1: Ideal normalized constellations for QPSK, 8-PSK, 16-QAM, and 64-QAM."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))

    # (a) QPSK
    _, qpsk_s, qpsk_map, _, _ = get_mpsk_constellation(4)
    axes[0, 0].scatter(np.real(qpsk_s), np.imag(qpsk_s), color="royalblue", s=80, edgecolors="black", linewidth=1.5)
    for b_str, sym in qpsk_map.items():
        axes[0, 0].annotate(f"'{b_str}'", (np.real(sym), np.imag(sym)), textcoords="offset points",
                            xytext=(10, 10), fontsize=10, fontweight="bold", color="darkblue")
    axes[0, 0].set_title(r"QPSK (4-PSK) Constellation ($E_s = 1.0, \eta = 2$ b/s/Hz)")

    # (b) 8-PSK
    _, psk8_s, psk8_map, _, _ = get_mpsk_constellation(8)
    axes[0, 1].scatter(np.real(psk8_s), np.imag(psk8_s), color="forestgreen", s=80, edgecolors="black", linewidth=1.5)
    for b_str, sym in psk8_map.items():
        axes[0, 1].annotate(f"'{b_str}'", (np.real(sym), np.imag(sym)), textcoords="offset points",
                            xytext=(8, 8), fontsize=9, fontweight="bold", color="darkgreen")
    axes[0, 1].set_title(r"8-PSK Constellation ($E_s = 1.0, \eta = 3$ b/s/Hz)")

    # (c) 16-QAM
    _, qam16_s, qam16_map, _, _ = get_mqam_constellation(16, use_gray=True)
    axes[1, 0].scatter(np.real(qam16_s), np.imag(qam16_s), color="crimson", s=70, edgecolors="black", linewidth=1.2)
    for b_str, sym in qam16_map.items():
        axes[1, 0].annotate(f"'{b_str}'", (np.real(sym), np.imag(sym)), textcoords="offset points",
                            xytext=(5, 5), fontsize=8, fontweight="bold", color="maroon")
    axes[1, 0].set_title(r"16-QAM Gray Constellation ($E_s = 1.0, \eta = 4$ b/s/Hz)")

    # (d) 64-QAM
    _, qam64_s, _, _, _ = get_mqam_constellation(64, use_gray=True)
    axes[1, 1].scatter(np.real(qam64_s), np.imag(qam64_s), color="purple", s=30, alpha=0.9, edgecolors="black", linewidth=0.8)
    axes[1, 1].set_title(r"64-QAM Gray Constellation ($E_s = 1.0, \eta = 6$ b/s/Hz)")

    for row in axes:
        for ax in row:
            ax.axhline(0, color="black", linestyle="--", alpha=0.5)
            ax.axvline(0, color="black", linestyle="--", alpha=0.5)
            circle = plt.Circle((0, 0), 1.0, color="gray", fill=False, linestyle=":", alpha=0.7)
            ax.add_artist(circle)
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.8, 1.8)
            ax.set_xlabel("In-Phase (I)")
            ax.set_ylabel("Quadrature (Q)")
            ax.grid(True)
            ax.set_aspect("equal")

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_received_clouds_snr(filename: str = "2_received_constellation_clouds_snr_sweep.png"):
    """Figure 2: Received constellation clouds at 0 dB, 5 dB, 10 dB, and 20 dB SNR."""
    _, qam16_s, _, _, _ = get_mqam_constellation(16)
    _, psk8_s, _, _, _ = get_mpsk_constellation(8)

    snr_levels = [0.0, 5.0, 10.0, 20.0]
    n_pts = 2500

    rng = np.random.default_rng(101)
    qam16_stream = qam16_s[rng.integers(0, 16, size=n_pts)]
    psk8_stream = psk8_s[rng.integers(0, 8, size=n_pts)]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))

    for i, snr in enumerate(snr_levels):
        # 16-QAM
        rx_qam = add_awgn_complex(qam16_stream, snr, bits_per_symbol=4, seed=200 + i)
        axes[0, i].scatter(np.real(rx_qam), np.imag(rx_qam), color="crimson", alpha=0.25, s=6)
        axes[0, i].scatter(np.real(qam16_s), np.imag(qam16_s), color="black", marker="+", s=60, linewidth=2)
        axes[0, i].set_title(f"16-QAM Cloud ($E_b/N_0 = {snr:.0f}$ dB)")

        # 8-PSK
        rx_psk = add_awgn_complex(psk8_stream, snr, bits_per_symbol=3, seed=300 + i)
        axes[1, i].scatter(np.real(rx_psk), np.imag(rx_psk), color="forestgreen", alpha=0.25, s=6)
        axes[1, i].scatter(np.real(psk8_s), np.imag(psk8_s), color="black", marker="+", s=60, linewidth=2)
        axes[1, i].set_title(f"8-PSK Cloud ($E_b/N_0 = {snr:.0f}$ dB)")

    for row in axes:
        for ax in row:
            ax.axhline(0, color="black", linestyle="--", alpha=0.5)
            ax.axvline(0, color="black", linestyle="--", alpha=0.5)
            ax.set_xlim(-1.8, 1.8)
            ax.set_ylim(-1.8, 1.8)
            ax.set_xlabel("In-Phase (I)")
            ax.set_ylabel("Quadrature (Q)")
            ax.grid(True)
            ax.set_aspect("equal")

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_decision_regions(filename: str = "3_decision_regions_and_voronoi_boundaries.png"):
    """Figure 3: 2D Minimum Euclidean distance decision regions for 8-PSK and 16-QAM."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))

    # Grid for background classification
    grid_res = 200
    x_lin = np.linspace(-1.8, 1.8, grid_res)
    y_lin = np.linspace(-1.8, 1.8, grid_res)
    xx, yy = np.meshgrid(x_lin, y_lin)
    grid_pts = xx + 1j * yy
    grid_flat = grid_pts.flatten()

    # 1. 8-PSK Decision Regions
    _, psk8_s, _, _, _ = get_mpsk_constellation(8)
    dists_psk = np.abs(grid_flat[:, np.newaxis] - psk8_s[np.newaxis, :]) ** 2
    classes_psk = np.argmin(dists_psk, axis=1).reshape(xx.shape)

    ax1.contourf(xx, yy, classes_psk, levels=np.arange(9) - 0.5, cmap="tab10", alpha=0.35)
    ax1.scatter(np.real(psk8_s), np.imag(psk8_s), color="black", s=70, zorder=5)
    for m in range(8):
        # Draw radial boundary lines
        angle = (2 * m) * np.pi / 8
        ax1.plot([0, 1.8 * np.cos(angle)], [0, 1.8 * np.sin(angle)], "k--", linewidth=1.0, alpha=0.7)
    ax1.set_title("8-PSK Decision Regions (Pies / Wedges)", fontsize=13)
    ax1.set_xlabel("In-Phase (I)")
    ax1.set_ylabel("Quadrature (Q)")
    ax1.set_xlim(-1.8, 1.8); ax1.set_ylim(-1.8, 1.8); ax1.grid(True); ax1.set_aspect("equal")

    # 2. 16-QAM Decision Regions
    _, qam16_s, _, _, _ = get_mqam_constellation(16)
    dists_qam = np.abs(grid_flat[:, np.newaxis] - qam16_s[np.newaxis, :]) ** 2
    classes_qam = np.argmin(dists_qam, axis=1).reshape(xx.shape)

    ax2.contourf(xx, yy, classes_qam, levels=np.arange(17) - 0.5, cmap="tab20", alpha=0.35)
    ax2.scatter(np.real(qam16_s), np.imag(qam16_s), color="black", s=60, zorder=5)
    # Decision grid lines for square QAM
    c_norm = 1.0 / np.sqrt(10.0)
    for boundary in [-2.0 * c_norm, 0.0, 2.0 * c_norm]:
        ax2.axvline(boundary, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
        ax2.axhline(boundary, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.set_title("16-QAM Decision Regions (Square Voronoi Cells)", fontsize=13)
    ax2.set_xlabel("In-Phase (I)")
    ax2.set_ylabel("Quadrature (Q)")
    ax2.set_xlim(-1.8, 1.8); ax2.set_ylim(-1.8, 1.8); ax2.grid(True); ax2.set_aspect("equal")

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_gray_vs_binary_comparison(filename: str = "4_gray_vs_binary_mapping_comparison.png"):
    """Figure 4: Gray vs Natural (Binary) mapping bit assignment and resulting BER curves."""
    eb_no_db = np.linspace(0.0, 14.0, 15)

    print("Simulating Gray vs Binary mapping on 16-QAM (50,000 symbols)...")
    ser_gray, ber_gray = simulate_ser_ber(16, "QAM", eb_no_db, use_gray=True, n_symbols=50000)
    ser_bin, ber_bin = simulate_ser_ber(16, "QAM", eb_no_db, use_gray=False, n_symbols=50000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Diagram showing difference on 1D slice
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10.0)
    gray_labels = ["00", "01", "11", "10"]
    bin_labels = ["00", "01", "10", "11"]

    ax1.axhline(0, color="black", linewidth=1.5)
    ax1.scatter(levels, [0]*4, color="crimson", s=150, zorder=5)
    for i, x in enumerate(levels):
        ax1.annotate(f"Gray: '{gray_labels[i]}'\nBinary: '{bin_labels[i]}'", (x, 0), textcoords="offset points",
                     xytext=(0, 25 if i % 2 == 0 else -40), ha="center", fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))
    ax1.set_title("1D Slice Bit Assignments (Levels: -3, -1, +1, +3)", fontsize=12)
    ax1.set_xlabel("In-Phase Axis Coordinate")
    ax1.set_xlim(-1.5, 1.5); ax1.set_ylim(-0.3, 0.3)
    ax1.set_yticks([])
    ax1.grid(True, linestyle="--", alpha=0.5)

    # (b) Semilog BER curves
    ax2.semilogy(eb_no_db, ser_gray, "k--", linewidth=1.8, label="Symbol Error Rate SER (Identical)")
    ax2.semilogy(eb_no_db, ber_gray, "ro-", linewidth=1.8, label="BER with Gray Mapping ($P_b \\approx P_s / 4$)")
    ax2.semilogy(eb_no_db, ber_bin, "bs-", linewidth=1.8, label="BER with Binary Mapping (Higher Errors)")

    ax2.set_title("16-QAM BER: Gray vs Binary Mapping over AWGN", fontsize=12)
    ax2.set_xlabel(r"Energy per Bit to Noise Spectral Density $E_b/N_0$ (dB)", fontsize=11)
    ax2.set_ylabel("Error Probability", fontsize=11)
    ax2.set_xlim(0, 14); ax2.set_ylim(1e-5, 1.0)
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.legend(loc="lower left", framealpha=0.95)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_ber_ser_curves(filename: str = "5_ber_ser_curves_mpsk_mqam.png"):
    """Figure 5: Publication-quality semilog BER and SER curves comparing M-PSK and M-QAM."""
    eb_no_db = np.linspace(0.0, 16.0, 17)
    eb_fine = np.linspace(0.0, 16.0, 100)

    print("Running Monte Carlo simulations for QPSK, 8-PSK, 16-QAM, 64-QAM...")
    _, ber_qpsk = simulate_ser_ber(4, "PSK", eb_no_db, use_gray=True, n_symbols=40000)
    _, ber_psk8 = simulate_ser_ber(8, "PSK", eb_no_db, use_gray=True, n_symbols=40000)
    _, ber_qam16 = simulate_ser_ber(16, "QAM", eb_no_db, use_gray=True, n_symbols=40000)
    _, ber_qam64 = simulate_ser_ber(64, "QAM", eb_no_db, use_gray=True, n_symbols=40000)

    # Theoretical curves
    _, th_qpsk = theoretical_ser_ber(4, "PSK", eb_fine)
    _, th_psk8 = theoretical_ser_ber(8, "PSK", eb_fine)
    _, th_qam16 = theoretical_ser_ber(16, "QAM", eb_fine)
    _, th_qam64 = theoretical_ser_ber(64, "QAM", eb_fine)

    fig, ax = plt.subplots(figsize=(11, 7.5))

    # QPSK
    ax.semilogy(eb_fine, th_qpsk, "b-", linewidth=1.8, label="QPSK (Theory)")
    ax.semilogy(eb_no_db, ber_qpsk, "bo", markersize=6, fillstyle="none", markeredgewidth=1.8, label="QPSK (Sim)")

    # 8-PSK
    ax.semilogy(eb_fine, th_psk8, "g-", linewidth=1.8, label="8-PSK (Theory)")
    ax.semilogy(eb_no_db, ber_psk8, "g^", markersize=6, fillstyle="none", markeredgewidth=1.8, label="8-PSK (Sim)")

    # 16-QAM
    ax.semilogy(eb_fine, th_qam16, "r-", linewidth=1.8, label="16-QAM (Theory)")
    ax.semilogy(eb_no_db, ber_qam16, "rs", markersize=6, fillstyle="none", markeredgewidth=1.8, label="16-QAM (Sim)")

    # 64-QAM
    ax.semilogy(eb_fine, th_qam64, "m-", linewidth=1.8, label="64-QAM (Theory)")
    ax.semilogy(eb_no_db, ber_qam64, "md", markersize=6, fillstyle="none", markeredgewidth=1.8, label="64-QAM (Sim)")

    ax.set_title("Bit Error Rate (BER) Comparison for M-PSK and M-QAM over AWGN", fontsize=13)
    ax.set_xlabel(r"Energy per Bit to Noise Spectral Density $E_b/N_0$ (dB)", fontsize=12)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=12)
    ax.set_xlim(0, 16); ax.set_ylim(1e-5, 0.6)
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.95)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_spectral_efficiency_tradeoff(filename: str = "6_ber_vs_spectral_efficiency.png"):
    """Figure 6: Spectral efficiency (bits/s/Hz) vs required Eb/N0 for BER = 1e-3."""
    # Data points for BER = 10^-3
    # QPSK (2 bps/Hz) ~ 6.8 dB
    # 8-PSK (3 bps/Hz) ~ 10.3 dB
    # 16-QAM (4 bps/Hz) ~ 10.5 dB
    # 64-QAM (6 bps/Hz) ~ 14.5 dB
    # 16-PSK (4 bps/Hz) ~ 14.8 dB (to highlight QAM superiority!)

    schemes = [
        ("QPSK", 2, 6.8, "royalblue", "o"),
        ("8-PSK", 3, 10.3, "forestgreen", "^"),
        ("16-QAM", 4, 10.5, "crimson", "s"),
        ("16-PSK", 4, 14.8, "orange", "D"),
        ("64-QAM", 6, 14.5, "purple", "p")
    ]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    # Shannon limit curve: C/B = log2(1 + S/N) = log2(1 + (C/B) * (Eb/N0))
    # Eb/N0_min = (2^eta - 1) / eta
    eta_shannon = np.linspace(0.5, 7.0, 200)
    eb_no_shannon_lin = (2.0 ** eta_shannon - 1.0) / eta_shannon
    eb_no_shannon_db = 10.0 * np.log10(eb_no_shannon_lin)

    ax.plot(eta_shannon, eb_no_shannon_db, "k--", linewidth=2.0, label="Shannon Capacity Limit ($C/B = \\log_2(1 + \\text{SNR})$)")
    ax.axhline(-1.59, color="gray", linestyle=":", label="Ultimate Shannon Limit (-1.59 dB)")

    for name, eta, req_ebno, col, marker in schemes:
        ax.scatter(eta, req_ebno, color=col, marker=marker, s=120, edgecolors="black", linewidth=1.5, zorder=5)
        ax.annotate(f"{name}\n({req_ebno:.1f} dB)", (eta, req_ebno), textcoords="offset points",
                    xytext=(15, -5 if "16-PSK" in name else 10),
                    fontsize=10, fontweight="bold", color=col,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=1.2))

    # Highlight 16-QAM vs 16-PSK gap
    ax.annotate(r"$\approx 4.3$ dB Power Savings" + "\nof 16-QAM over 16-PSK!",
                xy=(4, 10.5), xytext=(4.3, 12.3),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
                fontsize=10, fontweight="bold", color="darkred")

    ax.set_title(r"Bandwidth Efficiency ($\eta$) vs Required $E_b/N_0$ at $\text{BER} = 10^{-3}$", fontsize=13)
    ax.set_xlabel(r"Spectral Efficiency $\eta = \log_2(M)$ (bits/s/Hz)", fontsize=12)
    ax.set_ylabel(r"Required $E_b/N_0$ (dB) for $\text{BER} = 10^{-3}$", fontsize=12)
    ax.set_xlim(1.0, 7.0)
    ax.set_ylim(-3.0, 20.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.95)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_papr_distribution(filename: str = "7_constellation_power_and_peak_to_average_ratio.png"):
    """Figure 7: Symbol power distribution and Peak-to-Average Power Ratio (PAPR)."""
    _, qpsk_s, _, _, _ = get_mpsk_constellation(4)
    _, psk8_s, _, _, _ = get_mpsk_constellation(8)
    _, qam16_s, _, _, _ = get_mqam_constellation(16)
    _, qam64_s, _, _, _ = get_mqam_constellation(64)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Peak to Average Power Ratio comparison
    schemes = ["QPSK", "8-PSK", "16-QAM", "64-QAM"]
    papr_vals = [
        np.max(np.abs(qpsk_s)**2) / np.mean(np.abs(qpsk_s)**2),
        np.max(np.abs(psk8_s)**2) / np.mean(np.abs(psk8_s)**2),
        np.max(np.abs(qam16_s)**2) / np.mean(np.abs(qam16_s)**2),
        np.max(np.abs(qam64_s)**2) / np.mean(np.abs(qam64_s)**2)
    ]
    papr_db = [10.0 * np.log10(v) for v in papr_vals]

    bars = axes[0].bar(schemes, papr_db, color=["royalblue", "forestgreen", "crimson", "purple"], edgecolor="black", width=0.55)
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, h + 0.1, f"{h:.2f} dB", ha="center", va="bottom", fontweight="bold")
    axes[0].set_ylabel("PAPR (dB)")
    axes[0].set_ylim(0, 4.5)
    axes[0].set_title("Peak-to-Average Power Ratio (PAPR) of Constellations", fontsize=12)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.6)

    # (b) Energy ring distribution for 16-QAM
    # 16-QAM has 3 distinct energy rings:
    # Ring 1 (Inner 4 points): (1^2 + 1^2)/10 = 2/10 = 0.2
    # Ring 2 (Middle 8 points): (1^2 + 3^2)/10 = 10/10 = 1.0
    # Ring 3 (Outer 4 points): (3^2 + 3^2)/10 = 18/10 = 1.8
    p_qam16 = np.abs(qam16_s)**2
    unique_p, counts = np.unique(np.round(p_qam16, 4), return_counts=True)

    axes[1].bar([f"Ring {i+1}\n($|s|^2={p:.1f}$)" for i, p in enumerate(unique_p)], counts,
                color="darkorange", edgecolor="black", width=0.45)
    for i, c in enumerate(counts):
        axes[1].text(i, c + 0.2, f"{c} symbols\n({c/16*100:.0f}%)", ha="center", va="bottom", fontweight="bold")
    axes[1].set_ylabel("Number of Symbols")
    axes[1].set_ylim(0, 10)
    axes[1].set_title("16-QAM Distinct Symbol Energy Tiers (3 Concentric Rings)", fontsize=12)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


def plot_mandatory_validation(df_val: pd.DataFrame, filename: str = "8_mandatory_energy_normalization_validation.png"):
    """Figure 8: Graphical display of mandatory constellation energy normalization validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bar chart comparing unnormalized vs normalized energies
    x = np.arange(len(df_val))
    width = 0.35

    ax1.bar(x - width/2, df_val["Numerical Unnorm E"], width, label="Unnormalized Average Energy", color="crimson")
    ax1.bar(x + width/2, df_val["Normalized E (Es)"], width, label="Normalized Average Energy ($E_s = 1.0$)", color="forestgreen")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_val["Modulation"], fontsize=10)
    ax1.set_ylabel("Average Symbol Energy")
    ax1.set_title("Mandatory Validation: Constellation Energy Normalization", fontsize=12)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax1.legend(loc="upper left")

    # Table subplot
    ax2.axis("off")
    table_data = []
    for _, r in df_val.iterrows():
        table_data.append([
            r["Modulation"],
            f"{r['Bits/Sym (k)']}",
            f"{r['Theoretical Unnorm E']:.2f}",
            f"{r['Numerical Unnorm E']:.2f}",
            f"{r['Scale Factor']:.4f}",
            f"{r['Normalized E (Es)']:.6f}",
            r["Status"]
        ])
    col_labels = ["Modulation", "k", "Theory Unnorm", "Num Unnorm", "Scale", "Es (Norm)", "Status"]
    table = ax2.table(cellText=table_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)
    ax2.set_title("Numerical Energy Verification Summary", fontsize=12, pad=15)

    plt.tight_layout()
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath)
    plt.close()
    print(f"Saved: {filepath.name}")


# =============================================================================
# 8. MAIN EXECUTION DRIVER
# =============================================================================

def main():
    print("=" * 80)
    print("Digital Communication Lab - Experiment 11: M-PSK and M-QAM Simulation")
    print("=" * 80)

    # 1. Mandatory Syllabus Validation: Constellation energy calculation
    print("\n--- 1. Mandatory Syllabus Validation: Constellation Energy Normalization ---")
    df_val = validate_constellation_energies()
    print(df_val.to_string(index=False))

    # 2. Generate all publication figures
    print("\n--- 2. Generating Publication Visualizations (outputs/) ---")
    plot_ideal_constellations()
    plot_received_clouds_snr()
    plot_decision_regions()
    plot_gray_vs_binary_comparison()
    plot_papr_distribution()
    plot_spectral_efficiency_tradeoff()
    plot_mandatory_validation(df_val)
    plot_ber_ser_curves()

    print("\n" + "=" * 80)
    print("Experiment 11 simulation completed successfully!")
    print(f"All outputs generated in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
