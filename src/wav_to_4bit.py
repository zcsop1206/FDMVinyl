#!/usr/bin/env python3
"""
WAV -> 4-bit Audio Converter for FDM Vinyl (V3.0)

DSP preprocessing for OpenVinyl groove encoding. Conditions audio for
4-bit quantization within the bandwidth constraints of FDM-printed
groove geometry.

All DSP parameters are derived from groove geometry constraints:
  - HPF cutoff: 300 Hz (bandwidth floor — no harmonics below this
    in the [681, 2042] Hz encoding window)
  - LPF cutoff: geometry-derived inner-radius Nyquist
  - Compression: ≥6:1 ratio (24 dB range → 18 dB target)
  - Noise shaping: TPDF only (Lipshitz disabled — zero Nyquist
    headroom for spectral redistribution at inner radius)
  - Pre-emphasis: PLACEHOLDER (α=0.97) — requires measured H(f)

Outputs a 16-bit WAV containing 4-bit quantized audio.

Usage:
    python wav_to_4bit.py input.wav [output.wav]
    python wav_to_4bit.py input.wav output.wav --nozzle 0.2
    python wav_to_4bit.py input.wav output.wav --rpm 78

Dependencies: numpy, scipy (optional for better filtering), wave, struct
"""

import wave
import struct
import numpy as np
import sys
import os
import argparse
import time

try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Note: scipy not found. Install with 'pip install scipy' for better filtering and 50x speed boost.")


# =============================================================================
# AUDIO I/O
# =============================================================================

def read_wav(path: str):
    """Read WAV file, return (samples float64 [-1,1], framerate, n_channels)."""
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        s = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
        s /= 128.0
    elif sampwidth == 2:
        s = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        s /= 32768.0
    elif sampwidth == 3:
        n = len(raw) // 3
        s = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s[i] = int.from_bytes(raw[i*3:i*3+3], 'little', signed=True) / 8388608.0
    elif sampwidth == 4:
        s = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
        s /= 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        s = s.reshape(-1, n_channels)
    
    print(f"  Loaded: {framerate}Hz, {n_channels}ch, {sampwidth*8}bit, {n_frames/framerate:.2f}s")
    return s, framerate, n_channels


def write_wav(path: str, samples: np.ndarray, framerate: int, n_channels: int):
    """Write float64 samples [-1,1] to 16-bit WAV."""
    # Normalize to prevent clipping
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak * 0.95

    # Convert to int16
    int_samples = np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)  # 16-bit container
        wf.setframerate(framerate)
        wf.writeframes(int_samples.tobytes())

    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {path} ({size_kb:.0f} KB)")


# =============================================================================
# PROCESSING
# =============================================================================

def apply_pre_emphasis(samples: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """First-order high-pass pre-emphasis: y[n] = x[n] - alpha * x[n-1]

    PLACEHOLDER: α=0.97 is assumed, not measured. Pre-emphasis should be
    the inverse of the measured groove-stylus-cartridge transfer function
    H(f). Characterize by printing a sine sweep test record (300–2042 Hz
    in 50 Hz steps), recording playback output, and computing
    H(f) = Y(f)/X(f).
    """
    out = np.zeros_like(samples)
    out[0] = samples[0]
    out[1:] = samples[1:] - alpha * samples[:-1]
    return out


def apply_de_emphasis(samples: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """
    Inverse of pre-emphasis: first-order IIR low-pass.
    y[n] = x[n] + alpha * y[n-1]
    Restores natural frequency balance after quantization.
    
    V2: Uses scipy.signal.lfilter when available for ~50x speedup.
    """
    if HAS_SCIPY:
        # y[n] = x[n] + alpha * y[n-1]  =>  y[n] - alpha*y[n-1] = x[n]
        # Transfer function: H(z) = 1 / (1 - alpha*z^-1)
        # lfilter(b, a, x) where b=[1], a=[1, -alpha]
        return scipy_signal.lfilter([1.0], [1.0, -alpha], samples)
    else:
        # Fallback: pure Python loop
        out = np.zeros_like(samples)
        out[0] = samples[0]
        for i in range(1, len(samples)):
            out[i] = samples[i] + alpha * out[i-1]
        return out


def apply_highpass(samples: np.ndarray, framerate: int,
                   cutoff_hz: float = 300.0) -> np.ndarray:
    """4th-order Butterworth HPF, zero-phase.

    Cutoff = 300 Hz (derived): content below 300 Hz has no harmonics
    within the encoding bandwidth window [681, 2042] Hz and only
    consumes quantization levels without contributing to perceivable
    audio output.
    """
    if HAS_SCIPY:
        nyquist = framerate / 2.0
        wn = max(cutoff_hz / nyquist, 0.001)
        b, a = scipy_signal.butter(4, wn, btype='high')
        return scipy_signal.filtfilt(b, a, samples)
    else:
        # Fallback: subtract lowpass (crude HPF)
        window = max(1, int(framerate / (cutoff_hz * 2)))
        kernel = np.ones(window) / window
        lowpassed = np.convolve(samples, kernel, mode='same')
        return samples - lowpassed


def apply_lowpass(samples: np.ndarray, framerate: int,
                  cutoff_hz: float = 681.0) -> np.ndarray:
    """4th-order Butterworth LPF, zero-phase.

    Default cutoff = inner-radius Nyquist (geometry-derived).
    Content above this aliases into the passband. The previous 8 kHz
    cutoff was physically meaningless — the groove cannot encode
    content above ~2,042 Hz at any radius and above ~681 Hz at the
    inner radius (0.2 mm nozzle, 78 RPM).
    """
    if HAS_SCIPY:
        nyquist = framerate / 2.0
        wn = min(cutoff_hz / nyquist, 0.99)
        b, a = scipy_signal.butter(4, wn, btype='low')
        return scipy_signal.filtfilt(b, a, samples)
    else:
        window = max(1, int(framerate / (cutoff_hz * 2)))
        kernel = np.ones(window) / window
        return np.convolve(samples, kernel, mode='same')


def apply_compression(samples: np.ndarray, framerate: int,
                      ratio: float = 6.0, attack_ms: float = 2.0,
                      release_ms: float = 50.0,
                      target_dr_db: float = 18.0) -> np.ndarray:
    """Dynamic range compression before quantization.

    4-bit gives 24 dB theoretical dynamic range. Input audio typically
    has 40+ dB dynamic range — quiet passages occupy 2–3 quantization
    levels and are indistinguishable from quantization noise. Compressing
    to ≤18 dB ensures the signal occupies most of the 16 available levels
    throughout. 18 dB target leaves headroom for dithering overshoot.
    """
    # Compute envelope using RMS
    window = max(1, int(framerate * attack_ms / 1000.0))
    sq = samples ** 2
    kernel = np.ones(window) / window
    rms = np.sqrt(np.convolve(sq, kernel, mode='same') + 1e-12)

    # Compute gain reduction
    peak_rms = np.max(rms)
    if peak_rms <= 0:
        return samples

    rms_db = 20.0 * np.log10(rms / peak_rms + 1e-12)
    threshold_db = -target_dr_db

    # Compress above threshold
    gain_db = np.where(
        rms_db > threshold_db,
        rms_db * (1.0 - 1.0/ratio),
        0.0
    )
    gain = 10.0 ** (-gain_db / 20.0)

    # Smooth gain changes (attack/release)
    attack_samples = max(1, int(framerate * attack_ms / 1000.0))
    release_samples = max(1, int(framerate * release_ms / 1000.0))
    alpha_a = 1.0 - np.exp(-1.0 / attack_samples)
    alpha_r = 1.0 - np.exp(-1.0 / release_samples)

    smoothed_gain = np.zeros_like(gain)
    smoothed_gain[0] = gain[0]
    for i in range(1, len(gain)):
        if gain[i] < smoothed_gain[i-1]:
            smoothed_gain[i] = smoothed_gain[i-1] + alpha_a * (gain[i] - smoothed_gain[i-1])
        else:
            smoothed_gain[i] = smoothed_gain[i-1] + alpha_r * (gain[i] - smoothed_gain[i-1])

    compressed = samples * smoothed_gain

    # Normalize
    peak = np.max(np.abs(compressed))
    if peak > 0:
        compressed /= peak

    return compressed


def soft_saturate(samples: np.ndarray, amount: float = 0.2) -> np.ndarray:
    """
    Soft saturation / harmonic exciter.
    Adds even harmonics which give warmth and help mask quantization noise.
    amount=0: bypass. amount=1: heavy saturation.
    Uses tanh waveshaping.
    """
    if amount <= 0:
        return samples
    drive = 1.0 + amount * 4.0  # drive gain before tanh
    saturated = np.tanh(samples * drive) / np.tanh(drive)
    return saturated


def harmonic_exciter(samples: np.ndarray, amount: float = 0.06) -> np.ndarray:
    """
    Post-quantization harmonic exciter.
    Adds a small amount of 2nd and 3rd harmonic content to add warmth
    that psychoacoustically masks quantization noise.
    
    The 2nd harmonic (even) adds warmth/fullness.
    The 3rd harmonic (odd) adds presence/bite.
    """
    if amount <= 0:
        return samples
    # 2nd harmonic (warm, tube-like)
    h2 = samples * samples * np.sign(samples)  # rectified square preserves sign
    # 3rd harmonic (presence)
    h3 = samples ** 3
    # Mix: more 2nd harmonic than 3rd for warmth over harshness
    excited = samples + amount * 0.7 * h2 + amount * 0.3 * h3
    return excited


def adaptive_rms_envelope(samples: np.ndarray, framerate: int, 
                          window_ms: float = 15.0) -> np.ndarray:
    """
    Compute a smoothed RMS envelope for adaptive quantization.
    Returns an envelope array the same length as samples, with values in [0, 1].
    Uses a running RMS window with exponential smoothing.
    """
    window_samples = max(1, int(framerate * window_ms / 1000.0))
    
    # Compute running RMS using convolution (fast)
    sq = samples ** 2
    kernel = np.ones(window_samples) / window_samples
    rms = np.sqrt(np.convolve(sq, kernel, mode='same'))
    
    # Normalize envelope to [0, 1]
    peak_rms = np.max(rms)
    if peak_rms > 0:
        rms /= peak_rms
    
    # Smooth the envelope to avoid rapid gain changes (use exponential smoothing)
    # Attack fast (2ms), release slow (50ms) — like a compressor
    attack_samples = max(1, int(framerate * 0.002))
    release_samples = max(1, int(framerate * 0.050))
    alpha_attack = 1.0 - np.exp(-1.0 / attack_samples)
    alpha_release = 1.0 - np.exp(-1.0 / release_samples)
    
    smoothed = np.zeros_like(rms)
    smoothed[0] = rms[0]
    for i in range(1, len(rms)):
        if rms[i] > smoothed[i-1]:
            smoothed[i] = smoothed[i-1] + alpha_attack * (rms[i] - smoothed[i-1])
        else:
            smoothed[i] = smoothed[i-1] + alpha_release * (rms[i] - smoothed[i-1])
    
    return smoothed


def noise_shaped_quantize_4bit(samples: np.ndarray, order: int = 2,
                                adaptive_env: np.ndarray = None) -> np.ndarray:
    """
    4-bit quantization with noise shaping (error diffusion).
    
    V2 improvements:
    - Second-order noise shaping (Lipshitz-style, two error taps)
      pushes more quantization noise above the audible band
    - Optional adaptive quantization: scales quantization range 
      based on local signal level for better relative resolution
      in quiet passages
    
    Order 1: error feedback = e[n-1]                   (original)
    Order 2: error feedback = 1.5*e[n-1] - 0.5*e[n-2] (Lipshitz)
    
    Input: float64 in [-1, 1]
    Output: float64 in [-1, 1] with 4-bit resolution
    """
    NUM_LEVELS = 16  # 2^4
    out = np.zeros_like(samples)
    n = len(samples)
    
    if order >= 2:
        # Second-order noise shaping (Lipshitz F-weighted coefficients)
        e1 = 0.0  # error[n-1]
        e2 = 0.0  # error[n-2]
        c1 = 1.5   # first-order coefficient
        c2 = -0.5  # second-order coefficient
        
        for i in range(n):
            # Map to level space [0, 15]
            x = (samples[i] + 1.0) * 0.5 * (NUM_LEVELS - 1)
            
            # Apply noise shaping: feed forward weighted error
            x_shaped = x + c1 * e1 + c2 * e2
            
            # Quantize
            q = round(x_shaped)
            q = max(0, min(NUM_LEVELS - 1, q))
            
            # Update error history
            e2 = e1
            e1 = x_shaped - q
            
            # Map back to [-1, 1]
            out[i] = (q / (NUM_LEVELS - 1)) * 2.0 - 1.0
    else:
        # First-order noise shaping (original)
        error = 0.0
        for i in range(n):
            x = (samples[i] + 1.0) * 0.5 * (NUM_LEVELS - 1)
            x_shaped = x + error
            q = round(x_shaped)
            q = max(0, min(NUM_LEVELS - 1, q))
            error = x_shaped - q
            out[i] = (q / (NUM_LEVELS - 1)) * 2.0 - 1.0

    return out


def noise_shaped_quantize_4bit_fast(samples: np.ndarray, order: int = 2,
                                     adaptive_env: np.ndarray = None) -> np.ndarray:
    """
    Noise-shaped 4-bit quantization — optimized version.
    
    The noise shaping loop is inherently sequential (each sample depends on
    previous error), so true vectorization isn't possible. However we 
    optimize by:
    - Pre-computing the level-space mapping with numpy
    - Using local variables instead of array indexing in the hot loop
    - Minimizing Python overhead per iteration
    """
    NUM_LEVELS = 16
    n = len(samples)
    
    # Pre-map to level space [0, 15] with numpy (vectorized)
    x_levels = (samples + 1.0) * 0.5 * (NUM_LEVELS - 1)
    
    out = np.empty(n, dtype=np.float64)
    scale = 2.0 / (NUM_LEVELS - 1)  # pre-compute output scaling
    
    if order >= 2:
        # Second-order Lipshitz noise shaping
        e1 = 0.0
        e2 = 0.0
        
        for i in range(n):
            x_shaped = x_levels[i] + 1.5 * e1 - 0.5 * e2
            
            # Quantize with clamp
            q = int(x_shaped + 0.5)  # faster than round() for positive values
            if q < 0: q = 0
            elif q > 15: q = 15
            
            # Update error history
            e2 = e1
            e1 = x_shaped - q
            
            out[i] = q * scale - 1.0
    else:
        # First-order noise shaping
        error = 0.0
        for i in range(n):
            x_shaped = x_levels[i] + error
            q = int(x_shaped + 0.5)
            if q < 0: q = 0
            elif q > 15: q = 15
            error = x_shaped - q
            out[i] = q * scale - 1.0

    return out


def resample(samples: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """
    Resample audio to target sample rate using scipy.
    For sample-rate reduction (chiptune effect).
    """
    if orig_rate == target_rate:
        return samples
    if not HAS_SCIPY:
        # Simple nearest-neighbor resampling fallback
        ratio = target_rate / orig_rate
        n_out = int(len(samples) * ratio)
        indices = (np.arange(n_out) / ratio).astype(int)
        indices = np.clip(indices, 0, len(samples) - 1)
        return samples[indices]
    
    n_out = int(len(samples) * target_rate / orig_rate)
    return scipy_signal.resample(samples, n_out)


def to_mid_side(left: np.ndarray, right: np.ndarray):
    """Convert L/R stereo to Mid/Side."""
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    return mid, side


def from_mid_side(mid: np.ndarray, side: np.ndarray):
    """Convert Mid/Side back to L/R stereo."""
    left = mid + side
    right = mid - side
    return left, right


def process_channel(samples_1d: np.ndarray, framerate: int, args,
                    is_side: bool = False) -> np.ndarray:
    """
    Full processing chain for a single channel (or mid/side component).

    V3 processing chain (geometry-derived):
    1. HPF at 300 Hz (bandwidth floor)
    2. Soft saturation
    3. Optional sample-rate reduction
    4. LPF at inner-radius Nyquist (geometry-derived)
    5. Dynamic range compression (6:1 → 18 dB target)
    6. Pre-emphasis (PLACEHOLDER α=0.97)
    7. TPDF dithered 4-bit quantization (Lipshitz shaping disabled)
    8. De-emphasis
    9. Harmonic exciter
    """
    working_rate = framerate

    # 1. HPF — remove content below 300 Hz
    # Content below 300 Hz has no harmonics in the encoding bandwidth
    # window [681, 2042] Hz. It consumes quantization levels without
    # contributing to perceivable audio.
    if not args.no_highpass:
        samples_1d = apply_highpass(samples_1d, working_rate, args.highpass_hz)
        print(f"    HPF applied (cutoff={args.highpass_hz:.0f}Hz — "
              f"bandwidth floor)")

    # 2. Soft saturation (before quantization for harmonic content)
    sat_amount = args.saturate
    if is_side:
        sat_amount *= 0.5
    if sat_amount > 0:
        samples_1d = soft_saturate(samples_1d, sat_amount)
        print(f"    Soft saturation applied (amount={sat_amount:.2f})")

    # 3. Optional sample-rate reduction (chiptune effect)
    if args.sample_rate and args.sample_rate < framerate:
        target_rate = args.sample_rate
        samples_1d = resample(samples_1d, framerate, target_rate)
        working_rate = target_rate
        print(f"    Resampled {framerate}Hz -> {target_rate}Hz")

    # 4. LPF — anti-aliasing at geometry-derived Nyquist
    if not args.no_lowpass:
        cutoff = min(args.lowpass_hz, working_rate / 2.0 - 1)
        samples_1d = apply_lowpass(samples_1d, working_rate, cutoff)
        if cutoff > args.f_nyquist_inner:
            print(f"    WARNING: LPF cutoff ({cutoff:.0f}Hz) exceeds "
                  f"inner-radius Nyquist ({args.f_nyquist_inner:.0f}Hz)")
        print(f"    LPF applied (cutoff={cutoff:.0f}Hz — "
              f"geometry-derived Nyquist)")

    # 5. Dynamic range compression
    # 4-bit = 24 dB theoretical. Compress to ≤18 dB so signal occupies
    # most of the 16 levels throughout. 18 dB target leaves headroom
    # for dithering overshoot.
    if not args.no_compression:
        samples_1d = apply_compression(
            samples_1d, working_rate,
            ratio=args.compression_ratio,
            target_dr_db=18.0
        )
        print(f"    Compression applied (ratio={args.compression_ratio:.0f}:1, "
              f"target=18dB)")

    # 6. Pre-emphasis
    # PLACEHOLDER: α=0.97 is not measured. Pre-emphasis should be the
    # inverse of the measured groove-stylus-cartridge transfer function.
    # Characterize from sine sweep test record before final use.
    if not args.no_preemphasis:
        alpha = args.preemphasis_alpha if not is_side else args.preemphasis_alpha * 0.7
        samples_1d = apply_pre_emphasis(samples_1d, alpha)
        peak = np.max(np.abs(samples_1d))
        if peak > 0:
            samples_1d /= peak
        print(f"    Pre-emphasis applied (alpha={alpha:.2f} — PLACEHOLDER)")

    # 7. 4-bit quantization with TPDF dithering
    # Lipshitz noise shaping coefficients (1.5, -0.5) are designed for
    # 44.1 kHz audio with a shaping target above ~15 kHz. At inner-radius
    # Nyquist of {f_nyquist_inner} Hz, the audio band fills the full
    # Nyquist range — zero spectral headroom for noise redistribution.
    # Shaped error aliases back into the passband. TPDF dithering is
    # applied; spectral shaping is disabled.
    # Revisit if adaptive LPF is implemented on outer grooves.
    if not args.no_dither:
        # Force order=0 to disable Lipshitz shaping, use TPDF only
        print(f"    TPDF dithered 4-bit quantization "
              f"(Lipshitz shaping disabled — zero Nyquist headroom)...")
        t0 = time.time()
        # Use naive quantization + TPDF dither (no error feedback)
        NUM_LEVELS = 16
        mapped = (samples_1d + 1.0) * 0.5 * (NUM_LEVELS - 1)
        # TPDF dither: triangular PDF noise
        r1 = np.random.uniform(-0.5, 0.5, len(mapped))
        r2 = np.random.uniform(-0.5, 0.5, len(mapped))
        mapped = mapped + r1 + r2
        quantized = np.round(np.clip(mapped, 0, NUM_LEVELS - 1))
        samples_1d = (quantized / (NUM_LEVELS - 1)) * 2.0 - 1.0
        elapsed = time.time() - t0
        print(f"    Quantization took {elapsed:.2f}s")
    else:
        print(f"    Naive 4-bit quantization (no dithering)...")
        NUM_LEVELS = 16
        mapped = (samples_1d + 1.0) * 0.5 * (NUM_LEVELS - 1)
        quantized = np.round(np.clip(mapped, 0, NUM_LEVELS - 1))
        samples_1d = (quantized / (NUM_LEVELS - 1)) * 2.0 - 1.0

    # 8. De-emphasis (restore frequency balance)
    if not args.no_preemphasis:
        alpha = args.preemphasis_alpha if not is_side else args.preemphasis_alpha * 0.7
        samples_1d = apply_de_emphasis(samples_1d, alpha)
        peak = np.max(np.abs(samples_1d))
        if peak > 0:
            samples_1d /= peak
        print(f"    De-emphasis applied")

    # 9. Harmonic exciter
    if args.exciter > 0:
        samples_1d = harmonic_exciter(samples_1d, args.exciter)
        print(f"    Harmonic exciter applied (amount={args.exciter:.2f})")

    # 10. Upsample back if downsampled
    if args.sample_rate and args.sample_rate < framerate:
        samples_1d = resample(samples_1d, args.sample_rate, framerate)
        print(f"    Resampled {args.sample_rate}Hz -> {framerate}Hz")

    return samples_1d


def process_stereo_midside(left: np.ndarray, right: np.ndarray,
                           framerate: int, args) -> tuple:
    """
    Process stereo audio using Mid/Side encoding.
    
    Mid channel gets the full processing chain (it carries the melody/rhythm).
    Side channel gets gentler treatment (less saturation, milder emphasis) 
    to preserve stereo imaging without adding noise to the stereo field.
    """
    print("  Converting to Mid/Side...")
    mid, side = to_mid_side(left, right)
    
    print("  Processing Mid channel:")
    mid_out = process_channel(mid, framerate, args, is_side=False)
    
    print("  Processing Side channel:")
    side_out = process_channel(side, framerate, args, is_side=True)
    
    print("  Converting Mid/Side -> L/R...")
    left_out, right_out = from_mid_side(mid_out, side_out)
    
    return left_out, right_out


# =============================================================================
# MAIN
# =============================================================================

def main():
    import math as _math

    parser = argparse.ArgumentParser(
        description='WAV to 4-bit audio for FDM vinyl (V3.0)',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='Input WAV file')
    parser.add_argument('output', nargs='?',
                        help='Output WAV (default: input_4bit.wav)')

    # Groove geometry (for deriving DSP parameters)
    parser.add_argument('--nozzle', type=float, default=0.4,
                        help='Nozzle diameter in mm (default: 0.4)')
    parser.add_argument('--rpm', type=float, default=78.0,
                        help='Turntable RPM (default: 78)')
    parser.add_argument('--r-inner', type=float, default=40.0,
                        help='Inner groove radius in mm (default: 40)')

    # DSP controls
    parser.add_argument('--no-dither', action='store_true',
                        help='Disable TPDF dithering')
    parser.add_argument('--no-preemphasis', action='store_true',
                        help='Disable pre/de-emphasis')
    parser.add_argument('--no-lowpass', action='store_true',
                        help='Disable LPF')
    parser.add_argument('--no-highpass', action='store_true',
                        help='Disable HPF')
    parser.add_argument('--no-compression', action='store_true',
                        help='Disable dynamic range compression')
    parser.add_argument('--lowpass-hz', type=float, default=None,
                        help='LPF cutoff Hz (default: geometry-derived)')
    parser.add_argument('--highpass-hz', type=float, default=300.0,
                        help='HPF cutoff Hz (default: 300, bandwidth floor)')
    parser.add_argument('--preemphasis-alpha', type=float, default=0.97,
                        help='Pre-emphasis alpha (default: 0.97, PLACEHOLDER)')
    parser.add_argument('--compression-ratio', type=float, default=6.0,
                        help='Compression ratio (default: 6.0)')
    parser.add_argument('--saturate', type=float, default=0.15,
                        help='Soft saturation amount (default: 0.15)')
    parser.add_argument('--exciter', type=float, default=0.06,
                        help='Harmonic exciter amount (default: 0.06)')
    parser.add_argument('--mono', action='store_true',
                        help='Force mono output')
    parser.add_argument('--no-midside', action='store_true',
                        help='Disable Mid/Side stereo processing')
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Downsample before quantizing')
    parser.add_argument('--chiptune', action='store_true',
                        help='Preset: lo-fi (11025Hz, 4kHz LP, heavy sat)')
    # Kept for backward compat but ignored
    parser.add_argument('--noise-shaping-order', type=int, default=0,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Compute geometry-derived DSP parameters
    w_e = 1.2 * args.nozzle
    steps_inner = int(2.0 * _math.pi * args.r_inner / w_e)
    f_s_inner = steps_inner * args.rpm / 60.0
    f_nyquist_inner = f_s_inner / 2.0
    args.f_nyquist_inner = f_nyquist_inner

    # Default LPF to inner-radius Nyquist if not set
    if args.lowpass_hz is None:
        args.lowpass_hz = f_nyquist_inner

    # Chiptune preset
    if args.chiptune:
        if args.sample_rate is None:
            args.sample_rate = 11025
        if args.lowpass_hz == f_nyquist_inner:
            args.lowpass_hz = 4000.0
        if args.saturate == 0.15:
            args.saturate = 0.35
        if args.exciter == 0.06:
            args.exciter = 0.10

    in_path = args.input
    out_path = args.output or os.path.splitext(in_path)[0] + '_4bit.wav'

    print("=" * 60)
    print("WAV -> 4-bit Audio for FDM Vinyl (V3.0)")
    print("=" * 60)

    # Print geometry-derived parameters
    print(f"\n  Geometry-derived DSP parameters:")
    print(f"    Nozzle:           {args.nozzle} mm")
    print(f"    Extrusion width:  {w_e:.2f} mm")
    print(f"    Steps/rev (inner): {steps_inner}")
    print(f"    HPF cutoff:       {args.highpass_hz:.0f} Hz "
          f"(bandwidth floor)")
    print(f"    LPF cutoff:       {args.lowpass_hz:.1f} Hz "
          f"(inner-radius Nyquist — geometry-derived)")
    print(f"    Compression:      {args.compression_ratio:.0f}:1 "
          f"(24 dB range -> 18 dB target)")
    print(f"    Noise shaping:    TPDF only "
          f"(Lipshitz disabled — zero headroom at "
          f"inner Nyquist)")
    print(f"    Pre-emphasis:     PLACEHOLDER (alpha={args.preemphasis_alpha}) "
          f"— characterize from test record")

    print("\n[1/3] Loading audio...")
    samples, framerate, n_channels = read_wav(in_path)

    if args.mono and n_channels > 1:
        samples = samples.mean(axis=1)
        n_channels = 1
        print("  Converted to mono")

    print("\n[2/3] Processing...")
    t_start = time.time()

    if n_channels == 1:
        samples_out = process_channel(samples, framerate, args)
    elif n_channels == 2 and not args.no_midside:
        left_out, right_out = process_stereo_midside(
            samples[:, 0], samples[:, 1], framerate, args)
        samples_out = np.stack([left_out, right_out], axis=1)
    else:
        channels_out = []
        for ch in range(n_channels):
            print(f"  Channel {ch+1}/{n_channels}:")
            channels_out.append(
                process_channel(samples[:, ch], framerate, args))
        samples_out = np.stack(channels_out, axis=1)

    t_elapsed = time.time() - t_start
    print(f"\n  Processing time: {t_elapsed:.2f}s")

    print("\n[3/3] Saving...")
    write_wav(out_path, samples_out, framerate, n_channels)

    print("\n" + "=" * 60)
    print(f"  Input:  {in_path}")
    print(f"  Output: {out_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()