#!/usr/bin/env python3
"""
WAV → 3D-Printable STL Macro Audio Record Generator (V7.0)

Improvements over V6.1:
  - Configurable RPM (supports 33, 45, 78 RPM)
  - Reduced Gaussian smoothing (sigma configurable, default 0.8)
  - Treble pre-emphasis filter before quantization
  - Dithering before quantization to break up static quantization noise
  - STEPS_PER_REV scales with RPM for consistent angular resolution
  - All geometry derived from RPM at runtime

Usage:
    python wav_to_record.py input.wav [output.stl] [--rpm 45] [--duration 15]

Dependencies: wave, struct, math, numpy, sys, os, argparse
"""

import wave
import struct
import math
import numpy as np
import sys
import os
import argparse


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """All physical parameters in millimeters."""

    def __init__(self, rpm: float = 45.0, duration: float = 15.0):
        # -- Turntable --
        self.RPM = rpm
        self.REV_PER_SEC = rpm / 60.0
        self.ANGULAR_VELOCITY = self.REV_PER_SEC * 2.0 * math.pi

        # -- Target duration --
        self.TARGET_DURATION = duration

        # -- Record geometry (mm) --
        self.RECORD_DIAMETER_MM = 254.0
        self.OUTER_GROOVE_RADIUS = 120.0
        self.INNER_GROOVE_RADIUS = 40.0
        self.RECORD_OUTER_RADIUS = 127.0
        self.SPINDLE_HOLE_RADIUS = 3.62
        self.BASE_HEIGHT = 1.0

        # -- FDM constraints --
        self.NOZZLE_DIAMETER = 0.40
        self.LAYER_HEIGHT = 0.08

        # -- Groove parameters --
        self.GROOVE_PITCH = 0.55
        self.GROOVE_WIDTH = 0.42

        # -- Grid Topology --
        # Scale steps per rev with RPM so angular resolution stays consistent
        # At 45 RPM: 360 steps/rev. At 78 RPM: ~624. At 33 RPM: ~264.
        self.STEPS_PER_REV = int(360 * (rpm / 45.0))
        self.RAD_PER_STEP = (2.0 * math.pi) / self.STEPS_PER_REV

        # -- Audio Processing --
        # Lower sigma = less smoothing = more high frequency content
        # V6 used 2.5 which was too aggressive. 0.8 preserves more highs.
        self.SMOOTH_SIGMA = 0.8
        self.BIT_DEPTH = 4
        self.NUM_LEVELS = 2 ** self.BIT_DEPTH   # 16
        self.MIN_THICKNESS_LAYERS = 2

        # -- Pre-emphasis --
        # Boosts high frequencies before encoding to compensate for
        # mechanical rolloff in the cartridge/groove system.
        # Standard vinyl uses 3.18ms / 75us RIAA curve; we use a simpler 1st-order shelf.
        self.PRE_EMPHASIS = True
        self.PRE_EMPHASIS_ALPHA = 0.97  # Higher = more treble boost (0.95-0.99)

        # -- Dithering --
        # Adds low-level noise before quantization to break up harmonic
        # distortion patterns that cause static. Triangular PDF dither.
        self.DITHER = True
        self.DITHER_AMPLITUDE = 0.5  # In quantization steps (0.3-1.0)

        # -- Derived --
        self.RADIAL_TRAVEL = self.OUTER_GROOVE_RADIUS - self.INNER_GROOVE_RADIUS
        self.K = self.GROOVE_PITCH / (2.0 * math.pi)

    def summary(self):
        total_revs = self.TARGET_DURATION * self.REV_PER_SEC
        total_steps = int(total_revs * self.STEPS_PER_REV)
        groove_count = total_revs
        radial_used = groove_count * self.GROOVE_PITCH
        print(f"  RPM:            {self.RPM}")
        print(f"  Duration:       {self.TARGET_DURATION}s")
        print(f"  Total revs:     {total_revs:.1f}")
        print(f"  Steps/rev:      {self.STEPS_PER_REV}")
        print(f"  Total steps:    {total_steps:,}")
        print(f"  Grooves:        {groove_count:.1f}")
        print(f"  Radial used:    {radial_used:.1f} mm / {self.RADIAL_TRAVEL:.1f} mm available")
        if radial_used > self.RADIAL_TRAVEL:
            print(f"  WARNING: Duration too long for this RPM — grooves will overlap!")
            print(f"  Max duration at {self.RPM} RPM: {self.RADIAL_TRAVEL / (self.GROOVE_PITCH * self.REV_PER_SEC):.1f}s")


# =============================================================================
# AUDIO PROCESSING
# =============================================================================

def apply_pre_emphasis(samples: np.ndarray, alpha: float) -> np.ndarray:
    """
    First-order high-pass pre-emphasis filter: y[n] = x[n] - alpha * x[n-1]
    Boosts high frequencies to compensate for mechanical rolloff.
    Standard in vinyl mastering (simplified RIAA).
    """
    emphasized = np.zeros_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - alpha * samples[:-1]
    # Renormalize after emphasis
    peak = np.max(np.abs(emphasized))
    if peak > 0:
        emphasized /= peak
    return emphasized


def apply_triangular_dither(quantized: np.ndarray, amplitude: float) -> np.ndarray:
    """
    Triangular PDF dither: sum of two uniform distributions.
    Breaks up correlated quantization error patterns that manifest as static.
    Applied before final rounding.
    """
    r1 = np.random.uniform(-amplitude, amplitude, len(quantized))
    r2 = np.random.uniform(-amplitude, amplitude, len(quantized))
    return quantized + r1 + r2


def load_and_process_audio(wav_path: str, config: Config) -> np.ndarray:
    total_revs = config.TARGET_DURATION * config.REV_PER_SEC
    total_steps = int(total_revs * config.STEPS_PER_REV)
    print(f"  Target: {config.TARGET_DURATION}s @ {config.RPM} RPM -> {total_steps:,} steps")

    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        file_duration = n_frames / framerate
        frames_to_read = int(min(file_duration, config.TARGET_DURATION) * framerate)
        raw = wf.readframes(frames_to_read)
        print(f"  WAV: {framerate}Hz, {n_channels}ch, {sampwidth*8}bit, {file_duration:.1f}s")

    # Decode to float64 [-1, 1]
    if sampwidth == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0
        samples /= 128.0
    elif sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        samples /= 32768.0
    elif sampwidth == 3:
        n = len(raw) // 3
        samples = np.zeros(n, dtype=np.float64)
        for i in range(n):
            samples[i] = int.from_bytes(raw[i*3:i*3+3], 'little', signed=True) / 8388608.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
        samples /= 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Mix to mono
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    # Normalize
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak

    # Pad if audio shorter than target
    audio_dur = len(samples) / framerate
    if audio_dur < config.TARGET_DURATION:
        pad = int((config.TARGET_DURATION - audio_dur) * framerate)
        samples = np.pad(samples, (0, pad), 'constant')
        print(f"  Padded {pad} samples to reach target duration")

    # Pre-emphasis (treble boost)
    if config.PRE_EMPHASIS:
        samples = apply_pre_emphasis(samples, config.PRE_EMPHASIS_ALPHA)
        print(f"  Pre-emphasis applied (alpha={config.PRE_EMPHASIS_ALPHA})")

    # Resample to match total_steps
    x_old = np.linspace(0, 1, len(samples))
    x_new = np.linspace(0, 1, total_steps)
    resampled = np.interp(x_new, x_old, samples)

    # Map [-1, 1] to quantization levels [0, NUM_LEVELS-1]
    mapped = (resampled + 1.0) * 0.5 * (config.NUM_LEVELS - 1)

    # Dithering before quantization
    if config.DITHER:
        mapped = apply_triangular_dither(mapped, config.DITHER_AMPLITUDE)
        print(f"  Triangular dither applied (amplitude={config.DITHER_AMPLITUDE})")

    # Quantize
    quantized = np.round(mapped)
    quantized = np.clip(quantized, 0, config.NUM_LEVELS - 1)

    # Smoothing — reduced sigma vs V6
    sigma = config.SMOOTH_SIGMA
    if sigma > 0:
        kernel_radius = max(1, int(3.0 * sigma))
        k = np.exp(-np.arange(-kernel_radius, kernel_radius+1)**2 / (2*sigma**2))
        k /= k.sum()
        smoothed = np.convolve(quantized, k, mode='same')
        print(f"  Gaussian smoothing applied (sigma={sigma})")
    else:
        smoothed = quantized
        print(f"  Smoothing disabled")

    z_offsets = (smoothed + config.MIN_THICKNESS_LAYERS) * config.LAYER_HEIGHT
    return z_offsets


# =============================================================================
# BINARY STL WRITER
# =============================================================================

class BinarySTLWriter:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = None
        self.tri_count = 0

    def __enter__(self):
        self.file = open(self.filepath, 'wb')
        header = b'WAV2STL V7.0'
        self.file.write(header[:80].ljust(80, b'\x00'))
        self.file.write(struct.pack('<I', 0))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.seek(80)
            self.file.write(struct.pack('<I', self.tri_count))
            self.file.close()
            print(f"  STL: {self.tri_count:,} triangles, "
                  f"{os.path.getsize(self.filepath) / (1024*1024):.1f} MB")

    def write_tri(self, v0, v1, v2):
        ux, uy, uz = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
        vx, vy, vz = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
        nx = uy*vz - uz*vy
        ny = uz*vx - ux*vz
        nz = ux*vy - uy*vx
        ln = math.sqrt(nx*nx + ny*ny + nz*nz)
        if ln > 0:
            nx /= ln; ny /= ln; nz /= ln
        self.file.write(struct.pack('<3f', nx, ny, nz))
        self.file.write(struct.pack('<3f', *v0))
        self.file.write(struct.pack('<3f', *v1))
        self.file.write(struct.pack('<3f', *v2))
        self.file.write(struct.pack('<H', 0))
        self.tri_count += 1

    def write_quad(self, v0, v1, v2, v3):
        self.write_tri(v0, v1, v2)
        self.write_tri(v0, v2, v3)


# =============================================================================
# MONOLITHIC GEOMETRY GENERATION
# =============================================================================

def generate_monolithic_solid(writer: BinarySTLWriter, config: Config, z_offsets: np.ndarray):
    N = len(z_offsets)
    dTheta = config.RAD_PER_STEP

    def get_profile(i):
        theta = i * dTheta
        r_c = config.OUTER_GROOVE_RADIUS - config.K * theta
        z_mod = z_offsets[i] if 0 <= i < len(z_offsets) else 0.0

        h_base = config.BASE_HEIGHT
        h_groove = h_base + z_mod
        max_h = h_base + (config.NUM_LEVELS + config.MIN_THICKNESS_LAYERS) * config.LAYER_HEIGHT

        cos_t, sin_t = math.cos(theta), math.sin(theta)

        r_out = r_c + config.GROOVE_PITCH / 2.0
        p_out = (r_out*cos_t, r_out*sin_t, max_h)
        f_out = (r_out*cos_t, r_out*sin_t, 0.0)

        p_cen = (r_c*cos_t, r_c*sin_t, h_groove)

        r_in = r_c - config.GROOVE_PITCH / 2.0
        p_in = (r_in*cos_t, r_in*sin_t, max_h)
        f_in = (r_in*cos_t, r_in*sin_t, 0.0)

        return p_out, p_cen, p_in, f_out, f_in

    # --- 1. Outer Rim ---
    print("  Generating Outer Rim...")
    for k in range(config.STEPS_PER_REV):
        t0 = k * dTheta
        t1 = (k+1) * dTheta
        c0, s0 = math.cos(t0), math.sin(t0)
        c1, s1 = math.cos(t1), math.sin(t1)

        ro = config.RECORD_OUTER_RADIUS
        max_h = get_profile(k)[0][2]

        vo0_t = (ro*c0, ro*s0, max_h)
        vo1_t = (ro*c1, ro*s1, max_h)
        vo0_b = (ro*c0, ro*s0, 0.0)
        vo1_b = (ro*c1, ro*s1, 0.0)

        p0_out = get_profile(k)[0]
        p1_out = get_profile(k+1)[0]
        f0_out = get_profile(k)[3]
        f1_out = get_profile(k+1)[3]

        writer.write_quad(vo0_t, vo1_t, p1_out, p0_out)
        writer.write_quad(vo0_b, f0_out, f1_out, vo1_b)
        writer.write_quad(vo1_b, vo0_b, vo0_t, vo1_t)

    # --- 2. Spiral Groove Surface ---
    print(f"  Generating Spiral ({N:,} steps)...")
    p_prev = get_profile(0)

    for i in range(1, N + 1):
        p_curr = get_profile(i)

        writer.write_quad(p_prev[0], p_curr[0], p_curr[1], p_prev[1])
        writer.write_quad(p_prev[1], p_curr[1], p_curr[2], p_prev[2])
        writer.write_quad(p_prev[3], p_prev[4], p_curr[4], p_curr[3])

        p_prev = p_curr

    # --- 3. Inner Disk ---
    print("  Generating Inner Disk...")
    spindle_r = config.SPINDLE_HOLE_RADIUS
    start_exposed = max(0, N - config.STEPS_PER_REV)

    for k in range(start_exposed, N):
        p_k_in = get_profile(k)[2]
        p_kn_in = get_profile(k+1)[2]
        f_k_in = get_profile(k)[4]
        f_kn_in = get_profile(k+1)[4]

        tk, tkn = k * dTheta, (k+1) * dTheta
        sk_x, sk_y = spindle_r*math.cos(tk), spindle_r*math.sin(tk)
        skn_x, skn_y = spindle_r*math.cos(tkn), spindle_r*math.sin(tkn)

        s_k_top = (sk_x, sk_y, p_k_in[2])
        s_kn_top = (skn_x, skn_y, p_kn_in[2])
        s_k_bot = (sk_x, sk_y, 0.0)
        s_kn_bot = (skn_x, skn_y, 0.0)

        writer.write_quad(p_k_in, s_k_top, s_kn_top, p_kn_in)
        writer.write_quad(f_k_in, f_kn_in, s_kn_bot, s_k_bot)
        writer.write_quad(s_k_bot, s_kn_bot, s_kn_top, s_k_top)

    # --- 4. Caps ---
    def write_cap(idx, normal_dir):
        p_out, p_cen, p_in, f_out, f_in = get_profile(idx)
        if normal_dir > 0:
            writer.write_tri(p_out, f_out, p_cen)
            writer.write_tri(p_cen, f_out, f_in)
            writer.write_tri(p_cen, f_in, p_in)
        else:
            writer.write_tri(p_out, p_cen, f_out)
            writer.write_tri(p_cen, f_in, f_out)
            writer.write_tri(p_cen, p_in, f_in)

    write_cap(0, 1)
    write_cap(N, -1)

    print(f"  Monolithic Solid Complete ({N:,} steps).")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='WAV to 3D-printable vinyl STL (V7.0)')
    parser.add_argument('input', help='Input WAV file')
    parser.add_argument('output', nargs='?', help='Output STL file (default: input.stl)')
    parser.add_argument('--rpm', type=float, default=45.0,
                        help='Turntable RPM (default: 45). Typical: 33, 45, 78')
    parser.add_argument('--duration', type=float, default=15.0,
                        help='Target duration in seconds (default: 15)')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Gaussian smoothing sigma (default: 0.8, 0 to disable)')
    parser.add_argument('--no-preemphasis', action='store_true',
                        help='Disable treble pre-emphasis')
    parser.add_argument('--no-dither', action='store_true',
                        help='Disable triangular dithering')
    parser.add_argument('--preemphasis-alpha', type=float, default=0.97,
                        help='Pre-emphasis alpha (default: 0.97, range: 0.90-0.99)')

    args = parser.parse_args()

    wav_path = args.input
    stl_path = args.output or os.path.splitext(wav_path)[0] + f'_{int(args.rpm)}rpm.stl'

    config = Config(rpm=args.rpm, duration=args.duration)
    if args.sigma is not None:
        config.SMOOTH_SIGMA = args.sigma
    if args.no_preemphasis:
        config.PRE_EMPHASIS = False
    if args.no_dither:
        config.DITHER = False
    config.PRE_EMPHASIS_ALPHA = args.preemphasis_alpha

    print("=" * 60)
    print("WAV → STL Macro Audio Record (V7.0)")
    print("=" * 60)
    print("\n[Config]")
    config.summary()

    print("\n[1/3] Processing audio...")
    z_offsets = load_and_process_audio(wav_path, config)

    print(f"\n[2/3] Generating Mesh ({len(z_offsets):,} steps)...")
    with BinarySTLWriter(stl_path) as writer:
        generate_monolithic_solid(writer, config, z_offsets)

    sz = os.path.getsize(stl_path) / (1024*1024)
    print(f"\n[3/3] Complete!")
    print(f"  Output: {stl_path} ({sz:.1f} MB)")
    print("=" * 60)
    print("\nSuggested slicer settings for Bambu P1S:")
    print("  Layer height:     0.08mm")
    print("  Outer wall speed: 35mm/s")
    print("  Acceleration:     500 mm/s² (groove zones)")
    print("  Seam:             Random")
    print("  Supports:         None")
    print("=" * 60)


if __name__ == '__main__':
    main()