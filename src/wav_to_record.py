#!/usr/bin/env python3
"""
WAV → 3D-Printable STL Groove Geometry Generator (V8.0)

Encodes digital audio as Z-axis-modulated groove geometry in an Archimedean
spiral. All groove parameters are derived from two hardware inputs:
  - Nozzle diameter (0.4 mm or 0.2 mm)
  - Layer height (0.08 mm)

Derived constraints (see PARAMETERS.md and hacker_fab_answer.md):
  - Extrusion width w_e = 1.2 × nozzle diameter
  - Groove pitch ≥ 2 × w_e (land width ≥ w_e)
  - Steps/rev = floor(2π × r_inner / w_e) — inner radius sets Nyquist floor
  - Bit depth = 4 → 16 Z-levels × 0.08 mm = 1.28 mm total modulation
  - Sample rate = steps/rev × RPM / 60
  - Nyquist = sample_rate / 2

Mesh generation produces a watertight triangulated solid with:
  - Groove walls (outer + inner) as quad strips along the spiral
  - Explicit land surface triangulation at Z_surface
  - Inter-turn land bridging faces between adjacent spiral turns
  - Bottom face, outer rim, inner disk, start/end caps

Usage:
    python wav_to_record.py input.wav [output.stl] [--rpm 78] [--duration 60]

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
    """All physical parameters in millimeters. Groove geometry derived from
    nozzle diameter and layer height — not hardcoded."""

    # Extrusion width multiplier: actual bead width = k × nozzle_diameter.
    # k ∈ [1.1, 1.3] depending on slicer, speed, temperature. 1.2 is
    # conservative central estimate. Characterize with single-wall
    # calibration cube if precision is critical.
    EXTRUSION_WIDTH_FACTOR = 1.2

    def __init__(self, rpm: float = 78.0, duration: float = 60.0,
                 nozzle: float = 0.4):
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
        # Nozzle diameter is the hardware input. Effective feature size is
        # the extrusion width w_e = EXTRUSION_WIDTH_FACTOR × nozzle_diameter.
        # All geometry constraints are computed from w_e, not nozzle directly.
        self.NOZZLE_DIAMETER = nozzle
        self.EXTRUSION_WIDTH = self.EXTRUSION_WIDTH_FACTOR * nozzle

        # Layer height = quantization step size.
        # Z positioning error: ±0.01–0.02 mm = ±12.5–25% of one step.
        self.LAYER_HEIGHT = 0.08

        # -- Groove parameters (derived from extrusion width) --
        # Groove width = one extrusion width (minimum FDM feature).
        # Actual deposited width is w_e, not nozzle diameter.
        self.GROOVE_WIDTH = self.EXTRUSION_WIDTH

        # Groove pitch ≥ 2 × w_e ensures land width ≥ w_e.
        # Design margin: use 2.5 × w_e if land reliability is critical.
        self.GROOVE_PITCH = 2.0 * self.EXTRUSION_WIDTH

        # -- Angular resolution (derived from inner radius) --
        # steps_per_rev = floor(2π × r_inner / w_e)
        # This is the maximum angular resolution at the inner radius.
        # Outer grooves have higher spatial resolution but inner radius
        # sets the Nyquist floor for the entire recording.
        # WARNING: at inner radius with 0.2 mm nozzle, arc length per step
        # is 0.24 mm. Verify in Bambu Studio G-code preview that slicer
        # does not merge these segments.
        self.STEPS_PER_REV = int(
            (2.0 * math.pi * self.INNER_GROOVE_RADIUS) / self.EXTRUSION_WIDTH
        )
        self.RAD_PER_STEP = (2.0 * math.pi) / self.STEPS_PER_REV

        # -- Audio processing --
        self.SMOOTH_SIGMA = 0.8

        # Bit depth = 4 → 16 discrete Z-levels.
        # Effective bit depth may be ~3 bits under worst-case Z positioning
        # error (±0.02 mm = ±25% of 0.08 mm step). Layer height uncertainty
        # does not justify increasing bit depth until Z accuracy is
        # characterized by staircase test print.
        self.BIT_DEPTH = 4
        self.NUM_LEVELS = 2 ** self.BIT_DEPTH  # 16

        # Minimum thickness in layers below groove floor. Prevents groove
        # floor from coinciding with record base.
        self.MIN_THICKNESS_LAYERS = 2

        # -- Pre-emphasis --
        # PLACEHOLDER: α=0.97 is assumed, not measured. Pre-emphasis should
        # be the inverse of the measured groove-stylus-cartridge transfer
        # function H(f). Characterize by printing a sine sweep test record
        # (300–2042 Hz in 50 Hz steps), recording playback output, and
        # computing H(f) = Y(f)/X(f).
        self.PRE_EMPHASIS = True
        self.PRE_EMPHASIS_ALPHA = 0.97

        # -- Dithering --
        # TPDF dithering only. Lipshitz noise shaping (1.5, -0.5) is
        # disabled — at inner-radius Nyquist, the audio band fills the full
        # Nyquist range with zero spectral headroom for noise redistribution.
        # Shaped error aliases back into the passband.
        self.DITHER = True
        self.DITHER_AMPLITUDE = 0.5

        # -- Derived quantities --
        self.RADIAL_TRAVEL = self.OUTER_GROOVE_RADIUS - self.INNER_GROOVE_RADIUS
        self.K = self.GROOVE_PITCH / (2.0 * math.pi)
        self.LAND_WIDTH = self.GROOVE_PITCH - self.GROOVE_WIDTH

        # -- Signal parameters (derived from geometry) --
        self.F_SAMPLE_INNER = self.STEPS_PER_REV * self.RPM / 60.0
        self.F_NYQUIST_INNER = self.F_SAMPLE_INNER / 2.0
        steps_outer = int(
            (2.0 * math.pi * self.OUTER_GROOVE_RADIUS) / self.EXTRUSION_WIDTH
        )
        self.F_SAMPLE_OUTER = steps_outer * self.RPM / 60.0
        self.F_NYQUIST_OUTER = self.F_SAMPLE_OUTER / 2.0

        # Z_surface: maximum height of land surface
        self.Z_SURFACE = (
            self.BASE_HEIGHT
            + (self.NUM_LEVELS + self.MIN_THICKNESS_LAYERS) * self.LAYER_HEIGHT
        )

    def summary(self):
        total_revs = self.TARGET_DURATION * self.REV_PER_SEC
        total_steps = int(total_revs * self.STEPS_PER_REV)
        groove_count = total_revs
        radial_used = groove_count * self.GROOVE_PITCH
        recording_time = self.RADIAL_TRAVEL / (self.GROOVE_PITCH * self.REV_PER_SEC)
        tri_spiral = 10 * total_steps  # 6 groove + 4 land per step
        tri_rim = 6 * self.STEPS_PER_REV
        tri_total = tri_spiral + 2 * tri_rim + 6

        v_inner = 2.0 * math.pi * self.INNER_GROOVE_RADIUS * self.RPM / 60.0
        v_outer = 2.0 * math.pi * self.OUTER_GROOVE_RADIUS * self.RPM / 60.0
        R_stylus = 0.5  # mm — assumed, see PARAMETERS.md
        f_geo_inner = v_inner / (2.0 * math.pi * R_stylus)
        f_geo_outer = v_outer / (2.0 * math.pi * R_stylus)

        print(f"  Nozzle diameter:     {self.NOZZLE_DIAMETER} mm")
        print(f"  Extrusion width (w_e): {self.EXTRUSION_WIDTH:.2f} mm")
        print(f"  RPM:                 {self.RPM}")
        print(f"  Duration:            {self.TARGET_DURATION}s "
              f"(max: {recording_time:.1f}s)")
        print(f"  Groove pitch:        {self.GROOVE_PITCH:.2f} mm")
        print(f"  Groove width:        {self.GROOVE_WIDTH:.2f} mm")
        print(f"  Land width:          {self.LAND_WIDTH:.2f} mm "
              f"(min reliable: {self.EXTRUSION_WIDTH:.2f} mm)")
        print(f"  Steps/rev (inner):   {self.STEPS_PER_REV}")
        print(f"  Total revolutions:   {total_revs:.1f}")
        print(f"  Total steps:         {total_steps:,}")
        print(f"  Radial used:         {radial_used:.1f} mm / "
              f"{self.RADIAL_TRAVEL:.1f} mm")
        print(f"  Sample rate (inner): {self.F_SAMPLE_INNER:.1f} Hz")
        print(f"  Nyquist (inner):     {self.F_NYQUIST_INNER:.1f} Hz")
        print(f"  Nyquist (outer):     {self.F_NYQUIST_OUTER:.1f} Hz")
        print(f"  Recording time:      {recording_time:.1f} s")
        print(f"  Triangle count (est): ~{tri_total:,}")
        print(f"  Z_surface:           {self.Z_SURFACE:.2f} mm")
        print(f"  Stylus geometric filter (R={R_stylus}mm, inner): "
              f"{f_geo_inner:.1f} Hz")

        if f_geo_inner < self.F_NYQUIST_INNER:
            print(f"  WARNING: Stylus geometric filter ({f_geo_inner:.1f} Hz) "
                  f"is more restrictive than Nyquist ({self.F_NYQUIST_INNER:.1f} Hz) "
                  f"for R={R_stylus}mm.")
            R_required = (self.INNER_GROOVE_RADIUS * self.RPM
                          / (60.0 * self.F_NYQUIST_INNER))
            print(f"  Use R < {R_required:.3f} mm stylus for Nyquist "
                  f"to be the binding constraint.")

        if radial_used > self.RADIAL_TRAVEL:
            print(f"  WARNING: Duration exceeds capacity — "
                  f"max {recording_time:.1f}s at {self.RPM} RPM")


# =============================================================================
# AUDIO PROCESSING
# =============================================================================

def apply_pre_emphasis(samples: np.ndarray, alpha: float) -> np.ndarray:
    """First-order high-pass pre-emphasis: y[n] = x[n] - alpha * x[n-1]
    PLACEHOLDER: α=0.97 is not measured. See PARAMETERS.md."""
    emphasized = np.zeros_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - alpha * samples[:-1]
    peak = np.max(np.abs(emphasized))
    if peak > 0:
        emphasized /= peak
    return emphasized


def apply_triangular_dither(quantized: np.ndarray, amplitude: float) -> np.ndarray:
    """TPDF dithering: sum of two uniform distributions.
    Decorrelates quantization error without spectral redistribution.
    Lipshitz noise shaping disabled — zero Nyquist headroom at inner radius."""
    r1 = np.random.uniform(-amplitude, amplitude, len(quantized))
    r2 = np.random.uniform(-amplitude, amplitude, len(quantized))
    return quantized + r1 + r2


def load_and_process_audio(wav_path: str, config: Config) -> np.ndarray:
    total_revs = config.TARGET_DURATION * config.REV_PER_SEC
    total_steps = int(total_revs * config.STEPS_PER_REV)
    print(f"  Target: {config.TARGET_DURATION}s @ {config.RPM} RPM "
          f"-> {total_steps:,} steps")

    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        file_duration = n_frames / framerate
        frames_to_read = int(min(file_duration, config.TARGET_DURATION) * framerate)
        raw = wf.readframes(frames_to_read)
        print(f"  WAV: {framerate}Hz, {n_channels}ch, "
              f"{sampwidth*8}bit, {file_duration:.1f}s")

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
            samples[i] = int.from_bytes(
                raw[i*3:i*3+3], 'little', signed=True
            ) / 8388608.0
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
        print(f"  Padded {pad} samples")

    # Pre-emphasis (PLACEHOLDER — see Config docstring)
    if config.PRE_EMPHASIS:
        samples = apply_pre_emphasis(samples, config.PRE_EMPHASIS_ALPHA)
        print(f"  Pre-emphasis applied (alpha={config.PRE_EMPHASIS_ALPHA} "
              f"— PLACEHOLDER, not measured)")

    # Resample to match total_steps
    x_old = np.linspace(0, 1, len(samples))
    x_new = np.linspace(0, 1, total_steps)
    resampled = np.interp(x_new, x_old, samples)

    # Map [-1, 1] to quantization levels [0, NUM_LEVELS-1]
    mapped = (resampled + 1.0) * 0.5 * (config.NUM_LEVELS - 1)

    # TPDF dithering (noise shaping disabled — see Config)
    if config.DITHER:
        mapped = apply_triangular_dither(mapped, config.DITHER_AMPLITUDE)
        print(f"  TPDF dither applied (amplitude={config.DITHER_AMPLITUDE})")

    # Quantize
    quantized = np.round(mapped)
    quantized = np.clip(quantized, 0, config.NUM_LEVELS - 1)

    # Smoothing
    sigma = config.SMOOTH_SIGMA
    if sigma > 0:
        kernel_radius = max(1, int(3.0 * sigma))
        k = np.exp(-np.arange(-kernel_radius, kernel_radius+1)**2
                    / (2*sigma**2))
        k /= k.sum()
        smoothed = np.convolve(quantized, k, mode='same')
        print(f"  Gaussian smoothing (sigma={sigma})")
    else:
        smoothed = quantized

    # Z offsets: Z_floor = Z_base + (s + n_min) * h
    # Sign convention: Z_floor < Z_surface for all valid s.
    # s ∈ [0, NUM_LEVELS-1], so s + n_min < NUM_LEVELS + n_min always.
    # Inter-turn Z self-intersection is provably impossible given this mapping.
    z_offsets = (smoothed + config.MIN_THICKNESS_LAYERS) * config.LAYER_HEIGHT

    # Assertion: verify Z_floor < Z_surface for all samples
    z_floor_max = (config.NUM_LEVELS - 1 + config.MIN_THICKNESS_LAYERS) \
                  * config.LAYER_HEIGHT + config.BASE_HEIGHT
    assert z_floor_max < config.Z_SURFACE, \
        f"Sign convention violated: max Z_floor ({z_floor_max}) >= " \
        f"Z_surface ({config.Z_SURFACE})"

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
        header = b'WAV2STL V8.0'
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
# GEOMETRY GENERATION
# =============================================================================

def generate_monolithic_solid(writer: BinarySTLWriter, config: Config,
                              z_offsets: np.ndarray):
    """Generate a watertight triangulated solid encoding the groove geometry.

    Mesh structure per angular step:
      - 3 quad strips: outer wall, inner wall, bottom face (6 tri)
      - 2 quads: top land surface at Z_surface (4 tri)
    Between adjacent spiral turns:
      - 1 quad: inter-turn land bridging at Z_surface (2 tri)
    Boundary closures:
      - Outer rim, inner disk, start cap, end cap
    """
    N = len(z_offsets)
    dTheta = config.RAD_PER_STEP
    max_h = config.Z_SURFACE

    def get_profile(i):
        theta = i * dTheta
        r_c = config.OUTER_GROOVE_RADIUS - config.K * theta
        z_mod = z_offsets[i] if 0 <= i < len(z_offsets) else 0.0

        h_groove = config.BASE_HEIGHT + z_mod
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        r_out = r_c + config.GROOVE_PITCH / 2.0
        p_out = (r_out*cos_t, r_out*sin_t, max_h)
        f_out = (r_out*cos_t, r_out*sin_t, 0.0)

        p_cen = (r_c*cos_t, r_c*sin_t, h_groove)
        # Groove center projected to land surface height (for top surface)
        p_cen_top = (r_c*cos_t, r_c*sin_t, max_h)

        r_in = r_c - config.GROOVE_PITCH / 2.0
        p_in = (r_in*cos_t, r_in*sin_t, max_h)
        f_in = (r_in*cos_t, r_in*sin_t, 0.0)

        return p_out, p_cen, p_in, f_out, f_in, p_cen_top

    # --- 1. Outer Rim ---
    print("  Generating Outer Rim...")
    for k in range(config.STEPS_PER_REV):
        t0 = k * dTheta
        t1 = (k+1) * dTheta
        c0, s0 = math.cos(t0), math.sin(t0)
        c1, s1 = math.cos(t1), math.sin(t1)

        ro = config.RECORD_OUTER_RADIUS

        vo0_t = (ro*c0, ro*s0, max_h)
        vo1_t = (ro*c1, ro*s1, max_h)
        vo0_b = (ro*c0, ro*s0, 0.0)
        vo1_b = (ro*c1, ro*s1, 0.0)

        prof0 = get_profile(k)
        prof1 = get_profile(k+1)

        # Top: outer radius to first groove outer edge
        writer.write_quad(vo0_t, vo1_t, prof1[0], prof0[0])
        # Bottom
        writer.write_quad(vo0_b, prof0[3], prof1[3], vo1_b)
        # Outer wall
        writer.write_quad(vo1_b, vo0_b, vo0_t, vo1_t)

    # --- 2. Spiral Groove Surface ---
    print(f"  Generating Spiral ({N:,} steps)...")
    p_prev = get_profile(0)

    for i in range(1, N + 1):
        p_curr = get_profile(i)

        # Outer groove wall: p_out to p_cen
        writer.write_quad(p_prev[0], p_curr[0], p_curr[1], p_prev[1])
        # Inner groove wall: p_cen to p_in
        writer.write_quad(p_prev[1], p_curr[1], p_curr[2], p_prev[2])
        # Bottom face
        writer.write_quad(p_prev[3], p_prev[4], p_curr[4], p_curr[3])

        # --- Top land surface at Z_surface ---
        # Outer land: p_out → p_cen_top (both at Z_surface)
        writer.write_quad(p_prev[0], p_curr[0], p_curr[5], p_prev[5])
        # Inner land: p_cen_top → p_in (both at Z_surface)
        writer.write_quad(p_prev[5], p_curr[5], p_curr[2], p_prev[2])

        # --- Inter-turn land bridging ---
        # Connect p_in of previous turn to p_out of current step at the
        # same angular position, one full revolution earlier.
        # Previous turn's inner edge at this angle is at step (i - steps_per_rev).
        j = i - config.STEPS_PER_REV
        if j >= 0 and j < N:
            prev_turn = get_profile(j)
            prev_turn_next = get_profile(j + 1) if (j + 1) <= N else get_profile(j)
            # Bridge: p_in[turn N-1] to p_out[turn N] at Z_surface
            # Both vertices are at Z_surface, forming a flat quad
            writer.write_quad(
                prev_turn[2],      # p_in[j]    (prev turn inner edge)
                prev_turn_next[2], # p_in[j+1]  (prev turn inner edge, next step)
                p_curr[0],         # p_out[i]   (current turn outer edge)
                p_prev[0]          # p_out[i-1] (current turn outer edge, prev step)
            )

        p_prev = p_curr

    # --- 3. Inner Disk ---
    print("  Generating Inner Disk...")
    spindle_r = config.SPINDLE_HOLE_RADIUS
    start_exposed = max(0, N - config.STEPS_PER_REV)

    for k in range(start_exposed, N):
        prof_k = get_profile(k)
        prof_kn = get_profile(k+1)

        tk, tkn = k * dTheta, (k+1) * dTheta
        sk_x, sk_y = spindle_r*math.cos(tk), spindle_r*math.sin(tk)
        skn_x, skn_y = spindle_r*math.cos(tkn), spindle_r*math.sin(tkn)

        s_k_top = (sk_x, sk_y, prof_k[2][2])
        s_kn_top = (skn_x, skn_y, prof_kn[2][2])
        s_k_bot = (sk_x, sk_y, 0.0)
        s_kn_bot = (skn_x, skn_y, 0.0)

        # Top: groove inner edge to spindle
        writer.write_quad(prof_k[2], s_k_top, s_kn_top, prof_kn[2])
        # Bottom
        writer.write_quad(prof_k[4], prof_kn[4], s_kn_bot, s_k_bot)
        # Spindle wall
        writer.write_quad(s_k_bot, s_kn_bot, s_kn_top, s_k_top)

    # --- 4. Caps ---
    def write_cap(idx, normal_dir):
        p_out, p_cen, p_in, f_out, f_in, p_cen_top = get_profile(idx)
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

    print(f"  Mesh Complete ({N:,} steps, {writer.tri_count:,} triangles).")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='WAV to 3D-printable vinyl STL (V8.0)')
    parser.add_argument('input', help='Input WAV file')
    parser.add_argument('output', nargs='?',
                        help='Output STL file (default: input.stl)')
    parser.add_argument('--rpm', type=float, default=78.0,
                        help='Turntable RPM (default: 78)')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Target duration in seconds (default: 60)')
    parser.add_argument('--nozzle', type=float, default=0.4,
                        help='Nozzle diameter in mm (default: 0.4)')
    parser.add_argument('--sigma', type=float, default=None,
                        help='Gaussian smoothing sigma (default: 0.8)')
    parser.add_argument('--no-preemphasis', action='store_true',
                        help='Disable pre-emphasis')
    parser.add_argument('--no-dither', action='store_true',
                        help='Disable TPDF dithering')
    parser.add_argument('--preemphasis-alpha', type=float, default=0.97,
                        help='Pre-emphasis alpha (default: 0.97, PLACEHOLDER)')

    args = parser.parse_args()

    wav_path = args.input
    stl_path = args.output or os.path.splitext(wav_path)[0] + \
               f'_{int(args.rpm)}rpm.stl'

    config = Config(rpm=args.rpm, duration=args.duration, nozzle=args.nozzle)
    if args.sigma is not None:
        config.SMOOTH_SIGMA = args.sigma
    if args.no_preemphasis:
        config.PRE_EMPHASIS = False
    if args.no_dither:
        config.DITHER = False
    config.PRE_EMPHASIS_ALPHA = args.preemphasis_alpha

    print("=" * 60)
    print("WAV → STL Groove Geometry Generator (V8.0)")
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
    print("\nSlicer settings (Bambu P1S):")
    print(f"  Layer height:     {config.LAYER_HEIGHT}mm")
    print("  Outer wall speed: 35mm/s")
    print("  Acceleration:     500 mm/s²")
    print("  Seam:             Random")
    print("  Supports:         None")
    print("  Verify: inner-radius segments (arc length "
          f"{2*math.pi*config.INNER_GROOVE_RADIUS/config.STEPS_PER_REV:.2f}mm)"
          " are not merged in G-code preview")
    print("=" * 60)


if __name__ == '__main__':
    main()