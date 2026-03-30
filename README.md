# OpenVinyl

Encodes digital audio as Z-axis-modulated groove geometry into FDM-printed records played back on a turntable with a ceramic cartridge.

## Encoding scheme

Commercial vinyl: lateral stylus displacement, 90° V-groove, 25–55 µm features.
FDM constraint: minimum XY feature = extrusion width w_e ≈ 1.2 × nozzle diameter.
OpenVinyl: vertical (Z-axis) modulation — groove floor height encodes amplitude.
Encoding class: Edison hill-and-dale, not lateral-cut vinyl.

## Key derived constraints

All groove parameters are derived from two hardware inputs: nozzle diameter and layer height (0.08 mm).

| Parameter | 0.4 mm nozzle | 0.2 mm nozzle |
|:---|:---|:---|
| Extrusion width (w_e) | 0.48 mm | 0.24 mm |
| Groove pitch (≥ 2 × w_e) | 0.96 mm | 0.48 mm |
| Land width | 0.48 mm | 0.24 mm |
| Steps/rev (r = 40 mm) | 523 | 1,047 |
| Steps/rev (r = 120 mm) | 1,571 | 3,142 |
| Nyquist at inner radius | 340 Hz | 681 Hz |
| Nyquist at outer radius | 1,021 Hz | 2,042 Hz |
| Total grooves (80 mm travel) | 83 | 166 |
| Recording time at 78 RPM | 64 s | 128 s |
| Bit depth | 4 (16 levels, 1.28 mm Z-range, 24 dB) | 4 |

**Slope constraint:** A × f ≤ tan(θ_max) × r × RPM / 60.
At θ_max = 45°, r = 40 mm, 78 RPM: full-amplitude signals above 87 Hz exceed the slope limit. The slope constraint, not bit depth, binds dynamic range at the inner radius.

**Stylus geometric filter:** f_geo = v / (2πR).
At R = 0.5 mm: limits bandwidth to 104–312 Hz — more restrictive than Nyquist. For Nyquist to be the binding constraint: R < 0.076 mm required.

**Contact mechanics:** Hertzian analysis gives p_max = 66–78 MPa at 3–5 g VTF, exceeding PLA yield (~50 MPa). Plastic deformation of the groove floor is expected on first play.

## DSP pipeline

| Stage | Parameter | Derivation |
|:---|:---|:---|
| HPF | 300 Hz | Bandwidth floor — no harmonics below this in [681, 2042] Hz window |
| LPF | Inner-radius Nyquist (geometry-derived) | Anti-aliasing; replaces previous 8 kHz cutoff |
| Compression | ≥ 6:1, 18 dB target | 4-bit = 24 dB; 18 dB leaves dithering headroom |
| Noise shaping | TPDF only | Lipshitz disabled — zero Nyquist headroom for spectral redistribution |
| Pre-emphasis | **OPEN** — requires H(f) measurement | Characterize from sine sweep test record |

## Open parameters

See [PARAMETERS.md](PARAMETERS.md) for the full table of open parameters with bounding approaches and required measurements.

Key open parameters:
- θ_max (stylus max tracking slope) — bounded by cantilever resonance, not compliance
- Stylus tip radius R — critical design variable determining whether Nyquist or geometric filter binds
- Pre-emphasis transfer function — requires sine sweep test record
- RPM stability — 1% variation = 17 cents pitch error, 5% = 85 cents
- Z positioning accuracy — ±0.02 mm = ±25% of one quantization step

## Characterization procedure

1. **Z accuracy:** Print staircase test pattern (Z levels 0–15). Measure actual heights. Compute σ_Z. If σ_Z > h/4, reduce effective bit depth by 1.

2. **Noise floor:** Print blank groove (no audio modulation). Record playback. Measure noise floor spectrum. Identify staircase frequency (f = v/h = 4,088–12,238 Hz).

3. **Transfer function:** Print sine sweep (300–2,042 Hz in 50 Hz steps). Record playback. Compute H(f) = Y(f)/X(f). Design pre-emphasis as H⁻¹(f).

4. **RPM stability:** Measure with tachometer over 60+ seconds. 1% variation = 17 cents. 5% = 85 cents (above ±50 cent accuracy threshold).

5. **Slicer verification:** Inspect G-code in Bambu Studio. Verify inner-radius segments (0.24 mm arc length at 0.2 mm nozzle) are not merged.

## Usage

### DSP preprocessing

```
python src/wav_to_4bit.py input.wav [output.wav] [--nozzle 0.4] [--rpm 78]
```

Conditions audio for 4-bit quantization: highpass at 300 Hz, lowpass at geometry-derived Nyquist, 6:1 compression, TPDF dithering. Outputs 16-bit WAV container with 4-bit resolution.

### STL generation

```
python src/wav_to_record.py input.wav [output.stl] [--rpm 78] [--nozzle 0.4] [--duration 60]
```

Generates watertight binary STL with groove geometry. Prints all derived parameters (Nyquist limits, triangle count, stylus geometric filter warning) on execution.

### Print settings (Bambu P1S)

| Setting | Value |
|:---|:---|
| Layer height | 0.08 mm |
| Outer wall speed | 35 mm/s |
| Acceleration | 500 mm/s² |
| Seam | Random |
| Supports | None |

## File structure

```
src/
  wav_to_4bit.py    DSP preprocessing (V3.0)
  wav_to_record.py  Groove geometry + STL generation (V8.0)
PARAMETERS.md       Open parameters table (living document)
requirements.txt    Python dependencies
```

## References

- Ghassaei, A. (2012). 3D Printed Record. Instructables.
- IEC 60098: Analogue audio disk records and reproducing equipment.
