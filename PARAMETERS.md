# Open Parameters

Living document. Parameters move from Open to Closed as experimental characterization is completed.

## Status key

- **Open**: requires experimental measurement. Bounding approach stated.
- **Bounded**: analytical bound exists but exact value needs measurement.
- **Closed**: measured value available.

## Parameters

| Parameter | Current assumption | Bound | Measurement required | Status |
|:---|:---|:---|:---|:---|
| θ_max (max tracking slope) | 30°–60° estimated | Not compliance-limited for ceramic cartridges (cantilever travel limit never reached at any slope). Bounded by cantilever resonance and effective moving mass | Print linear-ramp test groove with increasing slope angles. Play back, identify mistracking onset (audible distortion, skipping) | Open |
| Stylus tip radius R | 0.5 mm assumed | R < groove_width/2 to physically fit in groove. R < 0.076 mm for Nyquist to be the binding bandwidth constraint (otherwise stylus geometric filter dominates at 104–312 Hz) | Test styli of varying R. Measure bandwidth vs. tracking stability tradeoff. Standard vinyl stylus (R ≈ 18 µm) fits but may rattle in 0.48 mm groove | Open |
| m_eff (cantilever effective mass) | Unknown | Determines cantilever resonant frequency and dynamic θ_max | Cartridge spec sheet lookup, or frequency response measurement (sweep input, measure output rolloff) | Open |
| Pre-emphasis H⁻¹(f) | α = 0.97 (placeholder, not measured) | No bound available without measurement | Print sine sweep test record (300–2,042 Hz in 50 Hz steps, 2 seconds per tone). Record electrical output from cartridge. Compute H(f) = Y(f)/X(f). Design pre-emphasis filter as H⁻¹(f) | Open |
| Extrusion width w_e(v, T) | k = 1.2 (factor × nozzle diameter) | k ∈ [1.1, 1.3] depending on slicer, print speed, temperature | Print single-wall calibration cube. Measure wall thickness at 5+ points with calipers or micrometer. Compute mean k and standard deviation | Open |
| PLA compressive creep | First-play plastic deformation confirmed by Hertzian analysis (p_max = 66–78 MPa > PLA yield ~50 MPa). Degradation rate unknown | Bounded above by Hertz contact analysis. Initial deformation is largest; subsequent plays produce diminishing additional deformation (strain hardening) | Measure groove floor profile with profilometer or optical microscope before and after N plays (N = 1, 5, 10, 50). Quantify Z profile change per play | Open |
| RPM stability | Unknown | 1% speed variation = ~17 cents pitch error (below 50-cent threshold). 5% = ~85 cents (above threshold) | Tachometer measurement over 60+ seconds. Record RPM time series, compute mean, standard deviation, and peak-to-peak variation | Open |
| Surface roughness spectrum | Spatial period = layer height (0.08 mm, theoretical). Produces tonal noise at f = v/h: 4,088 Hz (r=40 mm) to 12,238 Hz (r=120 mm) | Bounded by layer height and groove velocity | Profilometry or SEM of printed groove floor. Alternatively: optical microscope measurement of actual Z staircase profile | Open |
| Slicer minimum segment length | Bambu Studio: nominally 0.1–0.4 mm (configurable). Inner-radius arc length per step = 0.24 mm (0.2 mm nozzle) | Segments at 0.24 mm may be at the merge/drop threshold | Inspect G-code preview in Bambu Studio for merged segments at inner radius. Compare angular step count in G-code vs STL | Open |
| Archard wear coefficient K | K ∈ [10⁻⁴, 10⁻⁶] for polymer-on-hard-material sliding. At K = 10⁻⁴: ~32,000 plays per quantization step of wear. Negligible vs first-play plastic deformation | Bounded by polymer-on-hard contact literature (2 orders of magnitude range) | Measure groove floor profile after controlled play count (1, 10, 100 plays). Fit K from measured wear volume | Open |
| Compressor attack/release times | Attack ~2 ms, release ~50 ms (estimated from onset detection and musical phrasing requirements) | Bounded by onset timing (~2 ms attack) and sustain characteristics (~50 ms release) | Listening test: process test melody with varying attack (1–10 ms) and release (20–200 ms), evaluate against groove playback. Select parameters that maximize Level 1 success criterion | Open |
| Z positioning accuracy (Bambu P1S) | ±0.01–0.02 mm estimated | At ±0.02 mm: 25% of one quantization step. Bottom 1–2 bits may be unreliable. Effective bit depth may be 3 instead of 4 | Print staircase test pattern (16 Z-levels). Measure actual Z heights at each level. Compute σ_Z per level | Open |
