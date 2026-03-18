# Open Vinyl: 4-Bit Mechanical Audio Encoding

A full signal conditioning and structural mapping pipeline that encodes digital audio into 3D-printable groove geometry. Built to explore physical signal recovery under severe manufacturing constraints.

**[Link to 60-Second Video Demo]** *(Embed Deliverable 5 video here)*

## The Engineering Problem: The FDM Noise Floor

Standard FDM 3D printing with a 0.4mm nozzle and 0.08mm layer height introduces a massive structural noise floor. At 0.08mm Z-resolution, the stylus can only resolve **4 bits (16 discrete levels)** of vertical displacement. 

Naive quantization of a waveform to 4 bits yields unlistenable static due to severe quantization error. Extracting a listenable signal required building a custom DSP pipeline to condition the data prior to physical encoding.

## Signal Conditioning Pipeline (`src/wav_to_4bit.py`)

To push quantization noise out of the audible band and compensate for physical rolloff, the pipeline implements:

1. **Anti-Aliasing:** 4th-order Butterworth low-pass filter (`scipy.signal.butter`) applied via zero-phase `filtfilt`.
2. **Frequency Compensation:** First-order high-pass pre-emphasis ($\\alpha = 0.97$) to counteract the mechanical high-frequency attenuation of the PLA plastic boundary layer.
3. **Out-of-Band Error Diffusion:** Second-order Lipshitz noise-shaped dithering, pushing the 4-bit quantization error spectrum above the primary audio band.
4. **Decorrelation:** Triangular PDF (TPDF) dithering to eliminate harmonic distortion artifacts caused by the 16-level quantization limit.

## Physical Rig Design

Temporal resolution is strictly bound by the angular velocity of the record. To maximize the sample rate at the perimeter, I bypassed standard turntable speeds (33/45 RPM) and designed a custom direct-drive test rig operating at **150 RPM**.

The geometry generation (`src/wav_to_record.py`) is fully parameterized, dynamically scaling the STL modulation mapped against the variable `STEPS_PER_REV`.

## Running the Pipeline

```bash
pip install -r requirements.txt
python src/wav_to_4bit.py input.wav output_4bit.wav
python src/wav_to_record.py output_4bit.wav
```
