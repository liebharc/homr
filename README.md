# adversarial-homr

Robustness analysis of HOMR against adversarial attacks and real-world image perturbations in Optical Music Recognition (OMR).

This project evaluates how HOMR performs under conditions such as:
- Noise and blur
- Perspective distortion
- Compression artifacts
- Lighting variation
- Occlusion
- Adversarial perturbations

The goal is to measure the reliability and security of camera-based sheet music transcription pipelines that convert images into MusicXML.

## Research Focus

- Adversarial Machine Learning for OMR
- Robustness evaluation
- Image perturbation testing
- MusicXML transcription accuracy
- Failure case analysis

## Pipeline

Sheet Music Image → Perturbation/Attack → HOMR → MusicXML → Evaluation Metrics

## Metrics

- Symbol Error Rate (SER)
- Note Accuracy
- Rhythm Accuracy
- Structural Consistency
- Robustness under perturbation strength

## Tech Stack

- Python
- PyTorch
- OpenCV
- MusicXML

## References

- HOMR
- Polyphonic-TrOMR
- Adversarial Machine Learning literature
