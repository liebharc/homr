# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

### Added

- **Improved accuracy**
- **PDF input** and **multi-page support**.
- Support for **dynamics** and **strong accents**.
- **Multiple voices** per staff.
- **Triplets printed without their number** are recognized and restored.
  Thanks to [@slobaspeed](https://github.com/slobaspeed)!
- Ties are recovered from slurs between notes of the same pitch.
  Thanks to [@pschuegr](https://github.com/pschuegr)!
- The positions of notes in the input image are written as MusicXML comments.

### Changed

- Staff detection is more robust for phone photos.
- The time signature is estimated per segment of a staff.
- PDF output filenames are standardized.
- homr exits with code 1 on failure.
- Dependencies use the standardized (PEP 517) syntax in `pyproject.toml`.
  Thanks to [@geofft](https://github.com/geofft)!
- Repeat signs stay paired with their images in the training data.
  Thanks to [@aleksey-golovanov](https://github.com/aleksey-golovanov)!

### Fixed

- MusicXML export: note timing when staves move independently, clefs and barlines inside
  chords, and grace notes taking time.
  Thanks to [@slobaspeed](https://github.com/slobaspeed)!
- MusicXML export: durations of zero and multi-measure rests.
  Thanks to [@Jimmy-Mendez](https://github.com/Jimmy-Mendez)!
- MusicXML export: a computed time signature written before the recognized one, slurs across
  voices, rests merged into chords and grace notes without pitch.
- Multi-measure rests are written as that many measures.
- Single staffs were wrongly paired into grand staffs.
- Staffs were joined across the gutter of a book.
- Staffs which only contain rests.
- `--read-staff-positions`: crash on color and grayscale images, lost title and lost
  grand staff content.
  Thanks to [@GeoffreyPlitt](https://github.com/GeoffreyPlitt)!
- Duplicate images on case-insensitive file systems.

## [0.7.0]

### Added

- **Improved grandstaff detection**.
  Thanks to [@weixlu](https://github.com/weixlu)!

### Changed

- Minimum Python version is now **Python 3.11**

### Fixed

- MusicXML export inconsistencies and compatibility issues. Thanks to [@fablau](https://github.com/fablau)!

## [0.6.2]

### Added

- **Improved accuracy**

## [0.6.1]

### Added

- **Greatly improved GPU inference**.
  Big thanks again to [@aicelen](https://github.com/aicelen) for the contribution!
- **Faster Segmentation model**

### Changed

- Removed dependency on **scipy**
- Updated all other dependencies

### Fixed

- Error when running on small images.

## [0.6.0]

### Added

- **Faster inference and GPU support**.
  Big thanks again to [@aicelen](https://github.com/aicelen) for the contribution!
- **Support for more symbols**

### Changed

- Grandstaff are now notated as single voice with two clefs in music xml

## [0.5.0]

### Added

- Introduced **PyTorch-free inference**.
  Big thanks to [@aicelen](https://github.com/aicelen) for the contribution!

## [0.4.0]

### Added

- Support for **triplets**.

## [0.3.0]

### Changed

- Removed dependency on **TensorFlow**.

## [0.2.3]

### Fixed

- Development requirements were mistakenly included in production.

## [0.2.2]

### Fixed

- File extensions like `.jpeg` were not handled correctly.

## [0.2.1]

### Added

- Extended support for additional **time signatures**.

## [0.2.0]

### Improved

- Detection quality: reduced the number of errors per symbol.

## [0.1.0]

### Added

- Initial release.
