# homr

homr is an Optical Music Recognition (OMR) software designed to transform camera pictures of sheet music into machine-readable MusicXML format. The resulting [MusicXML](https://www.w3.org/2021/06/musicxml40/) files can be further processed using tools such as [musescore](https://musescore.com/).

## Prequisites

- Python 3.10
- Poetry
- Optional: NVidia GPU with CUDA 12.1

## Getting started

- Clone the repository
- Install dependencies using `poetry install`
- Run the program using `poetry run homr <image>`
- The resulting MusicXML file will be saved in the same directory as the input image
- To combine the MusicXML results from multiple images, you can use [relieur](https://github.com/papoteur-mga/relieur)

## Example

The example below provides an overview of the current performance of the implementation. While some errors are present in the output, the overall structure remains accurate.

|                                          Original Image                                           |                                                                               homr Result                                                                                |
| :-----------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/BreezeWhite/oemer/blob/main/figures/tabi.jpg?raw=true" width="400" > | <img src="https://github.com/liebharc/homr/blob/main/figures/tabi.svg?raw=true" alt="Go to https://github.com/liebharc/homr if this image isn't displayed" width="400" > |

The homr result is obtained by processing the [homr output](figures/tabi.musicxml) and rendering it with [musescore](https://musescore.com/).

## Limitations

The current implementation focuses on pitch and rhtyhm information, neglecting dynamics, articulation, and other musical symbols.

## Technical Details

homr employs segmentation techniques outlined in [oemer](https://github.com/BreezeWhite/oemer) to identify staff lines, clefs, bar lines, and note heads in an image. These components are combined to determine the position of staffs within the picture.

Subsequently, each staff image undergoes transformation using a transformer model (based on [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR)) to identify symbols present on the staff. Pitch information is cross-validated with note head data obtained from the segmentation model.

The results are then converted into MusicXML format and saved to disk.

### Image Predictions

homr utilizes oemer's UNet implementations to isolate staff lines and other symbols for note head identification. These predictions serve as input for staff and symbol detection.

Preprocessing the image has shown to enhance robustness against noisy backgrounds and variations in brightness.

### Staff and Symbol Detection

The detection process involves extracting model data types from the image predictions. A key concept is the "staff anchor," which serves as a reference point ensuring accurate staff detection amidst symbols that might obscure it. Clefs and bar lines are currently utilized as anchor symbols.

For each anchor, the algorithm attempts to locate five staff lines and constructs the remainder of the staff around these anchors.

#### Unit Sizes

The unit size denotes the distance between staff lines, which may vary due to camera perspective. To accommodate this, the unit size is calculated per staff.

#### Connecting Staffs

Support for multiple voices and grand staffs is facilitated by identifying braces and brackets to combine individual staffs.

### Rhythm Parsing

Dewarped images of each staff are computed and passed through a transformer to extract staff contents. From this point onward, semantic information from the sheet music is utilized rather than pixel-based data.

### XML Generation

The previous outputs in terms of result model objects are used to generate music XML.

## Citation

If you use this code in your research work, please cite [oemer](https://github.com/BreezeWhite/oemer) and [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR).

## Name

The name "homr" stands for Homer's Optical Music Recognition (OMR), leaving the interpretation of "Homer" to the user's discretion, whether referring to the ancient poet [Homer](https://en.wikipedia.org/wiki/Homer) or the iconic character from [The Simpsons](https://en.wikipedia.org/wiki/The_Simpsons).

## Thanks

This project builds upon previous work, including:

- The segmentation models of [oemer](https://github.com/BreezeWhite/oemer)
- The transformer model of [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR)
- The starter template provided by [Benjamin Roland](https://github.com/Parici75/python-poetry-bootstrap)
