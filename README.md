# homr

homr is an Optical Music Recognition (OMR) software which takes camera pictures of sheet music and turns them into machine readable [MusicXML](https://www.w3.org/2021/06/musicxml40/). MusicXML can then futher processed e.g. with tools such as [musescore](https://musescore.com/).

## Prequisites

- Python 3.10
- Poetry
- NVidia GPU with CUDA 12.1

## Technical Details

homr uses the segmentation as described in [oemer](https://github.com/BreezeWhite/oemer) to find possible staff lines, clefs, bar lines and note heads on an image. It combines the staff lines, clefs and bars to find the position of the staffs in the picture.

In an next step if passes each staff image into the transformer (based on [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR/tree/master/tromr/model)) to get the symbols on the staff.

Pitch information is double checked with the note head information obtained from the segmentation model.

The results are then converted into MusicXML and written to disk.

### Image predictions

homr uses oemers UNet implementations to separate stafflines and other symbols and to find noteheads.
The predictions are the input of the staff and symbol detection.

In test results we found that preprocessing the image improves the robustness against noisy background and brightness variations.

### Staff and symbol detection

Extracts model data types from the image predictions. A central concept is the "staff anchor": Sheet music can have a lot of symbols which confuse the staff detection. Ledger lines or slurs are examples here. A "staff anchor" is a symbol from which we know for sure that it's never above or below a staff (e.g. note heads change their position) but always on the staff. A second criteria is that we can detect those anchor symbols with high confidence so that we avoid incorrect detections. At the moment we use clefs and bar lines as anchors.

For each anchor we try to find five staff lines. And then we can build the rest of the staff around those anchors.

#### Unit sizes

The unit size describes the distance between staff lines. Due to the camera perspective it can be different depending on where you are on the picture. E.g. the staffs in the upper part of the image might appear larger than in the bottom. We take this into account by calculating the unit size per staff.

#### Connecting staffs

We want to support multiple voices and/or grand staffs. The algorithm tries to find braces and brackets and then combines the individual staffs.

### Rhythm parsing

Calculates dewarped images of every staff and with that the model input extracts staff contents. The output uses the results model as now we no longer work on the image level, but with the abstract content of what is written on the sheet music.

### XML generation

Takes the previous outputs in term of result model objects and generated music XML from it.

## Citation

Please cite [oemer](https://github.com/BreezeWhite/oemer) and [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR/tree/master/tromr/model) if you use this code in your research work.

## Name

homr stands for Homer's Optical Music Recognition (OMR). It's up to you if it's this [Homer](https://en.wikipedia.org/wiki/Homer) or that [Homer](https://en.wikipedia.org/wiki/The_Simpsons).

## Thanks

This project is based on prevous work.

- The segmentation models of [oemer](https://github.com/BreezeWhite/oemer)
- The transformer model of [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR/tree/master/tromr/model)
- The starter template from [Benjamin Roland](https://github.com/Parici75/python-poetry-bootstrap)
