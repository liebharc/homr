# homr

homr is an Optical Music Recognition (OMR) software which takes camera pictures of sheet music and turns them into machine readable [MusicXML](https://www.w3.org/2021/06/musicxml40/). MusicXML can then futher processed e.g. with tools such as [musescore](https://musescore.com/).

## Technical Details

homr uses the segmentation as described in [oemer](https://github.com/BreezeWhite/oemer) to find possible staff lines, clefs, bar lines and note heads on an image. It combines the staff lines, clefs and bars to find the position of the staffs in the picture.

In an next step if passes each staff image into the transformer (based on [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR/tree/master/tromr/model)) to get the symbols on the staff.

Pitch information is double checked with the note head information obtained from the segmentation model.

The results are then converted into MusicXML and written to disk.

## Citation

Please cite [oemer](https://github.com/BreezeWhite/oemer) and [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR/tree/master/tromr/model) if you use this code in your research work.

## Name

homr stands for Homer's Optical Music Recognition (OMR). It's up to you if it's this [Homer](https://en.wikipedia.org/wiki/Homer) or that [Homer](https://en.wikipedia.org/wiki/The_Simpsons).

## Thanks

This project is based on prevous work.

- The segmentation models of [oemer](https://github.com/BreezeWhite/oemer)
- The transformer model of [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR/tree/master/tromr/model)
- The starter template from [Benjamin Roland](https://github.com/Parici75/python-poetry-bootstrap)
