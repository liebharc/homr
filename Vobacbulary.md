# About the transformer vocabulary

The encoding uses in this repository is based on the encoding of [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR) which is based on the encoding of [PrIMuS](https://grfia.dlsi.ua.es/primus/).

PrIMus describes two types of formats:

- agnostic: the symbols on the staff without an interpretation of their meaning
- semantic: the symbols on the staff with an interpretation of their meaning

E.g. agnostic describes the position of a note on the staff, semantic describes the pitch of the note.

For this project we use a hybrid of the two formats. We stay close to semantic, but in
order to reduce the vocabulary size we use agnostic symbols in some cases.

Take this example:

```
clef-G2+keySignature-DM+timeSignature-4/4+note-C4_quarter+G#4_quarter+barline
```

This describes a G-clef, D major key signature, 4/4 time signature, two quarter notes (C#4 and G#4), and a barline.

The following rules apply:

- Like in the agnostic format, accidentals are only encoded if they have a symbol visible on the image

The following customizations have been done to reduce the vocabulary size and this way improve the model performance:

- For time signatures we only condider the denominator as we can easily infer the numerator from the number of notes in a measure. The transformer does't seem to make a lot of use from the time signature information.
- multirests are only supported up to a length of 9
