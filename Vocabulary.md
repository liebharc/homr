# About the transformer vocabulary

The encoding used in this repository is inspired by [Polyphonic-TrOMR](https://github.com/NetEase/Polyphonic-TrOMR), which itself is based on the encoding of [PrIMuS](https://grfia.dlsi.ua.es/primus/).

PrIMuS defines two types of encodings:

- **Agnostic**: describes the symbols on the staff without interpreting their meaning.
- **Semantic**: describes the symbols on the staff with their musical meaning.

For example, agnostic encoding may describe the position of a note on the staff, while semantic encoding describes the actual pitch of that note.

In this project, we use a **hybrid format**. We stay close to semantic, but we borrow from agnostic encoding when it helps reduce the vocabulary size.

Example:

```
clef-G2+keySignature-DM+timeSignature-4/4+note-C4_quarter+G#4_quarter+barline
```

This describes a G clef, a D major key signature, a 4/4 time signature, two quarter notes (C4 and G#4), and a barline.

## Our Vocabulary

Our encoding splits each symbol into **four parallel vocabularies**:

1. **Rhythm**: includes note durations (with augmentation dots), rests, clefs, key signatures, time signatures, barlines, and general musical symbols such as repeats, dynamics, and ties.
2. **Lift**: describes accidentals (sharp, flat, natural, etc.).
3. **Articulation**: describes articulation markings (staccato, accent, trill, etc.).
4. **Pitch**: describes absolute pitches (C0–B9).

### Rules and Customizations

- Accidentals are only encoded if they are explicitly visible in the score image (agnostic rule).
- For time signatures, **only the denominator is encoded**. The numerator can be inferred from the number of notes in a measure. In practice, the model makes little use of numerator information.
- Multi-rests are supported only up to a length of 9.
- Rhythm vocabulary also serves as a catch-all for general non-pitch symbols (e.g., barlines, ties, dynamics).

This separation into four dimensions keeps the vocabulary size manageable while preserving the essential musical information for the transformer model.
