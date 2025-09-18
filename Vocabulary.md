# About the transformer vocabulary

The vocabulary is closely linked to how the transformer operates. Symbols are divided into several branches. For example, `note_4 D5 # accent` represents a quarter note with pitch D5, a sharp accidental, and an accent articulation. The branches for each symbol are:

1. **Rhythm**: covers note durations (including augmentation dots), rests, clefs, key signatures, time signatures, barlines, and general musical symbols such as repeats, dynamics, and ties. The term _rhythm_ can therefore be misleading, as it encodes more than just rhythmic values.
2. **Pitch**: represents absolute pitches (C0–B9).
3. **Lift**: represents accidentals (sharp, flat, natural, etc.).
4. **Articulation**: represents articulation markings (staccato, accent, trill, etc.).

The vocabulary follows these rules:

1. All branches except rhythm are optional. For example, the symbol for a key signature is `keySignature_0 . . .`, where the dots indicate that a key signature has no pitch, lift, or articulation.
2. Symbols that occur simultaneously appear on one line and are separated by `&`:
   - notes and rests may occur at the same time
   - multiple clefs may occur at the same time
   - all other symbols are expected to appear sequentially
   - `&` serves as shorthand for `chord _ _ _`
3. Barlines and repeats mark the end of a measure.
4. Note durations follow the conventions of [Humdrum Kern](https://www.humdrum.org/guide/ch06/), which are also related to how [MusicXML encodes durations](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/duration/):
   - e.g. `4, 8` are quarter and eighth notes
   - `4.` is a dotted quarter note
   - `12` represents one note of an eighth-note triplet
5. Key signatures follow the [circle of fifths definition](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/fifths/).
6. Time signatures contain only the [beat type](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/time/). The number of beats is reconstructed from the median measure length.
7. Multiple articulations may appear together, e.g. `note_4 D5 # accent_arpeggiate_tenuto`. Not all combinations are supported, since the transformer can only generate symbols it encountered during training. Refer to the vocabulary for the full list.

The complete vocabulary is defined in `homr/transformer/vocabulary.py`.

## Full example

```
clef_G2 . . .
keySignature_4 . . .
timeSignature/8 . . .
note_4. G3 # _&note_4. C4 # _&note_16 E4 # _
note_16 F4 # _
note_4 E4 # _
note_8 E4 # _
note_8 C4 # _
note_8 D4 # _
barline . . .
```
