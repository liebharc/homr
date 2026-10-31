# About the transformer vocabulary

The vocabulary is closely linked to how the transformer operates. Symbols are divided into several branches. For example, `note_4 D5 # accent _ upper` represents a quarter note with pitch D5, a sharp accidental, and an accent articulation. The branches for each symbol are:

1. **Rhythm**: covers note durations (including augmentation dots), rests, clefs, key signatures, time signatures, barlines, and general musical symbols such as repeats and dynamics. The term _rhythm_ can therefore be misleading, as it encodes more than just rhythmic values.
2. **Pitch**: represents absolute pitches (C0–B9).
3. **Lift**: represents accidentals (sharp, flat, natural, etc.).
4. **Articulation**: represents articulation markings (staccato, accent, trill, etc.).
5. **Slur**: represents slurs and ties, and beams, see [Beams](#beams).
6. **Position**: represents staff position for notes, rests, and clefs (upper or lower staff), see [Positions and voices](#positions-and-voices).

The vocabulary follows these rules:

1. All branches except rhythm are optional. For example, the symbol for a key signature is `keySignature_0 . . . . .`, where the dots indicate that a key signature has no pitch, lift, articulation, slur or position.
2. Symbols that occur simultaneously appear on one line and are separated by `&`:
   - notes and rests may occur at the same time
   - multiple clefs may occur at the same time
   - all other symbols are expected to appear sequentially
   - `&` serves as shorthand for `chord . . . . .`
3. Barlines and repeats mark the end of a measure.
4. Note durations follow the conventions of [Humdrum Kern](https://www.humdrum.org/guide/ch06/), which are also related to how [MusicXML encodes durations](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/duration/):
   - e.g. `4, 8` are quarter and eighth notes
   - `4.` is a dotted quarter note
   - `12` represents one note of an eighth-note triplet
5. Key signatures follow the [circle of fifths definition](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/fifths/).
6. Time signatures contain only the [beat type](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/time/). The number of beats is reconstructed from the median measure length.
7. Multiple articulations may appear together, e.g. `note_4 D5 # accent_arpeggiate_tenuto`. Not all combinations are supported, since the transformer can only generate symbols it encountered during training. Refer to the vocabulary for the full list.

The complete vocabulary is defined in `homr/transformer/vocabulary.py`.

## Positions and voices

The position branch can be `upper`, `upper2`, `lower` or `lower2`:

- `upper` and `lower` refer to the upper and lower staff of a grand staff. A single staff always uses `upper`.
- The `2` suffix marks a second voice on the same staff. Stem direction tells the voices apart: notes with stems up get `upper`/`lower`, notes with stems down get `upper2`/`lower2`.

This was introduced in [#156](https://github.com/liebharc/homr/pull/156).

## Beams

The slur branch also holds the state of the primary beam of a note: `beamStart`, `beamContinue` or `beamStop`, e.g. `beamStop_slurStart`. Notes without a beam, e.g. an eighth note with a flag or a quarter note, have none.

- A chord has one beam state per position, on its first note which can have a beam.
- Rests don't have a beam state and don't interrupt a beam.
- Secondary beams and hooks follow from the durations, `homr/music_xml_generator.py` adds them.
- A beam can cross a barline and change the staff. A beam which continues on the other staff is drawn in one voice.

## Full example

```
clef_G2 _ _ _ _ upper
keySignature_0 . . . . .
timeSignature/8 . . . . .
note_4. C5 _ _ _ upper&note_4. E4 _ _ _ upper
note_8 D5 _ _ beamStart upper
note_16 E5 _ staccato beamContinue_slurStart upper
note_16 F5 # _ beamStop_slurStop upper
note_8 G5 _ accent _ upper
barline . . . . .
```
