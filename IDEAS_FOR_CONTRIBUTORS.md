# Ideas for Contributors

The ideas below outline potential directions for contributors.

## Staff Detection

The staff detection module (`staff_detection.py`) has proven robust in real-world scenarios, accurately identifying staffs in most cases. It can also determine how a staff is warped or curved, allowing this information to be used later for dewarping.

The most promising area for improvement is speed. The current algorithm could likely be optimized to reduce processing time.

## Support More Symbols

One of the most impactful enhancements would be to expand **homr**’s support for additional musical symbols. The main challenge lies in building a high-quality dataset for training. A strong dataset is essential—without it, the transformer model cannot reliably learn to extract new symbols. In our experience, dataset quality is the single most important factor in model performance.

To add new symbols, you should first understand the [TrOMR architecture as described in the paper](https://arxiv.org/abs/2308.09370). The process is roughly as follows:

1. **Define symbol encoding**

   - Extend an existing vocabulary (`vocabulary.py`) if the symbol stands alone. The rhythm vocabulary, despite its name, also includes symbols like clefs and keys.
     - Create a new branch (update `vocabulary.py`, plus changes to `encoder.py` and other files) if the symbol is combined with others—e.g., accidentals are combined with notes and therefore require their own branch.

2. **Update datasets**

   - Modify conversion scripts (e.g., `convert_lieder.py`).
   - Create new datasets if necessary.
   - Remove datasets that can’t reliably define the expected output (e.g., we removed datasets lacking reliable accidental information, as they degraded transformer performance).

3. **Run preliminary training**

   - Train on a subset of datasets to verify that training converges. Increased complexity may require transformer architecture changes.

4. **Update MusicXML generation**

   - Modify `xml_generator.py` to incorporate new symbols in the output.

| Symbol                                  | Status                             | How to Improve                                                                        |
| --------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------- |
| Staffs                                  | ✓                                  | —                                                                                     |
| Clefs                                   | ✓                                  | —                                                                                     |
| Key Signatures                          | ✓                                  | —                                                                                     |
| Time Signatures                         | ✓                                  | —                                                                                     |
| Bars                                    | ✓                                  | —                                                                                     |
| Braces                                  | ✓                                  | Based on segmentation output; transformer only processes a single staff               |
| Notes                                   | ✓, except octave shifts            | Encode octave shifts as their own symbol                                              |
| Accidentals                             | ✓                                  | —                                                                                     |
| Grace notes                             | ✓                                  | —                                                                                     |
| Dotted notes                            | ✓                                  | —                                                                                     |
| Triplets                                | ✓                                  | —                                                                                     |
| Rests                                   | ✓, multirests with max 10 measures | Consider alternative encoding to handle larger values without bloating the vocabulary |
| Slurs / Ties                            | ✗                                  | Slurs and ties are detected but the results aren't great yet                          |
| Articulation, Fermata                   | ✓                                  | —                                                                                     |
| Repeats, codas, da capo, volta brackets | ✓                                  | —                                                                                     |
| Glissando                               | ✗                                  | Add `glis_start`, `glis_end` to rhythm vocabulary                                     |
| Ornaments (trill, turns)                | ✓                                  | —                                                                                     |

## Transformer

The vision transformer at the core of **homr** (`tromr_arch.py`) has undergone several experimental modifications—some remain in the codebase, but most were reverted. Given ongoing research in the field, better architectures may now exist. It could be valuable to review current literature and collaborate with the AI community to identify improvements—or even replace the model entirely.

For details on past experiments, see [Training](Training.md) and the Git history (where failed experiments were preserved via reverts).

## MusicXML Generation

The final stage in **homr** converts transformer output to MusicXML (`music_xml_generator.py`). This stage likely offers significant room for improvement, especially for contributors familiar with MusicXML. For example, the current handling of chords and voices is likely suboptimal.
