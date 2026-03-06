# Ideas for Contributors

The ideas below outline potential directions for contributors, organized by impact and effort.

## High-Impact Opportunities

The most promising paths to improve **homr** lie in the transformer model. Below are the areas where contributions would have the highest impact.

### 1. Dataset Quality (Highest Priority)

Dataset quality is the **single most important factor** in transformer performance. The existing datasets are likely adequate in scope, but improvements to ground truth accuracy and staff image quality would yield significant gains.

**Focus areas:**

- **Ground truth verification**: Review existing dataset conversions to ensure the extracted symbols accurately represent the source music. Errors in ground truth directly degrade model performance.
- **Staff image quality**: Ensure that dewarped staff images are clear, properly centered, and representative of real-world conditions. Poor image quality limits what the transformer can learn.

Improving these areas requires careful data analysis and domain knowledge but typically yields the best return on investment.

### 2. Data Augmentation & Hyperparameters

Once dataset quality is solid, the next opportunity is optimizing how the model trains on that data.

**Focus areas:**

- **Hyperparameter tuning**: Experiment with learning rate schedules (warmup strategies, decay schedules), batch sizes, optimizer selection (AdamW variants, LAMB), weight decay, and regularization strength. The current settings may not be optimal.
- **Advanced training techniques**: Explore gradient accumulation, mixed precision training, learning rate warmup strategies, cosine annealing, and other modern optimization techniques. Consider layer-wise learning rate adaptation and other advanced methods from recent literature.
- **Training stability**: Investigate training curves to identify instabilities or convergence issues that might be addressed through parameter adjustments or training methodology improvements.

Success here requires patience and systematic experimentation, but no architectural knowledge is needed.

### 3. Transformer Architecture & Model Improvements

As of February 2026, homr appears to deliver state-of-the-art performance for optical music recognition. Further improvements are viable but will likely require substantial effort and expertise.

Work on the transformer itself should only be pursued by contributors with **very strong understanding** of transformer architectures and attention mechanisms. The vision transformer at the core of **homr** (`tromr_arch.py`) has undergone several experimental modifications—some remain in the codebase, but most were reverted due to performance degradation.

**Potential improvements** (for experienced contributors only):

- Review current literature for better vision transformer designs
- Experiment with different attention patterns or positional encodings
- Consider hybrid architectures combining CNNs with transformers
- Collaborate with the broader AI research community

For details on past experiments, see [Training](Training.md) and the Git history (where failed experiments were preserved via reverts).

---

## Complementary Improvements

### Staff Detection

The staff detection module (`staff_detection.py`) has proven robust in real-world scenarios, accurately identifying staffs in most cases. It can also determine how a staff is warped or curved, allowing this information to be used later for dewarping.

### Support More Symbols

Expanding **homr**'s support for additional musical symbols requires building high-quality training data first. Once datasets are improved (see above), adding new symbols becomes tractable.

To add new symbols, you should first understand the [TrOMR architecture as described in the paper](https://arxiv.org/abs/2308.09370). The process is roughly as follows:

1. **Define symbol encoding**

   - Extend an existing vocabulary (`vocabulary.py`) if the symbol stands alone. The rhythm vocabulary, despite its name, also includes symbols like clefs and keys.
   - Create a new branch (update `vocabulary.py`, plus changes to `encoder.py` and other files) if the symbol is combined with others—e.g., accidentals are combined with notes and therefore require their own branch.

2. **Update datasets**

   - Modify conversion scripts (e.g., `convert_lieder.py`).
   - Create new datasets if necessary.
   - Remove datasets that can't reliably define the expected output (e.g., we removed datasets lacking reliable accidental information, as they degraded transformer performance).

3. **Run preliminary training**

   - Train on a subset of datasets to verify that training converges. Increased complexity may require transformer architecture changes.

4. **Update MusicXML generation**
   - Modify `xml_generator.py` to incorporate new symbols in the output.

| Symbol                                  | Status                             | How to Improve                                                                        |
| --------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------- |
| Staffs                                  | ✓                                  | -                                                                                     |
| Clefs                                   | ✓                                  | -                                                                                     |
| Key Signatures                          | ✓                                  | -                                                                                     |
| Time Signatures                         | ✓                                  | -                                                                                     |
| Bars                                    | ✓                                  | -                                                                                     |
| Braces                                  | ✓                                  | Based on segmentation output; transformer only processes a single staff               |
| Notes                                   | ✓, except octave shifts            | Encode octave shifts as their own symbol                                              |
| Accidentals                             | ✓                                  | -                                                                                     |
| Grace notes                             | ✓                                  | -                                                                                     |
| Dotted notes                            | ✓                                  | -                                                                                     |
| Tuplets                                 | ✓                                  | -                                                                                     |
| Rests                                   | ✓, multirests with max 10 measures | Consider alternative encoding to handle larger values without bloating the vocabulary |
| Slurs / Ties                            | ✗ (detected but ignored)           | Quality of detection needs improvement before output can be reliable                  |
| Articulation, Fermata                   | ✓                                  | -                                                                                     |
| Repeats, codas, da capo, volta brackets | ✓                                  | -                                                                                     |
| Glissando                               | ✗                                  | Add `glis_start`, `glis_end` to articulation vocabulary                               |
| Ornaments (trill, turns)                | ✓                                  | -                                                                                     |

### MusicXML Generation

The final stage in **homr** converts transformer output to MusicXML (`music_xml_generator.py`). This stage offers room for improvement, especially for contributors familiar with MusicXML. For example, the current handling of chords and voices could be optimized.
