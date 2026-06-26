# Training log

This log contains the major milestones of the training process, as well as failed attempts to improve training. The focus of the training of the transformer and the segmentation model is documented only briefly.

## How to train

### Segmentation model (segnet)

`training/train.py segnet`

This will download the dataset and start the training. Training takes about 1 hour.

### Transformer (TrOmr)

Prerequisites:

- Linux
- CUDA
- rsvg-convert, e.g. via `sudo apt install librsvg2-bin`
- libfuse, e.g. via `sudo apt install libfuse2t64` on Ubuntu24 or `sudo apt install libfuse2` on Ubuntu22/20
- libjack, e.g. via `sudo apt install libjack-jackd2-0`

Download the datasets and convert them to the format required for training:

- `training/datasets/convert_primus.py`
- `training/datasets/convert_grandstaff.py`
- `training/datasets/convert_lieder.py`
  - This will also download and run MuseScore as an AppImage. If this fails, check your setup to ensure that you can run `datasets/MuseScore`.
  - Not all files are supported. At the end you'll see something like `Processed 1460/1467 files, skipped 350 files`, which is as expected.

Some checks:

- `training/transformer/training_vocabulary.py`: Check that the vocabulary is complete
- `training/validate_music_xml_conversion.py`: Can be used to test changes in MusicXML parsing and generation
- `training/validate_music_xml_conversion.py`: Visualize datasets; takes one of `datasets/*/index.txt` as an argument

Finally, start the training itself with: `training/train.py transformer`.  
You can check the log and see number of training files to be 190k: `Total number of training files to choose from 190722`
This takes around 2–4 days.

Batch size for FP16 - don't forget to also modify gradient_accumulation_steps:

- 8GB VRAM: 8
- 16GB VRAM: 18 (default)
- 24GB VRAM: 32

Distributed Training:

- Distributed training supports single-machine, multi-GPU setups. Using 4 GPUs as an example, run: `CUDA_VISIBLE_DEVICES=0,1,2,3 poetry run torchrun --standalone --nnodes=1 --nproc_per_node=4 training/train.py transformer`
- Dataset generation and model loading/saving are handled by rank0.
- It is recommended to use 2 or 4 GPUs to keep training consistent with single-GPU training. For other GPU counts, you need to manually adjust the `gradient_accumulation_steps` parameter. Multi-node, multi-GPU training is theoretically supported, but has not been tested in practice.

Cloud Training:

- Homr can be trained on community clouds like vast.ai since the dataset is opensource and only 12GB big (can be uploaded to google drive for free)
- As homr only uses bf16 training and doesn't use newer features like FP8, the RTX 3090 seems to provide the best value
- Homr's data loading requires a good CPU: try to get something similar to an i5-11400

## Results

The `homr` pipeline is a two-step system consisting of **staff detection (segnet)** followed by **music transcription (TrOmr transformer)**.  
Accordingly, we report results from two different validation setups with different purposes and scopes.

### Transcription Smoke Test

The _Transcription Smoke Test_ evaluates only the transformer-based transcription component on a small, fixed dataset.  
It is intended as a **fast indicator** of transcription model quality and is mainly used to detect regressions or larger performance changes during development.

Implementation: `symbol_error_rate_torch.py`

### System Level Validation

The _System Level Validation_ evaluates the full `homr` pipeline, including staff detection and transcription, on a dedicated test dataset.  
This validation provides a **more representative indication of overall system performance** and is used to compare training runs and pipeline changes.

**Note:** The test dataset cannot be published due to copyright restrictions. In addition, the dataset is subject to change over time, which may affect the comparability of results across different runs.

Implementation: `rate_validation_result.py`

## Run 407 - at epoch 6

Commit: fabab4b0f8480be20d13686edf5f91d913ab00fd
Day: 26 June 2026
Transformer Smoke Test: 17%
System Level: 9.3 diffs, SER: 7.4%
Polish scores with musicdiff: OMR-NED 46.4%

Quick training run after updates to the dataset conversions. Results look okay for only 6 epochs.

## Run 396

Commit: f6feedb42ff90087d898b0941a55d040fa6b2903
Day: 12 June 2026
Transformer Smoke Test: 7%
System Level: Total: 6.7 diffs, SER: 5.2%

Final training run for #86 (https://github.com/liebharc/homr/pull/86), up to date with all changes on main branch

## Run 396 - discarded at epoch 9

Commit: f6feedb42ff90087d898b0941a55d040fa6b2903
Day: 11 June 2026
Transformer Smoke Test: -

Reintroduced tie vocabulary and stopped adding all slurs&ties on the first note of a chord. The model performed significantly worse on slurs&ties than the runs before. This run was part of #86 (https://github.com/liebharc/homr/pull/86), code can be found (https://github.com/aicelen/homr/tree/feature/improve-slurs-ties).

## Run 396 - discarded at epoch 10

Commit: f6feedb42ff90087d898b0941a55d040fa6b2903
Day: 9 June 2026
Transformer Smoke Test: SER avg: 11%

Removed problematic files from grandstaff containing more slurStart than slurStop (https://github.com/liebharc/homr/pull/86)

## Run 396 - discarded at epoch 12

Commit: f6feedb42ff90087d898b0941a55d040fa6b2903
Day: 30 May 2026
Transformer Smoke Test: SER avg: 9%

Split articulations&slurs into two branches.
This resulted in the model being too eager to predict slurStart (https://github.com/liebharc/homr/pull/86)

## Run 384 - discarded

Commit: 0fee4303aeb8bb21a10fd0d7b457485a6d22fa0d
Day: 5 Jun 2026
Transformer Smoke Test: SER avg: 6%
System Level: 6.4 diffs, SER: 5.2%

Fusing 1&2

## Run 385 - discarded

Commit: 620d4591bcd716b722c894bc5546dc8f5d5e3a1c
Day: 5 Jun 2026
Transformer Smoke Test: SER avg: 7%
System Level: 6.5 diffs, SER: 4.9%

Fusing 1&2&3

## Run 386 - discarded

Commit: c1c0e3b126f2e2971c565197ce10bd752ad02687
Day: 5 Jun 2026
Transformer Smoke Test: SER avg: 6%
System Level: 8.4 diffs, SER: 5.5%

Fusing 2&3

## Run 381 - discarded

Commit: 6ced21726443ed037608f8610ff4e7dac445649a
Day: 1 Jun 2026
Transformer Smoke Test: SER 6%
System Level: 5.6 diffs, SER: 3.1%

FPN Style fusion, https://github.com/liebharc/homr/pull/85 . The run is on good as e.g. #367 and #331, but the model is more complex and a manual review of the results indicate a less robust pitch detection.

A possible way forward would be to not mix fused features, but instead concat stage 2 and stage 3. That however
increases the encoder_dim and decoder_dim from 512 to 1152 (768 + 384) which makes training and inference much more expensive.

## Run 367

Commit: 575b4737bca815d3a7b37169269fc548d7e945b9, which after a rebase is 0b865c41e56a1e58bf531ada27ad80ea1f5ab716
Day: 29 May 2026  
Transformer Smoke Test: SER 6%
System Level: 5.6 diffs, SER: 4.7%

Fixed an error in the accuracy calculation during training: https://github.com/liebharc/homr/pull/84

## Run 366

Commit: ac68c5e38711de6abaa6064eaf2773aa49352d01
Day: 29 May 2026  
Transformer Smoke Test: SER 6%
System Level: 5.8 diffs, SER: 4.7%

Using a newer MuseScore version for the Lieder dataset: https://github.com/liebharc/homr/pull/75

## Run 368 - discarded

Commit: aad3b0899cee5b931aae3dd4345d371a8634669c
Day: 24 May 2026  
Transformer Smoke Test: SER 5%
System Level: 12.3 diffs, SER: 16.2%

Agnostic data format, with less complexity when it comes to pitch and accidentals.

## Run 367 - discarded

Commit: 39ff998262112d75b425cdd99c43ab61441a3f4e
Day: 22 May 2026  
Transformer Smoke Test: SER 7%
System Level: 7.0 diffs, SER: 5.2%

Updated MuseScore version used to convert the Lieder dataset and switched to sinusoidal positional embedding.

## Run 334 - discarded

Commit: 6d996c3d118c1e183f8412832383168e52630ce8
Day: 17 Feb 2026  
Transformer Smoke Test: SER 6%
System Level: 10.2 diffs, SER: 13.6%

convnextv2_base, https://github.com/liebharc/homr/pull/59

## Run 333 - discarded

Commit: bb0ced2ff0cacdbd8ee33db4533a04c9e77f0ca8
Day: 17 Feb 2026  
Transformer Smoke Test: SER 18%
System Level: 14.4 diffs, SER: 14.5%

convnextv2_tiny, https://github.com/liebharc/homr/pull/59

## Run 331

Commit: e10346542968cc71fbcce0c0696f3ac963f11ae1
Day: 17 Feb 2026  
Transformer Smoke Test: SER 6%
System Level: 5.3 diffs, SER: 3.4%

Scheduled Sampling, https://github.com/liebharc/homr/pull/59

## Run 328 Epch 21

Commit: a78209527e2b8a4fb866fba9b2ef8540f4b8dad9
Day: 14 Feb 2026  
Transformer Smoke Test: SER 10%
System Level: 16.0 diffs, SER: 9.3%

Harder augmentation, https://github.com/liebharc/homr/pull/59

## Run 326 Epoch 13

Commit: 290d4e79aa377681523ca676b984b9cee3eb16ce
Day: 13 Feb 2026  
Transformer Smoke Test: SER 13%
System Level: 42.8 diffs, SER: 32.8%

Backtracking, removed 90deg rotation and sinusoidal bias, https://github.com/liebharc/homr/pull/59

## Run 322 Epoch 14

Commit: fd3d66d7d989003ec4cadd1d594ca2e820ece941
Day: 12 Feb 2026  
Transformer Smoke Test: SER 12%
System Level: 18.1 diffs, SER: 22.4%

Improved pitch accuracy, https://github.com/liebharc/homr/pull/59

## Run 317

Commit: 6f72a0bc2577907503e7ec84ac9850a5a972ded0
Day: 4 Feb 2026  
Transformer Smoke Test: SER 15%
System Level: 25.9 diffs, SER: 15.5%

ConvNext, https://github.com/liebharc/homr/pull/59

## Run 286 - after segnet update

Commit: 87d30ed79a81b4f07a38a8f6419334c59633709a  
Day: 30 Jan 2026  
Transformer Smoke Test: SER 14% (the SER reported for the previous runs was too large due to an unreasonable large temperature setting during the smoke test)
System Level: 6.9 diffs, SER: 5.7%

Updated segnet model for staff detection.

## Run 286

Commit: 0daf75fea21e6ea6a865405e03a4bc7e73e9aa14  
Day: 4 Jan 2026  
Transformer Smoke Test: SER 26% (higher, due to an error in the smoke test)
System Level: 7.4 diffs

After fixing an issue with accidentals during the conversion of the PrIMuS dataset Run 242 (a00be6debbedf617acdf39558c89ba6113c06af3)
was used as basis of a 15 epoch run which only trained the lift decoder.

## Run 242

Commit: a00be6debbedf617acdf39558c89ba6113c06af3  
Day: 9 Dec 2025  
Transformer Smoke Test: SER 23% (some errors in the smoke test fixed, still higher as some issue remained)
System Level: 7.2 diffs after fixing an error in the validation result calculation itself, before 8.1

Singe staff images now use the full resolution.

## Run 236

Commit: 922ad08f8895f6d9c0ae61954cd78a021ff950a7  
Day: 26 Oct 2025  
Transformer Smoke Test: SER 37% (higher, due to an error in the smoke test)
System Level: 9.8 diffs, 8.6 after some tweaks to the staff image preparation

Volta brackets, bf16

## Run 234

Commit:ea96f0150ec74388df8cb0bb78ee2c36782a00d9  
Day: 01 Oct 2025  
Transformer Smoke Test: SER 39% (higher, due to an error in the smoke test)
System Level: 9.8 diffs

Grandstaff support.

Some notes about other experiments which have been performed on the lieder dataset with only 15 epochs:

- Baseline
  - Final eval loss: 0.6007007360458374
  - SER: 112%
- State
  - Final eval loss: 0.5414645671844482
  - SER: 109%
- Clef unification
  - Final eval loss: 1.849281907081604
  - SER: 146%
- fp16 and flash attention
  - Final eval loss: 1.522333025932312
  - SER: 144%
- fp16 and no flash attention
  - Final eval loss: 1.00400710105896
  - SER: 130%
  - Trains about twice as fast so the poorer performance can be adressed with additional epochs

## Run 220

Commit: c50aec7de6469480cf6f547695f48aed76d8422e  
Day: 05 Sep 2025  
Transformer Smoke Test: SER 8%  
System Level: 8.8 diffs

Added articulations and other symbols.

## Run 197

Commit: 4c8d68b941c647c96f82d977ac0bb59d4f2b7a8c  
Day: 05 Aug 2025  
Transformer Smoke Test: SER 10%  
System levle: 9.1 diffs

New decoder branch just for triplets and dots. The result works mostly fine but it's too eager to detect triplets.

## Run 186 after adding triplet correction

Commit: 4915073f892f6ab199844b1bff0c968cdf8be03e  
Day: 01 Aug 2025  
Transformer Smoke Test: SER 8%  
System Level: 8.0 diffs

Larger encoder.

## Run 186

Commit: 4915073f892f6ab199844b1bff0c968cdf8be03e  
Day: 01 Aug 2025  
Transformer Smoke Test: SER 8%  
Sytem level: 8.3 diffs

Larger encoder.

## Run 186

Commit: 3f0631db15012e928ad3d4da739817f92d958979  
Day: 30 Jul 2025  
Transformer Smoke Test: SER 10%  
System levle: 7.9 diffs

Removed cases of incorrect triplets from the dataset

## Run 183

Commit: 74d500a5d94e553f24dbbd57a0e71b8566e2e554  
Day: 25 Jul 2025  
Transformer Smoke Test: SER 11%  
System levle: 9.6 diffs

Larger encoder.

Note: This branch was rebased, commit hash was updated to match the version which was merged to main.

## Run 181

Commit: a1ec2fff7d7ba562807f03badf5ed963b48649a5  
Day: 27 Jul 2025  
Transformer Smoke Test: SER 9%  
System Level: 11.6 diffs

Increased degrees of freedom in decoder.

## Run 180

Commit: eb5fbfd4692b56d24d615e2fa3586903ad681132  
Day: 26 Jul 2025  
Transformer Smoke Test: SER 8%  
System Level: 9.7 diffs

Added triplets.

## Run 161

Commit: 1cd1d06543e885e4d64a74d985b4725c50054c2a  
Day: 11 Jul 2025  
Transformer Smoke Test: SER 10%  
System levle: 8.0 diffs

Transformer depth of 8.

## Run 160

Commit: bf39c935c9081d04dc1d97e25dcda68ebb0ca40c  
Day: 10 Jul 2025  
Transformer Smoke Test: SER 9%  
System Level: 7.6 diffs

Transformer depth of 6.

## Run 159

Commit: a9dd113eb203979b6c2b21403574832da39fee76  
Day: 09 Jul 2025  
Transformer Smoke Test: SER 11%  
System levle: 8.4 diffs

Transformer depth of 4 (as Polymorphic-TrOmr is using). Training was stopped at epoch 59 by a Windows Update.

## Run 152 after update of the staff detection (1240eedca553155b3c75fc9c7f643465383430a0)

Commit: 46ff7e18fd85d9d2026f9ed18eacf7ae0638a14c  
Day: 07 Jul 2025  
Transformer Smoke Test: SER 9%  
System levle: 7.4 diffs

Updated staff detection:

- Resnet18 after 3 epochs (1240eedca553155b3c75fc9c7f643465383430a0): 7.4
- Resnet18 after 10 epochs (66dd2392759d1746cc9458c097e25aaaa1559fc5): 10.8 (overfitting?)
- Resnet34 after 3 epochs (1cd1d06543e885e4d64a74d985b4725c50054c2a): 7.3
- Resnet34 after 10 epochs (a9dd113eb203979b6c2b21403574832da39fee76): 8.3

Note at this point the transformer depth is 8 for the decoder and 12 for the encoder.

## Run 152

Commit: 46ff7e18fd85d9d2026f9ed18eacf7ae0638a14c  
Day: 05 Jul 2025  
Transformer Smoke Test: SER 9%  
System levle: 8.3 diffs

Updated dependencies.

## Run 140

Commit: 4a0d7991b3824f2a667a237b1370a8999cd3695e  
Day: 29 Jun 2025  
Transformer Smoke Test: SER 9%  
System Level: 7.8 diffs

Updated dependencies.

## Run 101

Commit: ba12ebef4606948816a06f4a011248d07a6f06da  
Date: 10 Sep 2024  
Transformer Smoke Test: SER 9%  
System Level: 6.4 diffs

Training runs now pick the last iteration and not the one with the lowest validation loss.

## Run 100

Commit: e317d1ba4452798036d2b24a20f37061b8441bae  
Date: 10 Sep 2024  
Transformer Smoke Test: SER 14%  
System Level: 7.6 diffs

Increased model depth from 4 to 8.

## Run 70

Commit: 11c1eeaf5760d617f09678e276866d31253a5ace  
Date: 30 Jun 2024  
Transformer Smoke Test: SER 16%  
System Level: 8.6 diffs

Fixed issue with courtesey accidentals in Lieder dataset and in splitting naturals into tokens.
Removed CPMS dataset as it seems impossible to reliably tell if a natural is in an image.

## Run 62

Commit: 78ace9d99ff38cde0196e47ab2a04309037b1e91  
Date: 28 Jun 2024  
Transformer Smoke Test: SER 17%  
System Level: 8.9 diffs

Fixed another issue with backups in music xml. The poorer validation result seems to be
mainly caused by one piece where it fails to detect the naturals.

## Run 57

Commit: e38ad001a548ffd9be89591ce68ed732565a38ae  
Date: 21 Jun 2024  
Transformer Smoke Test: SER 26%  
Sytem levle: 8.1 diffs

Fixed an issue with parallel voices in the data set. Added `Lieder` dataset.

## Run 46

Commit: f00a627e030828844c45ecde762146db719d72aa  
Date: 9 Jun 2024  
SER: 44%  
System Level: 8.0 diffs

Set dropout to 0, increased the number of samples and decreased the number of epochs.

## Run 31

Commit: 6a288bc25c99a10cdcdf19982d5df79d65c82910  
Date: 16 May 2024  
Transformer Smoke Test: SER 46%

Removed the negative data set and fixed the ordering of chords.

## Run 18

Commit: 8107bb8bdfaaeb3300477ec534b49cbf1c2a70c6  
Date: 12 May 2024  
Transformer Smoke Test: SER 53%

First training run within the `homr` repo.

## Previous runs

These runs where performed inside the [Polyphonic-TrOMR](https://raw.githubusercontent.com/liebharc/Polyphonic-TrOMR/master/Training.md) repo.

### Run 90 warped data sets, no neg dataset

Date: 6 May 2024  
Training time: ~14h (fast option)  
Commit: 8f774545179f3e7bfdbd58fe1a6c55473b8d4343

System Level: 14.5 diffs

### Run 86 Cleaned up data set

Date: 3 May 2024  
Training time: ~14h (fast option)  
Commit: b22104265be285b5a1d461c3fab2aa4589eb08cc

System Level: 17.9 diffs

### Run 84 More training data with naturals

Date: 1 May 2024  
Training time: ~17h (fast option)  
Commit: cf7313f0bcec82f4f7da738fbacabd56084f6604

System levle: 17.5 diffs

### Run 83 CustomVisionTransformer

Date: 30 Apr 2024  
Training time: ~18h (fast option)  
Commit: 80896fdba4dbe4f9b2bbba3dd66377b3b0d1faa5

Enabled CustomVisionTransformer again.

### Run 82 Increased alpha to 0.2

Date: 29 Apr 2024  
Training time: ~18h (fast option)  
Commit: acbdf6dc235f393ef75158bdcf539e3b2e5b435e  
System Level: 12.9 diffs

Increased alpha to 0.2.

### Run 81 Decreased depth

Date: 29 Apr 2024  
Training time: ~18h (fast option)  
Commit: 185c235cd0979faa2c087e59e71dbba684a68fb6  
System levle: 13.1 diffs

Reverting 9e2c14122607a63c25253d1c5378c706859395ab and reverting to a depth of 4.

### Run 80 fixes arround accidentals in the data set

Date: 28 Apr 2024  
Training time: ~18h (fast option)  
Commit: 840318915929e5efe780780a543ea053b479d375

### Run 79 Use semantic encoding without changes to the accidentals

Date: 27 Apr 2024  
Training time: ~18h (fast option)  
Commit: f732c3abc10b5b0b3e8942f722d695eb725e3e53  
System Level: 80.9 diffs

So far we used the format which TrOMR seems to use: Semantic format but with accidentals depending on how they are placed.

E.g. the semantic format is Key D Major, Note C#, Note Cb, Note Cb
so the TrOMR will be: Key D Major, Note C, Note Cb, Note C because the flat is the only visible accidental in the image.

With this attempt we try to use the semantic format without any changes to the accidentals.

### Run 77 Increased depth

Date: 26 Apr 2024  
Training time: ~19h (fast option)  
Commit: 9e2c14122607a63c25253d1c5378c706859395ab  
System Level: 22.3 diffs

Encoder & decoder depth was increased from 4 to 6

### Run 76 Training data fix for accidentals

Date: 25 Apr 2024  
Training time: ~16h (fast option)  
Commit: 75d8688719494169f4b629fc51224d4aa846eee7

Fixed that the training data didn't contain any natural accidentals.

### Run 74 Backtracking

Date: 24 Apr 2024  
Training time: ~24h (fast option)  
Commit: b4af54249fca5bf93650c518c7220f5de98c843c

After experiments with focal loss and weight decay, we are backtracking to run 63.

### Run 74

Date: 23 Apr 2024  
Training time: ~24h (fast option)  
Commit: 6580500e71602d5c74decde2946498c8e883392e

Adding a weight to the lift/accidental tokens.

### Run 71 Weight decay

Date: 22 Apr 2024  
Training time: ~17h (fast option)  
Commit: 3b92eee2e56647fcb538b4ef5ef3704f12bfb2d1

Reduced weight decay.

### Run 70 Focal loss

Date: 21 Apr 2024  
Training time: ~17h (fast option), aborted after epoch 16 from 25  
Commit: a6b87b71b3b69d87d424f3c86500081f6146d436

Looks like a focal loss doesn't help to improve the performance of the lift detection.

### Run 63 Negative data set

Date: 11 Apr 2024  
Training time: ~26h (fast option)  
Commit: c360ab726df18879973e6829a1423c627a99afd5  
System Level: 13.7 diffs

Increased data set size by introducing a negative data set with no musical symbols. And by using positive data sets more often with different mask values.

### Run 57 Dropout 0.8

Date: 07 Apr 2024  
Training time: ~14h (fast option)  
Commit: 3fc893c0ab547fe1958adf500b0afaf0f6990f80

Changes to the conversion of the grandstaff dataset haven't been applied yet.

### Run 56 Dropout 0.1

Date: 07 Apr 2024  
Training time: ~14h (fast option)  
Commit: 5ec6beaf461c034340ad0d2f832d842bef8bee75  
System Level: 13.8 diffs

Changes to the conversion of the grandstaff dataset haven't been applied yet.

### Run 55 Dropout 0.2

Date: 06 Apr 2024  
Training time: ~14h (fast option)  
Commit: d73d5a9d342d4d934c21409632f4e2854d14d333  
System Level: 17.0 diffs

Changes to the conversion of the grandstaff dataset haven't been applied yet.

### Run 51 Dropout 0

Start of dropout tests, number ranges for dropouts are mainly based on https://arxiv.org/pdf/2303.01500.pdf.

Date: 05 Apr 2024  
Training time: ~14h (fast option)  
Commit: cd445caa5337d86cf723854cb2ef9e98dd4c5b76  
System Level: 18.4 diffs

### Run50 InceptionResnetV2

We changed how we number runs and established a link between the run number and the git history.

Date: 05 Apr 2024  
Training time: ~19h (fast option)  
Commit: a57ee4c046842c0135adca84f06260cff8af732f

We tried InceptionResnetV2. The training run showed overfitting and the resulting SER indicates poor results. The model is over 3 times larger than the ResNetV2 model and might require more work to prevent overfitting.

### Run3

Date: 02 Apr 2024  
Training time: ~24h (fast option)  
Commit: 9ddfff8b5782473e8831ca3791d9bef99f726654  
System Level: 23.4 diffs

We decreased the vocabulary, the alpha/beta ratio in the loss function and made changes to the grandstaff dataset. While still performing worse than Run 0 in the manual validation, it gets closer now and in some specific tests performs even better than Run 0. We will have to backtrack from this point to find out which of the changes lead to an improved result.

### Run2

Date: 01 Apr 2024  
Training time: ~48h  
Commit: 516093a3f3840cb82922b4d7300d1568455277d568f85ea96fe41235a06ca8de6759f1db6b8fc39a

### Run1

Date: 24 Mar 2024  
Training time: ~24h (fast option)  
Commit: 516093a3f3841235a06ca8de6759f1db6b8fc39a

### Run 0

The weights from the [original paper](https://arxiv.org/abs/2308.09370).

System Level: 9.3 diffs
