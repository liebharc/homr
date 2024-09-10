# Transformer training log

This log contains the major milestones of the training progress as well as failed attempts to improve the training.

SER failues shown here should be taken with a grain of salt. The SER is calculated on data which was acquired differently than then training sets, which should make it fair. On the other hand the data is pretty wild and you couldn't expect the SER to be low.

## How to train

Run `training/train.py transformer`. The script will guide you by asking you to install a few things manually and will then continue to download the datasets, convert the datasets and run the training process.

Converting the datasets on itself already takes around 8 hours. This is only required once at the beginning and everytime
an improvement or fix was done to the conversion.

The training process itself takes depending on the hardware you use a 2-3 days.

## Run 101

Commit: ba12ebef4606948816a06f4a011248d07a6f06da
Date: 10 Sep 2024
SER: 9%
Validation result: 6.4

Training runs now also pick the last iteration and not the one with the lowest validation loss.

## Run 100

Commit: e317d1ba4452798036d2b24a20f37061b8441bae
Date: 10 Sep 2024
SER: 14%
Validation result: 7.6

Increased model depth from 4 to 8.

## Run 70

Commit: 11c1eeaf5760d617f09678e276866d31253a5ace
Date: 30 Jun 2024
SER: 16%
Validation result: 8.6

Fixed issue with courtesey accidentals in Lieder dataset and in splitting naturals into tokens.
Removed CPMS dataset as it seems impossible to reliably tell if a natural is in an image.

## Run 62

Commit: 78ace9d99ff38cde0196e47ab2a04309037b1e91
Date: 28 Jun 2024
SER: 17%
Validation result: 8.9

Fixed another issue with backups in music xml. The poorer validation result seems to be
mainly caused by one piece where it fails to detect the naturals.

## Run 57

Commit: e38ad001a548ffd9be89591ce68ed732565a38ae
Date: 21 Jun 2024
SER: 26%
Validation result: 8.1

Fixed an issue with parallel voices in the data set. Added `Lieder` dataset.

## Run 46

Commit: f00a627e030828844c45ecde762146db719d72aa
Date: 9 Jun 2024
SER: 44%
Validation result: 8.0

Set dropout to 0, increased the number of samples and decreased the number of epochs.

## Run 31

Commit: 6a288bc25c99a10cdcdf19982d5df79d65c82910
Date: 16 May 2024
SER: 46%

Removed the negative data set and fixed the ordering of chords.

## Run 18

Commit: 8107bb8bdfaaeb3300477ec534b49cbf1c2a70c6
Date: 12 May 2024
SER: 53%

First training run within the `homr` repo.

## Previous runs

These runs where performed inside the [Polyphonic-TrOMR](https://raw.githubusercontent.com/liebharc/Polyphonic-TrOMR/master/Training.md) repo.

### Run 90 warped data sets, no neg dataset

Date: 6 May 2024
Training time: ~14h (fast option)
Commit: 8f774545179f3e7bfdbd58fe1a6c55473b8d4343

Manual validation result: 14.5

### Run 86 Cleaned up data set

Date: 3 May 2024
Training time: ~14h (fast option)
Commit: b22104265be285b5a1d461c3fab2aa4589eb08cc

Manual validation result: 17.9

### Run 84 More training data with naturals

Date: 1 May 2024
Training time: ~17h (fast option)
Commit: cf7313f0bcec82f4f7da738fbacabd56084f6604

Manual validation result: 17.5

### Run 83 CustomVisionTransformer

Date: 30 Apr 2024
Training time: ~18h (fast option)
Commit: 80896fdba4dbe4f9b2bbba3dd66377b3b0d1faa5

Enabled CustomVisionTransformer again.

### Run 82 Increased alpha to 0.2

Date: 29 Apr 2024
Training time: ~18h (fast option)
Commit: acbdf6dc235f393ef75158bdcf539e3b2e5b435e
Manual validation result: 12.9

Increased alpha to 0.2.

### Run 81 Decreased depth

Date: 29 Apr 2024
Training time: ~18h (fast option)
Commit: 185c235cd0979faa2c087e59e71dbba684a68fb6
Manual validation result: 13.1

Reverting 9e2c14122607a63c25253d1c5378c706859395ab and reverting to a depth of 4.

### Run 80 fixes arround accidentals in the data set

Date: 28 Apr 2024
Training time: ~18h (fast option)
Commit: 840318915929e5efe780780a543ea053b479d375

### Run 79 Use semantic encoding without changes to the accidentals

Date: 27 Apr 2024
Training time: ~18h (fast option)
Commit: f732c3abc10b5b0b3e8942f722d695eb725e3e53
Manual validation result: 80.9

So far we used the format which TrOMR seems to use: Semantic format but with accidentals depending on how they are placed.

E.g. the semantic format is Key D Major, Note C#, Note Cb, Note Cb
so the TrOMR will be: Key D Major, Note C, Note Cb, Note C because the flat is the only visible accidental in the image.

With this attempt we try to use the semantic format without any changes to the accidentals.

### Run 77 Increased depth

Date: 26 Apr 2024
Training time: ~19h (fast option)
Commit: 9e2c14122607a63c25253d1c5378c706859395ab
Manual validation result: 22.3

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
Manual validation result: 13.7

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
Manual validation result: 13.8

Changes to the conversion of the grandstaff dataset haven't been applied yet.

### Run 55 Dropout 0.2

Date: 06 Apr 2024
Training time: ~14h (fast option)
Commit: d73d5a9d342d4d934c21409632f4e2854d14d333
Manual validation result: 17.0

Changes to the conversion of the grandstaff dataset haven't been applied yet.

### Run 51 Dropout 0

Start of dropout tests, number ranges for dropouts are mainly based on https://arxiv.org/pdf/2303.01500.pdf.

Date: 05 Apr 2024
Training time: ~14h (fast option)
Commit: cd445caa5337d86cf723854cb2ef9e98dd4c5b76
Manual validation result: 18.4

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
Manual validation result: 23.4

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

Manual validation result: 9.3
