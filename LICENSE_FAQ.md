# License FAQ

This is not legal advice. homr is a non-commercial volunteer project. If you use it commercially, the legal questions and risks are yours.

## Important: Do not send emails

Do not send emails with license questions. The workload really gets too much. We are not able to provide legal advice. This FAQ is meant to help with the main questions. If it doesn't then ask your lawyer for advice, and not us as open source contributors.

## Code

The code is licensed under [AGPL-3.0](LICENSE). The AGPL allows commercial use as long as you meet its terms, including publishing the source of your changes when you distribute homr or offer it as a network service. We chose the AGPL so that improvements make their way back to the community.

The code in `homr/` also comes with the [homr app permission](LICENSE_EXCEPTION.md). The training code in `training/` and all other code don't.

## Can I use homr in a closed source app?

Yes, under the [homr app permission](LICENSE_EXCEPTION.md). Your app may keep its own license if:

- the code from `homr/`, including your changes to it, stays under the AGPL and you publish its source,
- your app shows "Uses homr (https://github.com/liebharc/homr), licensed under AGPL-3.0" in its legal notices, for example on an About or credits screen.

If you change the code from `homr/` and run it as a network service, you must publish those changes. You don't have to publish the rest of your service.

The permission doesn't cover the training code. Changes to it stay under the AGPL only.

## Commercial license

We can't offer one. homr has many contributors, and nobody can relicense the code on their own.

## Model weights

The weights are licensed under [CC BY-SA 4.0](LICENSE_WEIGHTS.md). You may ship the unchanged weights in a closed source app, with credit. If you share fine-tuned weights, share them under CC BY-SA 4.0. However, the license can't grant more rights than the training data allows. Whether a dataset's license carries over to the weights trained on it is an open legal question.

| Model | Training data | License as stated by the source (checked October and November 2026) | Where it's stated |
|---|---|---|---|
| Segmentation (`segnet_*`) | [DeepScoresV2](https://zenodo.org/records/4012193) | CC BY 4.0 | Zenodo record |
| Transformer (`pytorch_model_*`, ONNX encoder/decoder) | [OpenScore Lieder](https://github.com/OpenScore/Lieder) | CC0 | LICENSE file |
| | [OpenScore StringQuartets](https://github.com/OpenScore/StringQuartets), merged into Lieder | CC0 | LICENSE file |
| | GrandStaff, downloaded from [grfia](https://sites.google.com/view/multiscore-project/datasets) | Not stated at the download source; MIT for the [copy on Hugging Face](https://huggingface.co/datasets/PRAIG/grandstaff) | Hugging Face metadata |
| | [PDMX](https://zenodo.org/records/15571083), `no_license_conflict` subset | CC BY 4.0 for the dataset; the subset contains scores whose MuseScore metadata consistently marks them as public domain | Zenodo record |
| | [homr-photo-pages](https://github.com/slobaspeed/homr-photo-pages), `photos/train/` (phone photos of paper; the printed music is erased and replaced by Lieder songs). Built with [UVDoc](https://github.com/tanguymagne/UVDoc) (MIT) | CC BY-SA 4.0 | README |
| Pretrained starting weights | ImageNet backbones: `convnext_tiny` from timm (transformer), `resnet18` from segmentation_models.pytorch (segmentation) | ImageNet's terms allow non-commercial research use; whether they carry over to weights trained from these backbones is open | ImageNet terms of access |

This reflects our reading of the sources and may be incomplete or out of date. The licenses of the underlying music can differ from those of the datasets. Check the sources yourself.

## Can homr be trained only on datasets that allow commercial use?

Possibly. None of the training datasets has non-commercial terms, but GrandStaff doesn't state a license at its download source (MIT on Hugging Face). We welcome PRs that replace training datasets with ones under permissive licenses, as long as accuracy doesn't regress. Such a PR needs to include a full training run and benchmark results, e.g. the system level validation and the SMB and polish scores. Weights released before such a change keep their current training data.

## Title detection

Title detection uses [RapidOCR](https://github.com/RapidAI/RapidOCR), which downloads its own models. Disable it with `--no-title` if you don't need it.
