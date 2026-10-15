# License FAQ

This is not legal advice. homr is a non-commercial volunteer project. If you use it commercially, the legal questions and risks are yours.

## Code

The code is licensed under [AGPL-3.0](LICENSE). The AGPL allows commercial use as long as you meet its terms, including publishing the source of your changes when you distribute homr or offer it as a network service. We chose the AGPL so that improvements make their way back to the community.

Is my application covered by the AGPL if it uses homr? You'll have to check that yourself. Running homr as a separate process or service (e.g. the Docker container in this repo) gives a clear separation.

## Commercial license

We can't offer one. homr has many contributors, and nobody can relicense the code on their own.

## Model weights

The weights are published under the AGPL as well. However, the AGPL can't grant more rights than the training data allows. Whether a dataset's license carries over to the weights trained on it is an open legal question.

| Model | Training data | License as stated by the source (checked October 2026) | Where it's stated |
|---|---|---|---|
| Segmentation (`segnet_*`) | [DeepScoresV2](https://zenodo.org/records/4012193) | CC BY 4.0 | Zenodo record |
| Transformer (`pytorch_model_*`, ONNX encoder/decoder) | [OpenScore Lieder](https://github.com/OpenScore/Lieder) | CC0 | LICENSE file |
| | [OpenScore StringQuartets](https://github.com/OpenScore/StringQuartets), merged into Lieder | CC0 | LICENSE file |
| | GrandStaff, downloaded from [grfia](https://sites.google.com/view/multiscore-project/datasets) | Not stated at the download source; MIT for the [copy on Hugging Face](https://huggingface.co/datasets/PRAIG/grandstaff) | Hugging Face metadata |
| | [Camera-PrIMuS](https://grfia.dlsi.ua.es/primus/) | Not stated at the download source; CC BY-NC 4.0 (non-commercial) for the [copy on Hugging Face](https://huggingface.co/datasets/PRAIG/PrIMuS) | Hugging Face metadata |
| | [PDMX](https://zenodo.org/records/15571083), `no_license_conflict` subset | CC BY 4.0 for the dataset; the subset contains scores whose MuseScore metadata consistently marks them as public domain | Zenodo record |
| | [MuseTrainer library](https://github.com/musetrainer/library) | No license file; the repo describes itself as public domain | Repo description |

Older checkpoints, e.g. 396, were trained on Lieder, StringQuartets, GrandStaff and PrIMuS only. All transformer checkpoints released so far include PrIMuS.

This reflects our reading of the sources and may be incomplete or out of date. The licenses of the underlying music can differ from those of the datasets. Check the sources yourself.

## Can homr be trained only on datasets that allow commercial use?

Possibly. We welcome PRs that replace training datasets with ones under permissive licenses, as long as accuracy doesn't regress. Such a PR needs to include a full training run and benchmark results, e.g. the system level validation and the SMB and polish scores. Weights released before such a change keep their current training data.

## Title detection

Title detection uses [RapidOCR](https://github.com/RapidAI/RapidOCR), which downloads its own models. Disable it with `--no-title` if you don't need it.
