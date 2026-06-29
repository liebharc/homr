# Surrogate robustness: before vs after PGD defense

## AutoAttack (white-box L-inf) on the staff-count head

|   epsilon |   before_count_accuracy |   after_count_accuracy |   before_token_accuracy |   after_token_accuracy |   before_mean_ser |   after_mean_ser |
|----------:|------------------------:|-----------------------:|------------------------:|-----------------------:|------------------:|-----------------:|
|    0.0000 |                  0.5417 |                 0.6250 |                  0.5702 |                 0.5824 |            0.4209 |           0.4015 |
|    0.0100 |                  0.1250 |                 0.5833 |                  0.5683 |                 0.5819 |            0.4225 |           0.4018 |
|    0.0200 |                  0.0000 |                 0.2917 |                  0.5622 |                 0.5812 |            0.4289 |           0.4035 |
|    0.0500 |                  0.0000 |                 0.0417 |                  0.5319 |                 0.5747 |            0.4542 |           0.4082 |
|    0.1000 |                  0.0000 |                 0.0000 |                  0.5267 |                 0.5504 |            0.4551 |           0.4262 |

Staff-count accuracy drop from epsilon 0.000 to 0.100: before defense 0.5417, after defense 0.6250.

## PGD epsilon-grid (token-level)

|   epsilon |   before_overall_accuracy |   after_overall_accuracy |   before_mean_ser |   after_mean_ser |
|----------:|--------------------------:|-------------------------:|------------------:|-----------------:|
|    0.0000 |                    0.5522 |                   0.5493 |            0.4277 |           0.4211 |
|    0.0100 |                    0.5281 |                   0.5467 |            0.4491 |           0.4234 |
|    0.0200 |                    0.4903 |                   0.5427 |            0.4899 |           0.4263 |
|    0.0500 |                    0.4109 |                   0.5325 |            0.5756 |           0.4341 |
|    0.1000 |                    0.4032 |                   0.5243 |            0.5823 |           0.4407 |
