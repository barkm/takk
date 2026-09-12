# isolated-sign-validation

A system that validates whether a sequence of human pose landmarks is signing a given sign.

**Input:**
- A sequence of human pose landmarks
- A sign

**Output:**
- Yes if the landmark sequence is signing the given sign, no otherwise

The vocabulary is not fixed: the system must also work for signs that were not seen during training and have only a few (possibly one) reference clips. The priority is telling a correct sign apart from a completely different sign; small execution errors are secondary.

## Approach

Train a sequence encoder that maps a landmark sequence to an embedding, so that clips of the same sign end up close together and clips of different signs end up far apart. A sign is validated by comparing the embedding of the attempt with embeddings of reference clips of the target sign, and thresholding the similarity.

Planned starting point: a 1D-conv + transformer encoder (as in the top Kaggle ASL Signs solutions), trained as a classifier with a margin-based loss (e.g. ArcFace), using its embedding layer for verification.

Baselines:
- DTW on normalized hand landmarks (no training)
- A small GRU encoder

## Data

| Dataset | Format | Signs | Signers | Clips |
|---|---|---|---|---|
| [Kaggle ASL Signs](https://www.kaggle.com/competitions/asl-signs/data) | MediaPipe Holistic landmarks | 250 | 21 | ~94k |
| [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) | Video | 2,731 | 52 | ~83k |
| [WLASL](https://dxli94.github.io/WLASL/) | Video | 2,000 | ~119 | ~21k |

Kaggle ASL Signs is the starting point since it already contains landmarks. ASL Citizen and WLASL add vocabulary, which matters most for generalizing to unseen signs. ASL Citizen and WLASL are licensed for non-commercial use only.

The data pipeline is dataset-agnostic: each dataset has an adapter that converts it into a common format (a landmark array per clip, plus metadata: dataset, sign, signer). Video datasets are first run through MediaPipe Holistic so their landmarks match the Kaggle layout. Sign labels have to be normalized when datasets are combined.

### Downloading Kaggle ASL Signs

1. Accept the competition rules on the [competition page](https://www.kaggle.com/competitions/asl-signs/data).
2. Create an API token (Kaggle → Settings → API) and save it as `~/.kaggle/access_token`.
3. Download and extract into the git-ignored `data/` directory:

```sh
uvx kaggle competitions download -c asl-signs -p data/raw
unzip -q data/raw/asl-signs.zip -d data/raw/asl-signs
```

## Evaluation

- **Held-out signers:** no signer appears in both training and test data.
- **Held-out signs:** test signs are never seen during training. Which signs are held out is rotated across folds.
- **k-shot verification episodes (k = 1, 3, 5):** k reference clips of a held-out sign come from some signers. Queries from other signers are either the same sign (positive) or another held-out sign (negative).
- **Metrics:** ROC-AUC and equal error rate. Closed-set accuracy on the training signs is reported as a sanity check. Threshold selection and sensitivity analysis come later.

## Future

- Validate Swedish Sign Language signs from [Svenskt teckenspråkslexikon](https://teckensprakslexikon.su.se/), which have little training data.
