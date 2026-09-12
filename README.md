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
| [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) | Video | 2,731 | 52 | ~83k |
| [Sem-Lex](https://github.com/leekezar/SemLex) | Video | 3,149 | 41 | ~91k |
| [Kaggle ASL Signs](https://www.kaggle.com/competitions/asl-signs/data) | MediaPipe Holistic landmarks | 250 | 21 | ~94k |
| [WLASL](https://dxli94.github.io/WLASL/) | Video | 2,000 | ~119 | ~21k |

ASL Citizen is the primary dataset: its large vocabulary matters most for generalizing to unseen signs, it is recorded with webcams with both hands free, and since it is video we extract the landmarks ourselves with the same setup the system will use in practice. Sem-Lex is the candidate second source (aligned with ASL Citizen through ASL-LEX, and annotated with phonological features). Kaggle ASL Signs comes from PopSign, which is one-handed smartphone signing (the other hand holds the phone), and its landmarks come from a MediaPipe version that is no longer available; it is kept only as an optional extra. ASL Citizen and WLASL are licensed for non-commercial use only.

The data pipeline is dataset-agnostic: each dataset has an adapter that converts it into a common format (a landmark array per clip, plus metadata: dataset, sign, signer). All video datasets are run through one fixed MediaPipe setup that outputs the common landmark layout. Sign labels have to be normalized when datasets are combined.

### Downloading ASL Citizen

No registration is needed. The zip is ~46 GB; download and extract into the git-ignored `data/` directory:

```sh
wget -c -P data/raw https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip
unzip -q data/raw/ASL_Citizen.zip -d data/raw/asl-citizen
```

### Downloading Kaggle ASL Signs

1. Accept the competition rules on the [competition page](https://www.kaggle.com/competitions/asl-signs/data).
2. Create an API token (Kaggle → Settings → API) and save it as `~/.kaggle/access_token`.
3. Download and extract into the git-ignored `data/` directory:

```sh
uvx kaggle competitions download -c asl-signs -p data/raw
unzip -q data/raw/asl-signs.zip -d data/raw/asl-signs
```

4. Convert to the common landmark format (a landmark store in `data/processed/kaggle_asl_signs/`, ~22 GB; the format is described in `src/isolated_sign_validation/landmarks.py`):

```sh
uv run python -m isolated_sign_validation.datasets.kaggle_asl_signs
```

## Evaluation

- **Held-out signers:** no signer appears in both training and test data.
- **Held-out signs:** test signs are never seen during training. Which signs are held out is rotated across folds.
- **k-shot verification episodes (k = 1, 3, 5):** k reference clips of a held-out sign come from some signers. Queries from other signers are either the same sign (positive) or another held-out sign (negative).
- **Metrics:** ROC-AUC and equal error rate. Closed-set accuracy on the training signs is reported as a sanity check. Threshold selection and sensitivity analysis come later.

## Future

- Validate Swedish Sign Language signs from [Svenskt teckenspråkslexikon](https://teckensprakslexikon.su.se/), which have little training data.
