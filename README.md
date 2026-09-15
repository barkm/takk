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
| [MM-WLAuslan](https://uq-cvlab.github.io/MM-WLAuslan-Dataset/) (Australian Sign Language) | Video (4 cameras) | 3,215 | 73 | ~283k |

ASL Citizen is the primary dataset: its large vocabulary matters most for generalizing to unseen signs, it is recorded with webcams with both hands free, and since it is video we extract the landmarks ourselves with the same setup the system will use in practice. Sem-Lex is the candidate second source (aligned with ASL Citizen through ASL-LEX, and annotated with phonological features). Kaggle ASL Signs comes from PopSign, which is one-handed smartphone signing (the other hand holds the phone), and its landmarks come from a MediaPipe version that is no longer available; it is kept only as an optional extra. MM-WLAuslan adds training signs from another sign language (all new classes); it has no signer ids. ASL Citizen and WLASL are licensed for non-commercial use only, MM-WLAuslan under CC BY-NC-SA 4.0.

The data pipeline is dataset-agnostic: each dataset has an adapter that converts it into a common format (a landmark array per clip, plus metadata: dataset, sign, signer). All video datasets are run through one fixed MediaPipe setup that outputs the common landmark layout. Sign labels have to be normalized when datasets are combined.

### Downloading ASL Citizen

No registration is needed. The zip is ~46 GB; download and extract into the git-ignored `data/` directory:

```sh
wget -c -P data/raw https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip
unzip -q data/raw/ASL_Citizen.zip -d data/raw/asl-citizen
```

Then extract the landmarks into a landmark store in `data/processed/asl_citizen/` (~44 GB). This runs MediaPipe's HolisticLandmarker on the CPU and takes about half a day, so run it in `tmux` to survive a dropped SSH connection. If interrupted, run the same command again to resume:

```sh
tmux new -s extract
uv run python -m isolated_sign_validation.datasets.asl_citizen
```

For a quick look at the data, extract only all videos of a few randomly chosen signs into a separate store (`data/processed/asl_citizen_10_signs/`, a few minutes) and inspect them with the clip viewer:

```sh
uv run python -m isolated_sign_validation.datasets.asl_citizen --signs 10
uv run scripts/view_clips.py --store data/processed/asl_citizen_10_signs --sign <SIGN>
```

### Downloading MM-WLAuslan

The dataset is in a public Google Drive folder, downloaded with [rclone](https://rclone.org/) and a Google Drive remote (here named `personal gdrive`). Only the RGB videos of the front Kinect camera are used, from the subsets Train, Valid, Test-STU, Test-ITW and Test-SYN (27.6 GB of zips; the label files and the dictionary of English keywords are small):

```sh
rclone copy "personal gdrive:" data/raw/mm-wlauslan --drive-root-folder-id 1EQ1Nh3lidEcu1QLFw0IjRN7YqEq1N48q \
    --include "Annotation/Labels & Split/**" --include "WWW_CV_ISLR_Challenge/Dictionary_Mapping/**" \
    --include "{Train,Valid,Test-STU,Test-ITW,Test-SYN}/Kinect_F/rgb.zip" -P
for subset in Train Valid Test-STU Test-ITW Test-SYN; do
    unzip -q data/raw/mm-wlauslan/$subset/Kinect_F/rgb.zip -d data/raw/mm-wlauslan/$subset/Kinect_F
done
```

Then extract the landmarks of the 64,300 videos into `data/processed/mm_wlauslan/` (~36 GB, about 11 hours; run it in `tmux`, and run it again to resume). `--subsets` and `--signs N` extract a sample into a separate store instead:

```sh
uv run python -m isolated_sign_validation.datasets.mm_wlauslan
uv run python -m isolated_sign_validation.datasets.mm_wlauslan --subsets Valid --signs 30
```

### Downloading ASL-LEX

[ASL-LEX 2.0](https://osf.io/zpha4/) (CC BY 4.0) describes the phonology of the ASL signs that ASL Citizen's glosses are coded against (handshape, location, movement, ...). Its sign data (2 MB) is public:

```sh
mkdir -p data/raw/asl-lex
curl -L -o data/raw/asl-lex/signdata.csv https://osf.io/download/9nygd/
curl -L -o data/raw/asl-lex/signdataKEY.csv https://osf.io/download/ygq4v/  # column descriptions
```

### Preparing training data

Training uses prepared clips (`src/isolated_sign_validation/preparation.py`): hands resting low below the shoulders treated as undetected (out of view, as in close webcam framings), broken clips excluded, trimmed to the frames with hands, corrected for the video's aspect ratio, short hand gaps interpolated, normalized by the shoulders, mirrored so the dominant hand is always in the `right_hand` slot, reduced to the hands, upper body and face reference points, and resampled to 30 fps (at most 128 frames). ASL Citizen clips also get their sign's ASL-LEX phonological features. This takes a few seconds and writes `data/prepared/asl_citizen-<config id>/` (~1 GB):

```sh
uv run scripts/prepare_asl_citizen.py
uv run scripts/view_clips.py --prepared data/prepared/asl_citizen-<config id> --sign APPLE --augment
```

The viewer shows the prepared clips below the original ones, and with `--augment` a random training augmentation below them. `src/isolated_sign_validation/dataset.py` serves the prepared clips of a split as a PyTorch dataset.

MM-WLAuslan is prepared as extra training data: all its clips are training clips, except the signs whose gloss or English keyword matches an ASL Citizen val or test sign, which could look like held-out signs. Its sign labels get the prefix `auslan:`. It writes `data/prepared/mm_wlauslan-<config id>/`:

```sh
uv run scripts/prepare_mm_wlauslan.py
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

## Training and baselines

Baselines without training (a hand-crafted embedding and dynamic time warping) are evaluated with:

```sh
uv run scripts/evaluate_baselines.py  # results in outputs/results/
```

The learned baseline, a bidirectional GRU embedding model trained with an ArcFace loss over the training signs (`src/isolated_sign_validation/models.py`, `training.py`), is trained with:

```sh
uv run scripts/train.py --name gru_arcface
```

To train on several datasets, pass their prepared directories: `--prepared data/prepared/asl_citizen-<config id> data/prepared/mm_wlauslan-<config id>`.

A run writes `outputs/runs/<name>/`: `config.json`, per-epoch `metrics.csv` and `curves.png` (updated during training), the checkpoint with the best validation AUC (`best.pt`), and its full validation evaluation.

## Evaluation

Splits (`src/isolated_sign_validation/splits.py`) hold out both signs and signers:

- **Held-out signs:** signs are assigned to train, val or test (~80/10/10) by a hash of the sign label, so a held-out sign stays held out when datasets are added.
- **Held-out signers:** test clips are test signs performed by the official ASL Citizen test signers, who never appear in training or validation.
- **Validation:** val signs performed by the training signers. It measures generalization to unseen signs but not to unseen signers, so validation scores are somewhat optimistic; only test measures both.
- **One fixed split** during development, with bootstrap confidence intervals. Retraining on several sign folds is reserved for final numbers or close comparisons.

For ASL Citizen this gives 40,126 train clips (2,172 signs, 40 signers), 5,342 val clips (290 signs, 13–20 signers per sign) and 3,239 test clips (269 signs, 11 signers).

Evaluation (`src/isolated_sign_validation/evaluation.py`) works on a similarity matrix between the clips of a split, e.g. cosine similarities of embeddings or negative DTW distances:

- **k-shot references (k = 1, 3, 5):** for each query clip and each sign of the split, k reference clips of the sign by k different signers, never the query's signer. The query's score for a sign is its mean similarity to the references.
- **Verification:** the query's own sign is a positive trial, every other sign a negative trial (so confusable signs are included). ROC-AUC and equal error rate over all trials pooled, i.e. for one global threshold.
- **Identification:** top-1 and top-5 accuracy of the query's own sign among all signs of the split.
- **Uncertainty and breakdown:** pooled over 5 random draws of references, with 95% bootstrap confidence intervals over signs, and per-sign metrics to find hard signs.

Model decisions are made on validation; the test split is for final numbers and important comparisons. To evaluate runs on it (the gains are relative to the first run, paired over signs):

```sh
uv run scripts/evaluate_test.py gru_hide_low gru_auslan  # writes outputs/runs/<run>/test_*.parquet
```

Threshold selection and sensitivity analysis come later.

## Future

- Validate Swedish Sign Language signs from [Svenskt teckenspråkslexikon](https://teckensprakslexikon.su.se/), which have little training data.
