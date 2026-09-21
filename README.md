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
| [Slovo](https://github.com/hukenovs/slovo) (Russian Sign Language) | Video | 1,000 | 194 | ~20k |
| [Svenskt teckenspråkslexikon](https://teckensprakslexikon.su.se/) (Swedish Sign Language) | Video | 21,692 (2,476 with ≥2 clips) | a few (not labeled) | 21,692 |

ASL Citizen is the primary dataset: its large vocabulary matters most for generalizing to unseen signs, it is recorded with webcams with both hands free, and since it is video we extract the landmarks ourselves with the same setup the system will use in practice. Sem-Lex is the candidate second source (aligned with ASL Citizen through ASL-LEX, and annotated with phonological features). Kaggle ASL Signs comes from PopSign, which is one-handed smartphone signing (the other hand holds the phone), and its landmarks come from a MediaPipe version that is no longer available; it is kept only as an optional extra. MM-WLAuslan adds training signs from another sign language (all new classes); it has no signer ids. Slovo is the opposite: it is never trained on, but has signer ids, so it serves as a held-out cross-language evaluation set that measures how well the model transfers to a sign language it has never seen (relevant for Swedish Sign Language later). Svenskt teckenspråkslexikon is the Swedish Sign Language dictionary and the goal vocabulary: one recording per entry, so it is an evaluation set only, with classes built from the entries that share a sign form. ASL Citizen and WLASL are licensed for non-commercial use only, MM-WLAuslan under CC BY-NC-SA 4.0 and Slovo under a variant of CC BY-SA 4.0.

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

### Downloading Slovo

No registration is needed. The trimmed videos and their annotations are a single ~16 GB zip:

```sh
wget -c -P data/raw https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/slovo/slovo.zip
unzip -q data/raw/slovo.zip -d data/raw/slovo
```

Then extract the landmarks of the 20,000 videos into `data/processed/slovo/` (5.5 hours; run it
in `tmux`, and run it again to resume). The 400 `no_event` videos hold no signing and are left out:

```sh
uv run python -m isolated_sign_validation.datasets.slovo
```

### Downloading WLASL

WLASL is distributed as links to YouTube and ASL dictionary sites, many of them dead (the dataset is
C-UDA: academic and computational use only). The surviving videos come from the
[wlasl-processed](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) mirror on Kaggle
(11,880 of the 21,083 instances, 4.8 GB, `missing.txt` lists the dead ones), which includes WLASL's
own metadata. It needs the Kaggle API token from the Kaggle ASL Signs section below:

```sh
uvx kaggle datasets download -d risangbaskoro/wlasl-processed -p data/raw
unzip -q data/raw/wlasl-processed.zip -d data/raw/wlasl
```

The mirror on Hugging Face ([Voxel51/WLASL](https://huggingface.co/datasets/Voxel51/WLASL)) has the
same videos as single files, but downloading 11,880 of them runs into rate limiting.

The mirror has none of WLASL's 5,135 YouTube instances, but most of their videos are still public.
`scripts/download_wlasl_youtube.py` downloads them with yt-dlp (no cookies needed as of 2026-09;
it uses `node` as yt-dlp's JavaScript runtime) and cuts each instance out at its frame range into
`data/raw/wlasl/youtube/`, where the adapter finds them next to the mirror's clips. Unavailable
videos are recorded in `youtube/unavailable.tsv` and skipped when run again, which resumes an
interrupted run:

```sh
uv run scripts/download_wlasl_youtube.py
```

Then extract the landmarks of the 11,980 videos into `data/processed/wlasl/` (5.1 GB, about 4.5
hours; run it in `tmux`, and run it again to resume). `--signs N` extracts a sample into a separate
store instead:

```sh
uv run python -m isolated_sign_validation.datasets.wlasl
uv run python -m isolated_sign_validation.datasets.wlasl --signs 20
```

### Downloading Svenskt teckenspråkslexikon

[Svenskt teckenspråkslexikon](https://teckensprakslexikon.su.se/) (CC BY-NC-SA 4.0) is the Swedish
Sign Language dictionary and the goal vocabulary, so it is an evaluation set only and never trained
on. It has no bulk download and no registration either: `scripts/download_sts_lexikon.py` crawls the
entry pages, the same-form groups they link to, and the sign videos, into `data/raw/sts-lexikon/`.
Each phase skips what an earlier run already has, so an interrupted crawl resumes when run again:

```sh
uv run scripts/download_sts_lexikon.py
```

The crawl of 2026-09-18 took about 50 minutes and found 21,692 entries with a sign video (16.2 GB).
The lexicon is updated continuously, so note the crawl date.

Then extract the landmarks of every entry into `data/processed/sts_lexikon/` (16 GB, about 5.5 hours
on the CPU; run it again to resume). `--signs N` extracts a sample into a separate store instead:

```sh
uv run python -m isolated_sign_validation.datasets.sts_lexikon
uv run python -m isolated_sign_validation.datasets.sts_lexikon --signs 20
```

The dictionary publishes one recording per entry and no signer ids. Entries the lexicon marks as
sharing a sign form ("Teckenformen kan också betyda") are one sign, a class of separate recordings of
that form under different Swedish meanings, usually by different model signers: 2,476 classes of
8,138 clips. Every other entry is a sign with a single clip. See ROADMAP.md for what this does and
doesn't buy.

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

Slovo is prepared as an evaluation set instead: every clip is a `test` clip, and its sign labels get the prefix `rsl:`. Classes whose label is a phrase rather than a single sign are left out, since the task is validating one sign, as are classes whose median clip takes longer than `--max_median_seconds` (3 s) to sign, which catches phrases the label doesn't reveal. It writes `data/prepared/slovo-<config id>/` (935 signs, 18,563 clips, 0.4 GB):

```sh
uv run scripts/prepare_slovo.py
```

Svenskt teckenspråkslexikon is prepared the same way, as an evaluation set whose labels get the
prefix `sts:`. Nothing is filtered out, but the lexicon's resting pose (hands clasped at the waist,
in frame) is hidden with `max_hand_y` 0.9 instead of the default 1.0. It writes
`data/prepared/sts_lexikon-<config id>/` (21,680 clips of 16,026 signs, 0.64 GB), the glossary for
recording Swedish signs (see Recording your own clips):

```sh
uv run scripts/prepare_sts_lexikon.py
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
uv run scripts/train.py --name gru_best
```

The defaults are the best configuration of the validation search (see ROADMAP.md): hidden size 384, 20 epochs, validation and checkpoints of an exponential moving average of the weights, and training on ASL Citizen, MM-WLAuslan and WLASL together. Linear heads on the pooled encoder output also predict each sign's ASL-LEX phonological features (handshape, location, movement, ...) as an auxiliary loss with weight 1 (`phonology_weight`; `phonology_input=embedding` puts the heads on the embedding instead); clips without features (e.g. MM-WLAuslan) only get the ArcFace loss. Settings are changed with `--set key=value`.

To train on other datasets, pass their prepared directories: `--prepared data/prepared/asl_citizen-<config id> ...`.

A run writes `outputs/runs/<name>/`: `config.json`, per-epoch `metrics.csv` and `curves.png` (updated during training), the checkpoint with the lowest validation EER at k = 1 (`best.pt`), and its full validation evaluation.

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

The same script evaluates the Slovo cross-language set, which measures k-shot verification of Russian
signs the model was never trained on, with references by other signers as everywhere else:

```sh
uv run scripts/evaluate_test.py gru_auslan_phonpool1 --prepared data/prepared/slovo-<config id> --name slovo
```

A query is scored against every sign of the evaluated set, so top-1 and top-5 drop as the set holds
more signs and cannot be compared between sets of different sizes. `--signs` evaluates random subsets
of a given number of signs instead (5 draws, metrics averaged), which matches Slovo's 935 signs to the
test split's 269:

```sh
uv run scripts/evaluate_test.py gru_auslan_phonpool1 --prepared data/prepared/slovo-<config id> --name slovo --signs 269
```

Threshold selection and sensitivity analysis come later.

### Exploring the embedding space

`scripts/explore_embeddings.py` serves a web page in the spirit of the [ASL-LEX visualization](https://asl-lex.org/visualization/),
built from a run's embeddings instead of hand-coded phonology: each sign is a point (the mean of its
clips' embeddings), laid out by t-SNE and grouped into clusters. Points can be colored by cluster,
split or ASL-LEX feature, and clicking one shows clips of the sign, its nearest signs and its cluster:

```sh
uv run scripts/explore_embeddings.py --run gru_auslan_phonpool1   # then open http://localhost:8001
uv run scripts/explore_embeddings.py --prepared data/prepared/asl_citizen-<config id> data/prepared/slovo-<config id>
```

## Recording your own clips

The deployment setting is a user copying a dictionary clip in front of their own camera, which no
public dataset covers. `scripts/collect.py` serves a small web app that collects exactly that: it
shows reference clips of a sign from a glossary, any prepared evaluation set, by a few different
signers, and records the signer's attempt with their webcam. The default glossary is the held-out
test signs of ASL Citizen; the prepared Svenskt teckenspråkslexikon gives Swedish signs, in a
language the model has never seen:

```sh
uv run scripts/collect.py --signs 25 --takes 3   # then open http://localhost:8000
uv run scripts/collect.py --glossary data/prepared/sts_lexikon-<config id> --signs 25 --takes 3
```

The browser only gives access to the camera on `localhost` or over https, so forward the port when
the machine is remote (VS Code's Remote-SSH does it in its Ports tab). The sign list follows from
`--glossary`, `--signs` and `--seed`, so several people can record the same signs, which is what lets
the k-shot protocol draw references from other signers. A glossary without signer ids, like the
lexicon, shows one reference clip per sign.

No score is ever shown. The recordings are only checked for whether they are *usable* — whether
preparation would keep the clip, and how steadily a hand was detected while signing — so that a
session cannot turn out to be unusable after the fact, and so that the signer cannot retake until
the model happens to agree, which would bias the set toward clips the model already likes. Every
take is stored, including the discarded ones, along with the signer, their handedness and whether
they felt sure of the sign. Each take is replayed with its extracted landmarks drawn over it, so a
lost hand or a cut-off shoulder is visible at once. Landmarks are extracted on the server by
`extraction.py`, the same setup every dataset went through, so the recordings are not a second,
subtly different extractor. The
session also collects a few `no_event` clips of not signing, as negatives that look like real usage.

Recordings of every glossary land in `data/raw/recordings/` (videos plus `clips.csv`, which also
lists the clips each take was shown) and are converted like any other dataset, using the kept takes
only:

```sh
uv run python -m isolated_sign_validation.datasets.recordings  # -> data/processed/recordings/
uv run scripts/prepare_recordings.py                          # -> data/prepared/recordings-<config id>/
```

They are then evaluated in the setting they were collected for: the glossary as the references, the
recordings of its signs as the queries. This needs no separate protocol, since `evaluation.evaluate`
takes a query mask and the recordings' signer is never a glossary signer, so the references of a
recording are dictionary clips by other people as everywhere else. Every result comes twice:
*copied*, against the whole glossary including the clips the recording was shown, which is the
product; and *uncopied*, without them, which asks whether the model knows the sign rather than the
one performance that was copied (a lexicon entry with a single clip has no uncopied trial):

```sh
uv run scripts/evaluate_recordings.py --run gru_auslan_phonpool1
uv run scripts/evaluate_recordings.py --name recordings_sts --glossary data/prepared/sts_lexikon-<config id>
```

It also prints each recording with the rank of its own sign among the whole glossary and the signs it
scored highest, which is what a handful of clips can actually say something about.

## Practicing signs

The app itself lives in its own package, `src/takk/` (see ROADMAP-takk.md); everything else in this
repository is the isolated sign verification work it is built on. It serves a small web app: search
the Swedish lexicon for a word, watch its clip, sign it to the webcam, and learn whether it was that
sign. The landmarks are extracted in the browser, live on the GPU while the camera runs (the CPU if
there is no GPU), with the same MediaPipe version and model as `extraction.py`, and drawn over the
camera image; only the landmarks of a recording are sent to the server, the video never leaves the
device. The server checks and prepares the attempt like a recording, embeds it with the run's model,
and accepts it when its mean cosine similarity to the sign's lexicon clips reaches `--threshold` (a
provisional 0.38, see ROADMAP.md). It also names the closest sign of the whole lexicon:

```sh
uv run takk   # then open http://localhost:8002
```

Picking several words makes a sentence, as TAKK signs the key words of a spoken sentence. Sign them
in order and say the sentence aloud while you sign, the way TAKK is used: the recording is split into
its signs by when the key words are spoken, and each part is scored against the sign at its place in
the sentence, with a verdict per sign. The words are known, so they are not recognized but timed, by
forced alignment against a Swedish CTC model (`takk/speech.py`, about 1.2 GB, downloaded on the first
run); a wildcard between them absorbs everything else that is said, so the sentence around the key
words can be any Swedish. Speaking is what locates the signs, so the microphone is always needed,
for a single sign as much as for a sentence: there the alignment has nothing to cut and its job is
to say the word was said at all. When the number of signs found differs from the sentence's, or a
word was not heard, the page says so and nothing is scored.

The page shows the rate the landmarks are tracked at. Below 20 fps it skips so many camera frames
that the answer is less reliable, since the model was trained on every frame. The lexicon's
embeddings are cached in `outputs/runs/<run>/`, so only the first start takes about a minute. As with
the collection app, forward the port when the machine is remote.
`scripts/compare_browser_extraction.py` compares the browser's landmarks with the Python extraction
on the recordings.

"Dagens pass" (`/pass`) is the practice session. The page opens on the choices — the kind of pass,
how many signs it holds, how many accepted attempts finish a sign, and which packs it draws new words
from — and the pass begins when the learner starts it. A daily pass is the signs that are due, filled
up with words never practised; a review pass ("Repetera allt du kan") draws from every box instead,
weighted by `1 / DAYS[box - 1]`, so it leans on the words that sit worst and teaches nothing new. A
word never practised is taught, with its clip; a repetition is a flash card, so the word alone is
shown and the clip follows the verdict. The sign can be looked up first, which does not move it up a
box. The summary offers another pass and the settings again.

Each sign has a Leitner box and a next-due date in the browser's `localStorage`
(`frontend/src/lib/progress.ts`), the four boxes falling due after 1, 3, 7 and 21 days. A sign moves
up a box once a pass has accepted it as many times as the learner asked for, two by default, and a
miss drops it to the first box at once and keeps it there for that pass. A sign is only promoted if
it was actually due: a box claims its sign is still remembered after that many days, and an answer
given on the second day of a seven-day box has not tested that, so an early answer moves nothing but
the time the sign was last seen. A miss counts whatever the day, since forgetting a sign that was not
due means its interval was already too long. A sign put back in the first box comes back in the same
pass, so a pass drills what was missed.

The packs say where new words come from and nothing else: repetitions come from every box, so
unticking a pack stops it teaching new words without stranding the words it already taught. A day
introduces at most a set number of new words, five until the learner changes it; repetitions are
never held back. Without the cap a session fills every spare slot from 9,887 unpractised words, so a
keen day leaves a month of repetitions behind it. Each box keeps the time its sign was first met,
which is the only way to tell a new word from an old one that was missed: both sit in the first box.
Nothing is stored on the server, so clearing the browser's storage starts the learner over.

`/pass?days=1` shows the session as it will look that many days from now, which is how the spaced
repetition is tried without waiting for it: only what counts as due moves, and an attempt is still
recorded at the real time.

A box belongs to a sign class, and a class has many words: `sts:spader-00016` is "svart" in Färger
and "Oden" in Mytologi. The word on the card is therefore stored with the box, so progress is shown
under the word that was practised rather than whichever pack happened to name the sign last, and the
packs listed beside it are the ones that teach that word, not every pack the sign class appears in.

"Sagan" (`/saga`) is a prototype of story mode: the server writes one Swedish story over the words
the learner already knows (`takk/story.py`, `POST /api/story`), in 3, 6 or 10 parts, and the learner
reads a part aloud and signs the words marked in it. The words turn green or red and the story goes
on whatever happened, with a summary at the end. Words are drawn towards the low boxes, so a story
leans on what sits worst, and nothing new is taught — new words belong to the daily pass. A word the
learner did not say is a miss of its sign rather than a refused recording (`unheard_is_miss`), which
is what lets the story keep going. A part is recorded for as long as its text takes to read aloud
rather than for as long as its signs, since most of a part is spoken and only a few of its words are
signed.

"Mina tecken" (`/framsteg`) lists what the boxes hold: how many signs are due now, how many sit in
each box, and every practised sign with its box, when it is due and which packs it is in, with a
reset behind a fold. It is also how the spaced repetition is inspected while it is being built.

A card also shows how the lexicon describes the sign, in Swedish ("Flata handen, framåtriktad och
uppåtvänd, förs åt vänster ..."), which is the only teaching text the lexicon publishes and is on all
but 30 of the entries with a video. It comes from `GET /api/form/{entry}` per card, since the whole
glossary's descriptions are about 3 MB, and the entry is the word's own rather than the sign class's.

A card says only whether the sign was right. The score, the closest sign in the lexicon and the
threshold are free practice's business, where a sign is looked into; in a pass the answer is right or
wrong and the next card follows by itself. Nothing is pressed between cards: an accepted sign gives
way to the next one at once, and a missed one is armed for another recording with its verdict and its
clip still up, so a pass is signed and spoken from beginning to end without touching anything.

The learner chooses what the session draws from, by turning packs on and off. New words are taken
from the chosen packs in turn, a word at a time, so a large pack cannot hide a small one: in plain
order Djur's 196 words would all come before Mat och dryck's first. A pack is a starter
pack or one of the lexicon's categories, which to the learner are the same kind of thing: a named
list of words. `GET /api/packs` serves all of them (`takk/vocabulary.py`), the starter packs first,
and the choice is kept in `localStorage` as well.

### The frontend

The page is a SvelteKit app in `frontend/` (TypeScript, Vite, `adapter-static`). It is a separate
deployment from the API and is never served by it (see the decisions in ROADMAP-takk.md): Vite proxies
`/api` to the server in development and in `preview`, and a deployed build is pointed at the API with
`VITE_API_BASE`. `src/lib/landmarks.ts` and `tracking.ts` hold the camera, the landmarker loop and the
common landmark layout; the components only show them.

```sh
uv run takk                      # the API, in one terminal
cd frontend && npm install       # once
npm run dev                      # then open http://localhost:5173 (forward this port, not 8002)
npm run build && npm run preview  # the built site, http://localhost:4173
npm run check                    # svelte-check
npm test                         # the landmark layout, the resampling and the Leitner boxes
```

### The word sets

The lessons are built from the lexicon's own subject categories (Djur, Kläder, Mat och dryck, ...,
listed at https://teckensprakslexikon.su.se/kategori). Every entry page names the ones it belongs to,
as a path ("Sport > klubbar och föreningar > NHL"), and the crawl stores them in the `categories`
field of each entry, together with the entry's other wording (`also`: the sign for "arbetsvetenskap"
is also "ergonomi"), its English translation and its hit counts in the lexicon, the corpus and the
surveys. `takk/vocabulary.py` turns a category into words to practise. Nothing is
generated: re-running the crawler picks up whatever the lexicon has changed.

A category's word is the heading of an entry in that category, not the name of the sign that scores
it: signs of one form are one class labelled by its lowest entry, and that entry often belongs to
another subject, so "Djur" would otherwise start at "hane" and "batteri". The words come in the order
the lexicon counts them (`lexicon_hits`, then `corpus_hits`), so a category starts where a learner
starts: Sport with träna, fotboll, ishockey rather than 2,238 signs in entry order. Only 27% of
entries are counted at all, so the tail of a large category keeps the lexicon's own order.

A category is a subject area of a dictionary and not a learning order (Sport is the largest), so the
first lessons come from a starter pack instead: a short list of everyday Swedish words in
`takk/packs.json`, resolved to lexicon entries at startup and served by `GET /api/packs`. The word a
learner reads is kept apart from the sign that scores it, because entries that share a sign form are
one class labelled by the lowest of them ("äta" is scored as `sts:livsmedel-01265`).

The category listing pages are not used: they miss categories an entry page names (15 of 84 in a
sample of 120 entries) and hide the deeper levels of the path.

```sh
uv run scripts/download_sts_lexikon.py   # the entry pages carry the categories
```

## Future

- Validate Swedish Sign Language signs from [Svenskt teckenspråkslexikon](https://teckensprakslexikon.su.se/), which have little training data.
