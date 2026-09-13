# Roadmap

Status of the work and the plan ahead. Update this file when a step is finished, a decision is made, or the plan changes. See README.md for the task, approach and evaluation protocol.

## Steps

1. **Data exploration and pipeline (Kaggle ASL Signs)** — done
   - Kaggle ASL Signs downloaded; exploration in `scripts/eda_asl_signs.py` (figures in `outputs/eda/`). The raw Kaggle files have since been deleted to free disk space; only the landmark store in `data/processed/kaggle_asl_signs/` is kept, so re-running the exploration script requires downloading again.
   - Common, dataset-agnostic landmark store (`landmarks.py`) and Kaggle adapter (`datasets/kaggle_asl_signs.py`). All 543 landmarks are stored; subsets are selected at load time.
   - Landmark groups relevant for signing (`LANDMARK_GROUPS`, ~100 landmarks: hands, upper body, face reference points, lips).
   - Clip viewer rendering sequences as animated GIFs (`scripts/view_clips.py`).

2. **ASL Citizen data** — done
   - Download (~46 GB zip) and inspect the contents: videos, metadata, official splits. — done (see findings)
   - Choose the landmark extractor: run MediaPipe (Tasks API) on a handful of videos, check the result in the viewer, and measure extraction speed. — done (see decisions and findings)
   - Adapter extracting landmarks from the videos into the common layout (`datasets/asl_citizen.py`, resumable). — done
   - Full extraction (~12–14 h, see README). — done: `data/processed/asl_citizen/` (42 GB), 83,399 clips, 6,901,733 frames, 2,731 signs, 52 signers; clip order matches the split CSVs, no empty clips, fps 11.3–120.
   - Exploration of the extracted data (sequence lengths, hand presence, handedness). — done: `scripts/explore_store.py` (works on any store; figures in `outputs/eda/asl_citizen/`), see findings.

3. **Splits** — done: `splits.py` (see decisions). ASL Citizen: 40,126 train clips (2,172 signs, 40 signers), 5,342 val clips (290 signs, 39 signers, 13–20 per sign), 3,239 test clips (269 signs, 11 signers, 8–11 per sign); 34,692 clips belong to no split.

4. **Training data loader** — done (see the loader design decision)
   - Frame size stored per clip (`width`, `height`), backfilled into the existing stores.
   - `preparation.py`: exclusion, trimming, aspect correction, gap filling, normalization, mirroring, landmark selection, resampling; `scripts/prepare_asl_citizen.py` writes `data/prepared/asl_citizen-b7bd1b06/` (48,559 of 48,707 split clips; 0.98 GB; 12 s).
   - `dataset.py`: PyTorch `SignDataset` per split with augmentation, and `collate` with padding and a frame mask (~25,000 clips/s with 8 workers).
   - Clip viewer: `--prepared` shows prepared clips below their originals, `--augment` adds augmented versions.

5. **Evaluation harness** — done: `evaluation.py` (see README). Evaluates a similarity matrix: per query and sign, k references by different signers other than the query's; pooled ROC-AUC and EER over the positive and all negative trials; top-1/top-5 identification; per-sign metrics; 5 reference draws; 95% bootstrap CIs over signs. A full evaluation of the val split takes 20–35 s; for checks during training use one k, one draw and no bootstrap.

6. **Baselines** — done
   - Baselines without training (`baselines.py`, `scripts/evaluate_baselines.py`, results in `outputs/results/`): hand-crafted embedding and DTW. — done (see findings)
   - Small GRU embedding trained with ArcFace (`models.py`, `training.py`, `scripts/train.py`, run `outputs/runs/gru_arcface/`). — done (see findings)

7. **Strong model** — in progress
   - Experiment tooling: `scripts/train.py --set key=value` overrides, `scripts/compare_runs.py`; checkpoints selected on validation EER at k = 1. — done
   - Regularization (GRU, 30 epochs): stronger augmentation + dropout 0.4 + weight decay 0.05 are the new defaults (see findings). — done
   - Conv + transformer encoder (`ConvTransformerEncoder`): width and depth comparison. — done: worse than the GRU (see findings)
   - Conv + transformer with longer training (80 epochs) and less dropout (0.2). — done: with dropout 0.2 it matches the GRU; longer training doesn't help (see findings)
   - Generalization to unseen signers (`holdout_signers`, `scripts/signer_gap.py`). — done: small gap (see findings)
   - Vocabulary scaling (`train_sign_fraction`: 25/50/75/100% of the training signs, same number of training steps). — done: steady gains with more signs (see findings; plot in `outputs/results/vocab_scaling.png`)
   - Conv + transformer encoder with heavy augmentation.
   - Experiments: landmark groups (hands only / + body / + face), loss (ArcFace vs supervised contrastive), sequence length.

8. **More data**
   - MM-WLAuslan (3,215 Auslan signs, see findings): downloadable; inspected labels, splits and sample videos. — in progress
   - Sem-Lex (blocked: no access to the Drive files yet): adapter with the same extractor; sign labels normalized with ASL Citizen via ASL-LEX. Its phonological feature annotations (handshape, location, movement) could serve as auxiliary training targets.
   - Optionally Kaggle ASL Signs, keeping in mind it is one-handed signing from a different extractor.

Later: sensitivity analysis and threshold selection, including score normalization (see the baseline findings); Swedish Sign Language signs from teckensprakslexikon.su.se.

## Decisions

- **ASL Citizen is the primary dataset** instead of Kaggle ASL Signs: ~11x the vocabulary (the main lever for unseen signs), 52 signers, both hands free, and videos so the landmark extractor is under our control and matches what the system will use in practice. Kaggle ASL Signs is one-handed smartphone signing with landmarks from a removed MediaPipe version.

- **Landmark extractor:** MediaPipe Tasks `HolisticLandmarker` (successor of the legacy Holistic) in video mode with default settings, on the CPU with 8 worker processes (`extraction.py`). Its landmarks match the Kaggle layout and hand label convention. MediaPipe's GPU mode was rejected: the holistic model fails on the GPU, and the separate face/hand/pose models run on the GPU but scale worse than the CPU (see findings).
- **Frame rate:** each clip's source fps is stored in the store metadata (null when unknown), so sequences can be resampled to a common rate.
- **Resumable extraction:** long extractions write the store in chunks of 1,024 clips and resume from the last finished chunk (`write_store_resumable`).
- **Raw ASL Citizen data is kept** (zip and videos), e.g. for re-extraction or rendering video frames.
- **Splits** (`splits.py`): signs are assigned to train/val/test (~80/10/10) by a hash of the sign label, stable across datasets. Test = test signs by the official ASL Citizen test signers (held out completely). Train and val share the official train and val signers; val = val signs by those signers, so it measures unseen signs but not unseen signers. Chosen over the official signer split (train/val/test signers), which left only 31,909 training clips (57% of all clips unused) and 2–5 signers per val sign, too few for 5-shot validation.
- **One fixed split during development**, with bootstrap confidence intervals over test signs; retraining on several sign folds (each fold is a full training run) only for final numbers or comparisons too close to call.

- **Training data loader design** (step 4): PyTorch. Two stages: a deterministic preparation, cached in `data/prepared/`, and random augmentation at training time.
  - Landmarks: hands + upper body + face reference points (60) by default, lips as a later experiment; x and y only by default (z is noisy).
  - Image aspect ratio: store each clip's frame width and height, and convert coordinates to correct proportions before normalizing (ASL Citizen mixes 4:3 and 16:9).
  - Normalization per clip: center on the mean shoulder midpoint, scale by the mean shoulder width.
  - Mirroring so the dominant hand is always on the same side (the `right_hand` slot): a one-handed clip (both hands in < 20% of its frames with a hand) by its detected hand; a two-handed clip by the signer's handedness, the majority over their one-handed clips. At inference the user's handedness is therefore needed (e.g. an app setting, or estimated from their one-handed signs). Training without mirroring but with random mirroring augmentation is a later experiment.
  - Time: trim to the first to last frame with a hand plus ~0.1 s, resample to 30 fps, cap at 128 frames by resampling; variable length with padding and a mask.
  - Hand gaps: interpolate gaps of up to ~5 frames; longer gaps are zeros with a per-frame hand-present flag.
  - Exclude broken clips (no hands, hands for under ~0.2 s or over ~10 s), after checking examples.
  - Augmentation: rotation, scale, shift, shear, speed change, frame dropping, random hand dropout.
  - The loader outputs normalized coordinates and hand-present flags; derived features (velocities, hand-local coordinates) are computed in the model.

- **Next direction after the architecture experiments: more data, starting with Sem-Lex** (step 8), since architectures and regularization converge to the same level. Measuring generalization to unseen signers (with training signers held out, not the test split) can run alongside.

## Open decisions

- MM-WLAuslan has no signer ids: how to use it (training-only data vs deriving signer ids), and which camera views and subsets to extract.

## Findings

Kaggle ASL Signs:
- 94,477 clips of 250 signs by 21 signers. Balanced: 299–415 clips per sign, 3.3k–5k clips per signer.
- Hand detections are almost entirely `left_hand` or entirely `right_hand` per signer: 9 left-dominant, 12 right-dominant signers. Explained by the data's origin: PopSign ASL is one-handed smartphone signing, with the other hand presumably holding the phone.
- The hand labels appear to refer to the signer's own hands in an unmirrored image (`left_hand` shows up on the image's right side). Seen in a few clips only.
- 39% of frames have no hand detected; every clip has a hand in at least one frame.
- Sequence length: median 22 frames, 95th percentile 135, max 537. 43 clips have gaps in their frame numbers (closed by the adapter). The capture frame rate is unknown.
- Coordinates go far outside [0, 1] (x from −1.3 to 2.9, y up to 3.6), mostly from out-of-frame lower-body pose landmarks.
- Some clips are very short and seem to capture only part of the sign.

ASL Citizen:
- Single 45.9 GB zip from the Microsoft Download Center, no registration. Commercial use requires contacting ASL_Citizen@microsoft.com.
- License (`use.txt`, Microsoft Research License Terms): non-commercial research only; the data and modifications of it (e.g. extracted landmarks) may not be distributed; personal data must be destroyed when the research is completed.
- Layout: `ASL_Citizen/videos/*.mp4` (83,399 videos, ~50 GB) and `ASL_Citizen/splits/{train,val,test}.csv` with columns `Participant ID`, `Video file`, `Gloss`, `ASL-LEX Code`.
- Official splits are signer-disjoint: 35 / 6 / 11 signers with 40,154 / 10,304 / 32,941 videos. All 2,731 glosses appear in every split; 21–45 videos per gloss (mean 31). Videos per signer are very uneven (2 to 3,004, median 1,496).
- Glosses are cleaned labels (e.g. file `NOT MIND` → gloss `NOTMIND`, `SAIL` → `SAIL1`). 2,723 ASL-LEX codes; only RESEARCH1/2 and WHATFOR1/2/3 share a real code (all in the train split, so no leakage), and a few glosses have code `NA`.
- Videos (sample of 300): mostly H.264 640x480, some 960x540 and MPEG-4. Frame rate varies (mostly ~30 fps, also 25 and 15 fps). Mean length 2.7 s / 80 frames (33–366), so ~6.7M frames in total and a ~44 GB landmark store at float32.

- First extraction test (32 videos): face and pose detected in 100% of frames; `left_hand` in 33%, `right_hand` in 42%, both in 30%, neither in 55% (recordings start and end with the hands down). Two-handed signing is present, unlike Kaggle.
- Hand dropouts (30-sign subset, 932 clips, 76,918 frames): 57% of frames have no hand, of which 53 points are before the first / after the last detected hand (hands down) and 4 points are 948 short gaps during signing. During signing, with the pose wrist clearly inside the image (y < 0.85), the hand is detected in ~91% of frames.
- Extractor comparison on 40 videos (hand detected when the pose wrist is clearly visible, all frames): holistic 83.9%; holistic with hand confidence 0.2 identical (the option has no effect); separate hand + pose landmarkers 83.0%; the same with thresholds 0.3: 84.6% but more hands far from their wrist; image mode (no tracking) 77.6%. No setup meaningfully reduces dropouts; separate hand + pose without face mesh is ~20% cheaper on the CPU.
- Full store (`scripts/explore_store.py`): fps mostly 30 (70,744 clips), then 31 (5,804), 25 (3,173), 15 (1,698). Clip duration median 2.6 s (99%: 6.9 s, max 22.6 s); first to last frame with a hand median 1.2 s (95%: 2.5 s, 99%: 4.0 s, max 19.9 s).
- 57.2% of frames have no hand: 53.1 points before the first / after the last hand, 4.1 points gaps during signing. 109 clips have no hand at all; ~250 clips have hands for under 0.2 s and ~50 for over 10 s (likely broken or multi-attempt recordings).
- Two-handedness is bimodal: ~28% of clips have (almost) no frame with both hands; 54.8% have both hands in at least half of their frames with a hand.
- Dominant hand: unlike Kaggle, no signer is one-sided. For most signers `left_hand` is the more-detected hand in 15–45% of their clips (right-handed); ~10 signers are at 55–80% (left-handed, or mirrored webcam video, which the landmarks cannot tell apart). Counting detected frames per clip is ambiguous for two-handed signs.
- Handedness: in one-handed clips (26,278 with both hands in < 10% of their hand frames) the detected hand's label is on the expected image side in 97% of clips (pose right shoulder on the image's left in 99.95% of frames), and per signer it is clear-cut: 28 signers are ≤ 10% `left_hand`, 7 are ≥ 90%, 12 are in between (possibly mirroring that changed between recording sessions, or switching hands). For the 35 clear-cut signers, the detected hand matches the signer's handedness in 97.2% of one-handed clips, but no per-clip rule works for two-handed clips: which hand moves more (hand wrist path, frames with both hands) matches in only 51.6% (63% when one hand moves 3x more). Pose wrist motion is useless for this: the pose model's estimates for resting, out-of-frame wrists jitter as much as the signing hand moves.
- Suspected broken clips, inspected as video frames: 109 clips without any hand (22 signers, 40 from P18) and 152 with hands for under 0.2 s are mostly extraction failures on real signing (hands in front of the face, raised above the head, blurred or partly out of frame), plus a few truncated recordings; 50 clips with over 10 s of hand activity (47 from P46) are long recordings with idle time, so hand presence cannot locate the sign. All 311 (0.37%) are excluded with the planned thresholds. Hands in front of the face are a weak spot of hand detection.
- Prepared data (default config): train 40,015 clips (2,172 signs), val 5,326 (290), test 3,218 (269); 148 excluded. Prepared length median 41–42 frames, 95th percentile 74–80, max 128. Mirrored: 17% of train and val clips but 44% of test clips, so several of the 11 test signers are left-dominant or recorded mirrored.
- Evaluation harness sanity check on val (290 signs): random embeddings give AUC 0.50, EER 0.50, top-1 0.003 (= 1/290), top-5 0.017 (= 5/290). A trivial hand-crafted embedding (dominant hand's mean shape relative to the wrist, mean wrist position and wrist path at 8 time points) gives AUC 0.83 / 0.88 / 0.89, EER 0.25 / 0.20 / 0.19, top-1 16% / 19% / 20% for k = 1 / 3 / 5; 95% CI for AUC at k = 1: 0.815–0.839. Hardest signs for it: BARK2, END, MISSING, AXE1, BOW2. Trained models have to beat this.
- Baselines without training on val (k = 1 / 3 / 5, 95% CI about ±0.01 for AUC):
  - Hand features (cosine of the hand-crafted embedding): AUC 0.827 / 0.875 / 0.887, EER 0.249 / 0.204 / 0.191, top-1 16.3% / 19.0% / 20.0%.
  - DTW (both hands + upper body, 32 frames, missing hands at the pose wrist): AUC 0.817 / 0.850 / 0.859, EER 0.259 / 0.231 / 0.223, top-1 17.8% / 20.7% / 22.4%. Better at identification but worse at verification than the hand features.
  - The difference is calibration: raw DTW distances vary in scale between queries, which hurts a single global threshold. Z-normalizing each query's similarities (against all other clips of the split, i.e. a cohort of other signs) lifts DTW to AUC 0.832 / 0.877 / 0.890 and EER 0.185 at k = 5, on par with the hand features (0.894, 0.186 normalized), without changing identification. Relevant for threshold selection later; a cohort of other signs' references would be needed at inference.
- GRU + ArcFace baseline (bidirectional 2-layer GRU, hidden 256, 256-d embedding, ArcFace scale 30 / margin 0.3, AdamW 1e-3, 60 epochs of ~9 s on the RTX 3090): val AUC 0.979 / 0.991 / 0.992, EER 0.061 / 0.036 / 0.033, top-1 75.3% / 83.7% / 86.1% for k = 1 / 3 / 5 (95% CI for top-1 at k = 5: 84.0–87.8%), far above the baselines without training (AUC 0.887, top-1 22% at k = 5). Already after 2 epochs: AUC 0.969, top-1 48% (k = 5).
- The GRU overfits: validation peaks around epochs 15–20 (best checkpoint epoch 15) while training accuracy keeps rising to 99.7% at epoch 60 and validation slowly declines. With ~18 training clips per sign, more regularization, augmentation or data should help.
- Validation AUC is close to saturated (0.99 with CIs of ±0.002), so EER and top-1, especially at k = 1, are the more sensitive metrics for comparing models. Validation signers are also training signers, so the test split will show how much of this carries over to unseen signers.
- Regularization experiments (GRU, 30 epochs, val EER / top-1 at k = 1): defaults 0.0593 / 75.7%; dropout 0.4 + weight decay 0.05: 0.0605 / 75.6%; stronger augmentation (rotation 20°, scale 0.3, shift 0.15, shear 0.2, speed 0.7–1.4, frame drop 0.2, hand drop 0.25): 0.0583 / 75.8%; both: 0.0575 / 75.9%. All within noise (CI about ±0.006 EER, ±2 points top-1). Regularization does reduce the train/val gap (train accuracy 92% instead of 98% at epoch 30) and keeps validation at its best level through epoch 30 instead of degrading, but the best validation result doesn't improve: overfitting is not what limits the GRU; the architecture or the amount of data is.
- Conv + transformer vs GRU (new defaults, 30 epochs, val EER / top-1 at k = 1): GRU 0.0575 / 75.9% (2.3M parameters, 9 s/epoch); conv + transformer width 192, 2 stacks 0.0676 / 73.2% (1.2M, 7 s/epoch); width 256, 3 stacks 0.0651 / 73.4% (3.0M, 11 s/epoch); width 384, 2 stacks 0.0680 / 73.0% (4.5M, 12 s/epoch). Size makes no difference; the transformers still improve until epoch 22–30 and reach only 86–89% training accuracy, which points to undertraining rather than overfitting.
- Conv + transformer follow-ups (width 192, val EER / top-1 at k = 1 / top-1 at k = 5): 80 epochs 0.0657 / 74.0% / 85.0%, peaking around epoch 26–30 and declining afterwards (not undertrained); dropout 0.2 instead of 0.4: 0.0621 / 75.3% / 86.4%, within noise of the GRU (0.0575 / 75.9% / 86.1%). Dropout 0.4 was too strong for the transformer.
- Across 11 runs, different architectures (GRU, conv + transformer, 1.2–4.5M parameters) and regularization settings converge to about the same level (EER ≈ 0.058–0.062, top-1 ≈ 76% at k = 1), which suggests the models are limited by the data (2,172 training signs, ~18 clips each, 40 signers) rather than by the architecture.
- Unseen signers: the GRU (current defaults) retrained without 7 training signers (P16, P14, P51, P30, P23, P7, P43; 20% of the training clips) vs trained on all (`gru_reg_aug`), on the val signs. On the 7 held-out signers' clips (1,071) EER / top-1 at k = 1 go from 0.0605 / 75.6% to 0.0654 / 73.6%; on the other signers' clips (4,255), which only see the smaller training set, from 0.0568 / 76.0% to 0.0586 / 74.7%. Signer gap: +0.003 EER and −0.7 points top-1 at k = 1 (−1.3 points at k = 5), within noise (CIs about ±0.01 EER, ±2.5 points top-1 per group). The validation numbers are thus hardly inflated by signer overlap, more signers alone are unlikely to help much, and 20% less training data costs only ~1 point; more signs (vocabulary) is the more promising lever, so the overlap of Sem-Lex's vocabulary with ASL Citizen's matters.
- Vocabulary scaling (GRU, current defaults, same number of training steps, val EER / top-1 at k = 1): 543 signs (10k clips) 0.0766 / 64.2%; 1,086 (20k) 0.0658 / 70.1%; 1,629 (30k) 0.0617 / 73.9%; 2,172 (40k) 0.0575 / 75.9% (CIs about ±2.3 points top-1). Log-linear: each doubling of the training vocabulary gives about +6 points top-1 and −0.009 EER at k = 1 (+5.5 points, −0.0065 at k = 5, flattening slightly). Unlike architecture and regularization, more signs (with their clips) is a real lever; for comparison, 20% fewer clips of the same signs (signer holdout) cost only ~1.3 points. Extrapolated, doubling the vocabulary with Sem-Lex could give roughly +4–6 points top-1 at k = 1, depending on how many of its signs are new.
- Kaggle ASL Signs vocabulary vs ASL Citizen: 208 of its 250 signs match an ASL Citizen gloss by label (ignoring case, sense numbers and punctuation); most of the other 42 are the same signs under different labels (e.g. dad/FATHER, mom/MOTHER). Kaggle would add many clips for existing signs but few new signs.
- Hardest val signs for the GRU (k = 5, by AUC): CHEERLEADER2, BOW2, SKATEBOARDING3, EGO, CRAWL1, FAN, LONGWORD, LUNCH1.
- Frame size (backfilled from the videos' first decoded frame): 80,184 clips 640x480, 3,211 at 960x540, 4 at 480x640.
- Signers: 26 of 52 recorded nearly the whole vocabulary (~3,000 clips each); P13 and P19 have 2 clips each.
- Coordinates (0.1–99.9 percentiles): hands x −0.02–1.03, y −0.01–1.12; upper body x −0.1–1.14, y 0.17–1.8 (arms below the frame when the hands are down). `right_hand` lies mostly on the image's left (x 0.08–0.76 at 1–99%), consistent with the Kaggle hand label convention.
- Visual inspection of missed hands during signing: motion blur (most common), overlapping or touching hands, and hands seen edge-on. These are limits of the footage and hand models, not of the setup, so the holistic extractor is kept and gaps are handled in the loader.

Sem-Lex:
- Access: agree to the terms of use in a Google form linked from https://github.com/leekezar/SemLex, which gives the download links. The repository says pose files will be "available soon" (apparently not updated since 2023), so landmarks are extracted with our own setup.
- Of the 84,568 benchmark videos, 65,935 are matched to ASL-LEX or ASL SignBank; 18,393 only have free-text descriptions (not usable as sign labels). The ASL-LEX alignment links its signs to ASL Citizen's. The paper is CC BY-NC-SA 4.0; the data's own terms are in the form.

Other datasets (for more training signs; signs of other sign languages are all new classes):
- ASL: WLASL (2,000 signs, 21k clips, YouTube links, many dead), MS-ASL (1,000 signs, 25.5k clips, 200+ signers, YouTube links); both mostly common words overlapping with ASL Citizen.
- Other languages: MM-WLAuslan (Auslan, 3,215 signs, 282k videos from 4 camera views, 73 signers, public, CC BY-NC-SA 4.0), Slovo (Russian, 1,000 signs, 20k videos, 194 signers, public on Kaggle, 16 GB), SLR500 (Chinese, 500 signs, 125k videos, 50 signers), NMFs-CSL (Chinese, 1,067 signs), AUTSL (Turkish, 226 signs, 38k videos, 43 signers).
- Dictionaries with one video per sign (reference sets, not training data): ASL-LEX, ASL SignBank, the Swedish lexicon.
- Related work: "Representing Signs as Signs" (Vandendriessche et al., Ghent University, 2025, arXiv 2502.20171) trains an embedding model on MediaPipe pose and hand landmarks of ASL Citizen and recognizes signs one-shot in a Flemish Sign Language dictionary of 10,235 signs: recall@1 0.374, recall@5 0.670, MRR 0.508; trained on a Flemish corpus with 292 signs instead, recall@1 is only 0.089. Supports both the vocabulary finding and cross-language transfer (relevant for Swedish Sign Language later).

MM-WLAuslan:
- Public Google Drive folder `1EQ1Nh3lidEcu1QLFw0IjRN7YqEq1N48q` (linked from https://uq-cvlab.github.io/MM-WLAuslan-Dataset/), readable with the rclone remote: `rclone copy "personal gdrive:" <dir> --drive-root-folder-id 1EQ1Nh3lidEcu1QLFw0IjRN7YqEq1N48q --include <path>`. License CC BY-NC-SA 4.0.
- Layout: `{Train,Valid,Test-STU,Test-ITW,Test-SYN,Test-TED}/<camera>/{rgb,depth}.zip` for cameras `Kinect_F`, `Kinect_L`, `Kinect_R` (Kinect V2 front, left-front, right-front) and `RealSense_F`; videos inside are `rgb/<id>_<kf|kl|kr|rf>_rgb.mp4`. RGB `Kinect_F` sizes: Train 14.5 GB, Valid 2.4, STU 2.6, ITW 4.5, SYN 3.5, TED 2.0 GB (`RealSense_F` is ~5x larger). Also a `Test-MTV` zip (multi-view challenge) and `WWW_CV_ISLR_Challenge/` (a copy of Train/Valid for the WWW 2025 challenge).
- Labels: `Annotation/Labels & Split/<subset>.json` maps a random 5-digit sample id to its gloss (3,215 glosses, all in every subset): Train 38,580 (12 per gloss), Valid and each test subset 6,430 (2 per gloss). The same sample id is shared by its four camera videos. `Dictionary.json` gives each gloss's English keywords and region.
- No per-sample signer ids are released, although the paper lists them as part of each sample; the label files, dictionary and pose pickles (gzipped `{id: float64 array (frames, 136, 3)}`, AlphaPose keypoints) contain none. Per the paper: 73 signers; Train 55, Valid 53, STU 12, ITW 15, SYN 62, TED 63 signers; 18 signers appear only in test subsets.
- Test subsets: STU is the studio setting of the training data; ITW and SYN replace the green screen with real or synthetic backgrounds; TED removes frames at the start or end and changes playback speed (could cut signs).
- Videos (`Kinect_F`, 50 from Valid): 512x408, 30 fps, 51–127 frames (mean 85); green screen, the signer cropped to about the thighs, small in the frame. Our extractor runs as on ASL Citizen (~45 frames/s single-process); hands are detected during signing. Estimated extraction at ~140 frames/s with 8 workers: ~1 h per 6,430 clips, ~6.5 h for Train.

MediaPipe extraction speed (machine: Ryzen 9 5950X, 16 cores / 32 threads; RTX 3090):
- HolisticLandmarker on the CPU, one process: ~22 ms per frame wall, ~40 ms CPU (uses ~1.7 cores).
- Parallel throughput peaks at 8 worker processes (~136 frames/s in short benchmarks, ~150 expected with long-lived workers) and drops with more workers (16: ~103, 32: ~84). CPU time per frame doubles at 8 workers (contention for physical cores, SMT, clock speed). Pinning workers to cores makes it slower (8 pinned: 107). The Python API does not expose MediaPipe's thread count.
- GPU: the HolisticLandmarker fails on the GPU (its face blendshapes model has unsupported operations, and the bundle cannot be built without it). The separate FaceLandmarker, HandLandmarker and PoseLandmarker run on the GPU (~6 ms per frame each, ~18 ms together), but multi-process throughput saturates at ~64 frames/s: MediaPipe's OpenGL path processes one frame at a time with CPU-GPU synchronization, and model loading takes 0.5–2 s per video. Using the GPU well would require reimplementing MediaPipe's pipeline with batched inference.

MediaPipe:
- MediaPipe 1.0 has removed the legacy Holistic API that the Kaggle landmarks were extracted with. Its Tasks API has a `HolisticLandmarker` (model: `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`) as well as separate face, hand and pose landmarkers. The Tasks API also provides the landmark connection definitions (`FaceLandmarksConnections`, `HandLandmarksConnections`, `PoseLandmarksConnections`).
