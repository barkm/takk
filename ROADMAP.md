# Roadmap

Status of the work and the plan ahead. Update this file when a step is finished, a decision is made, or the plan changes. See README.md for the task, approach and evaluation protocol.

## Steps

1. **Data exploration and pipeline** — done
   - Kaggle ASL Signs downloaded; exploration in `scripts/eda_asl_signs.py` (figures in `outputs/eda/`).
   - Common, dataset-agnostic landmark store (`landmarks.py`) and Kaggle adapter (`datasets/kaggle_asl_signs.py`). All 543 landmarks are stored; subsets are selected at load time.
   - Landmark groups relevant for signing (`LANDMARK_GROUPS`, ~100 landmarks: hands, upper body, face reference points, lips).
   - Clip viewer rendering sequences as animated GIFs (`scripts/view_clips.py`).

2. **Splits** — next
   - Held-out signers (validation and test each with both left- and right-dominant signers).
   - Held-out signs rotated over 5 folds: per fold 200 training signs, 25 validation signs, 25 test signs.
   - Works on combined metadata, so held-out signs are held out across all datasets once more are added.

3. **Training data loader**
   - Select landmark groups and load them into RAM (~4 GB for Kaggle).
   - Normalize relative to the body (shoulder center and width).
   - Mirror each clip so the dominant hand (the one detected in more frames) is always on the same side, swapping hand labels.
   - Trim leading and trailing frames without hands; cap sequence length by resampling long clips.
   - Augmentation: rotation, scaling, time stretching, frame dropping.
   - Extend the viewer to show preprocessed clips.

4. **Evaluation harness**
   - k-shot verification episodes (k = 1, 3, 5): reference clips of a held-out sign from some signers; positive and negative queries from other signers.
   - Metrics: ROC-AUC, equal error rate, per-sign breakdown.

5. **Baselines**
   - DTW on normalized hand landmarks (no training).
   - Small GRU embedding trained with ArcFace, to validate the training and evaluation pipeline end to end.

6. **Strong model**
   - Conv + transformer encoder with heavy augmentation.
   - Experiments: landmark groups (hands only / + body / + face), loss (ArcFace vs supervised contrastive), sequence length.

7. **More data: ASL Citizen**
   - Adapter running MediaPipe on the videos; sign labels normalized across datasets.
   - Check landmark consistency with Kaggle early (see findings).

Later: sensitivity analysis and threshold selection; Swedish Sign Language signs from teckensprakslexikon.su.se.

## Open decisions

- **Signer split:** fixed 15/3/3 (train/validation/test) signers, or cross-validation over signers (more reliable with only 21 signers, ~5x training cost). Recommended: fixed split.

## Findings

Kaggle ASL Signs:
- 94,477 clips of 250 signs by 21 signers. Balanced: 299–415 clips per sign, 3.3k–5k clips per signer.
- Hand detections are almost entirely `left_hand` or entirely `right_hand` per signer: 9 left-dominant, 12 right-dominant signers.
- The hand labels appear to refer to the signer's own hands in an unmirrored image (`left_hand` shows up on the image's right side). Seen in a few clips only.
- 39% of frames have no hand detected; every clip has a hand in at least one frame.
- Sequence length: median 22 frames, 95th percentile 135, max 537. 43 clips have gaps in their frame numbers (closed by the adapter). The capture frame rate is unknown.
- Coordinates go far outside [0, 1] (x from −1.3 to 2.9, y up to 3.6), mostly from out-of-frame lower-body pose landmarks.
- Some clips are very short and seem to capture only part of the sign.

MediaPipe:
- MediaPipe 1.0 has removed the legacy Holistic API that the Kaggle landmarks were extracted with. Landmarks from the newer Tasks API may differ, which matters for adding video datasets consistently.
