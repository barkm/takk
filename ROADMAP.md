# Roadmap

Status of the work and the plan ahead. Update this file when a step is finished, a decision is made, or the plan changes. See README.md for the task, approach and evaluation protocol.

## Steps

1. **Data exploration and pipeline (Kaggle ASL Signs)** — done
   - Kaggle ASL Signs downloaded; exploration in `scripts/eda_asl_signs.py` (figures in `outputs/eda/`). The raw Kaggle files have since been deleted to free disk space; only the landmark store in `data/processed/kaggle_asl_signs/` is kept, so re-running the exploration script requires downloading again.
   - Common, dataset-agnostic landmark store (`landmarks.py`) and Kaggle adapter (`datasets/kaggle_asl_signs.py`). All 543 landmarks are stored; subsets are selected at load time.
   - Landmark groups relevant for signing (`LANDMARK_GROUPS`, ~100 landmarks: hands, upper body, face reference points, lips).
   - Clip viewer rendering sequences as animated GIFs (`scripts/view_clips.py`).

2. **ASL Citizen data** — next
   - Download (~46 GB zip) and inspect the contents: videos, metadata, official splits.
   - Choose the landmark extractor: run MediaPipe (Tasks API) on a handful of videos, check the result in the viewer, and measure extraction speed.
   - Adapter extracting landmarks from the videos into the common layout, then the full extraction.
   - Exploration of the extracted data (sequence lengths, hand presence, handedness).

3. **Splits**
   - Use ASL Citizen's official signer-disjoint splits.
   - Held-out signs rotated over folds; held out across all datasets once more are added.

4. **Training data loader**
   - Select landmark groups and load them into RAM.
   - Normalize relative to the body (shoulder center and width).
   - Mirror each clip so the dominant hand (the one detected in more frames) is always on the same side, swapping hand labels.
   - Trim leading and trailing frames without hands; cap sequence length by resampling long clips.
   - Augmentation: rotation, scaling, time stretching, frame dropping.
   - Extend the viewer to show preprocessed clips.

5. **Evaluation harness**
   - k-shot verification episodes (k = 1, 3, 5): reference clips of a held-out sign from some signers; positive and negative queries from other signers.
   - Metrics: ROC-AUC, equal error rate, per-sign breakdown.

6. **Baselines**
   - DTW on normalized hand landmarks (no training).
   - Small GRU embedding trained with ArcFace, to validate the training and evaluation pipeline end to end.

7. **Strong model**
   - Conv + transformer encoder with heavy augmentation.
   - Experiments: landmark groups (hands only / + body / + face), loss (ArcFace vs supervised contrastive), sequence length.

8. **More data**
   - Sem-Lex: adapter with the same extractor; sign labels normalized with ASL Citizen via ASL-LEX. Its phonological feature annotations (handshape, location, movement) could serve as auxiliary training targets.
   - Optionally Kaggle ASL Signs, keeping in mind it is one-handed signing from a different extractor.

Later: sensitivity analysis and threshold selection; Swedish Sign Language signs from teckensprakslexikon.su.se.

## Decisions

- **ASL Citizen is the primary dataset** instead of Kaggle ASL Signs: ~11x the vocabulary (the main lever for unseen signs), 52 signers, both hands free, and videos so the landmark extractor is under our control and matches what the system will use in practice. Kaggle ASL Signs is one-handed smartphone signing with landmarks from a removed MediaPipe version.

## Open decisions

- **Landmark extractor:** MediaPipe Tasks API setup (a holistic landmarker if available, otherwise separate pose, hand and face landmarkers) and its settings.
- **Held-out sign folds for ASL Citizen:** number of folds and signs per fold; decide after inspecting the data.

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

MediaPipe:
- MediaPipe 1.0 has removed the legacy Holistic API that the Kaggle landmarks were extracted with. The Tasks API provides the landmark connection definitions (`FaceLandmarksConnections`, `HandLandmarksConnections`, `PoseLandmarksConnections`).
