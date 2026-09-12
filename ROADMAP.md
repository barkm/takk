# Roadmap

Status of the work and the plan ahead. Update this file when a step is finished, a decision is made, or the plan changes. See README.md for the task, approach and evaluation protocol.

## Steps

1. **Data exploration and pipeline (Kaggle ASL Signs)** — done
   - Kaggle ASL Signs downloaded; exploration in `scripts/eda_asl_signs.py` (figures in `outputs/eda/`). The raw Kaggle files have since been deleted to free disk space; only the landmark store in `data/processed/kaggle_asl_signs/` is kept, so re-running the exploration script requires downloading again.
   - Common, dataset-agnostic landmark store (`landmarks.py`) and Kaggle adapter (`datasets/kaggle_asl_signs.py`). All 543 landmarks are stored; subsets are selected at load time.
   - Landmark groups relevant for signing (`LANDMARK_GROUPS`, ~100 landmarks: hands, upper body, face reference points, lips).
   - Clip viewer rendering sequences as animated GIFs (`scripts/view_clips.py`).

2. **ASL Citizen data** — next
   - Download (~46 GB zip) and inspect the contents: videos, metadata, official splits. — done (see findings)
   - Choose the landmark extractor: run MediaPipe (Tasks API) on a handful of videos, check the result in the viewer, and measure extraction speed. — done (see decisions and findings)
   - Adapter extracting landmarks from the videos into the common layout (`datasets/asl_citizen.py`, resumable). — done
   - Full extraction (~12–14 h, see README). — pending
   - Delete `data/raw/ASL_Citizen.zip` (46 GB) once the extraction has been checked.
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

- **Landmark extractor:** MediaPipe Tasks `HolisticLandmarker` (successor of the legacy Holistic) in video mode with default settings, on the CPU with 8 worker processes (`extraction.py`). Its landmarks match the Kaggle layout and hand label convention. MediaPipe's GPU mode was rejected: the holistic model fails on the GPU, and the separate face/hand/pose models run on the GPU but scale worse than the CPU (see findings).
- **Frame rate:** each clip's source fps is stored in the store metadata (null when unknown), so sequences can be resampled to a common rate.
- **Resumable extraction:** long extractions write the store in chunks of 1,024 clips and resume from the last finished chunk (`write_store_resumable`).

## Open decisions

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
- License (`use.txt`, Microsoft Research License Terms): non-commercial research only; the data and modifications of it (e.g. extracted landmarks) may not be distributed; personal data must be destroyed when the research is completed.
- Layout: `ASL_Citizen/videos/*.mp4` (83,399 videos, ~50 GB) and `ASL_Citizen/splits/{train,val,test}.csv` with columns `Participant ID`, `Video file`, `Gloss`, `ASL-LEX Code`.
- Official splits are signer-disjoint: 35 / 6 / 11 signers with 40,154 / 10,304 / 32,941 videos. All 2,731 glosses appear in every split; 21–45 videos per gloss (mean 31). Videos per signer are very uneven (2 to 3,004, median 1,496).
- Glosses are cleaned labels (e.g. file `NOT MIND` → gloss `NOTMIND`, `SAIL` → `SAIL1`). 2,723 ASL-LEX codes; a few codes are shared by several glosses.
- Videos (sample of 300): mostly H.264 640x480, some 960x540 and MPEG-4. Frame rate varies (mostly ~30 fps, also 25 and 15 fps). Mean length 2.7 s / 80 frames (33–366), so ~6.7M frames in total and a ~44 GB landmark store at float32.

- First extraction test (32 videos): face and pose detected in 100% of frames; `left_hand` in 33%, `right_hand` in 42%, both in 30%, neither in 55% (recordings start and end with the hands down). Two-handed signing is present, unlike Kaggle.

MediaPipe extraction speed (machine: Ryzen 9 5950X, 16 cores / 32 threads; RTX 3090):
- HolisticLandmarker on the CPU, one process: ~22 ms per frame wall, ~40 ms CPU (uses ~1.7 cores).
- Parallel throughput peaks at 8 worker processes (~136 frames/s in short benchmarks, ~150 expected with long-lived workers) and drops with more workers (16: ~103, 32: ~84). CPU time per frame doubles at 8 workers (contention for physical cores, SMT, clock speed). Pinning workers to cores makes it slower (8 pinned: 107). The Python API does not expose MediaPipe's thread count.
- GPU: the HolisticLandmarker fails on the GPU (its face blendshapes model has unsupported operations, and the bundle cannot be built without it). The separate FaceLandmarker, HandLandmarker and PoseLandmarker run on the GPU (~6 ms per frame each, ~18 ms together), but multi-process throughput saturates at ~64 frames/s: MediaPipe's OpenGL path processes one frame at a time with CPU-GPU synchronization, and model loading takes 0.5–2 s per video. Using the GPU well would require reimplementing MediaPipe's pipeline with batched inference.

MediaPipe:
- MediaPipe 1.0 has removed the legacy Holistic API that the Kaggle landmarks were extracted with. Its Tasks API has a `HolisticLandmarker` (model: `https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task`) as well as separate face, hand and pose landmarkers. The Tasks API also provides the landmark connection definitions (`FaceLandmarksConnections`, `HandLandmarksConnections`, `PoseLandmarksConnections`).
