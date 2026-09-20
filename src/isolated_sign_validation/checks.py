"""Whether a recording made in a browser can be used at all, and why not.

Both browser apps run this on an upload, so that an unusable recording is caught while the person is
still in front of the camera: the collection app (`collection.py`) before storing a take, the
practice app before scoring an attempt.
"""

import numpy as np

from isolated_sign_validation.extraction import VideoInfo
from isolated_sign_validation.preparation import PrepConfig, hand_presence, hide_low_hands, prepare_clip


NOTES = {
    "empty": "The recording is empty.",
    "recorded": "Recorded.",
    "no_hands": "No hands were detected. Are your hands inside the frame while signing?",
    "short": "Only {seconds:.1f} s of signing was detected. Record again, a little slower.",
    "long": "{seconds:.0f} s of signing was detected, which is too long for one sign.",
    "no_body": "Your upper body was not detected. Sit so that both shoulders are in the frame.",
    "lost_hand": "Usable, but a hand was lost in {lost:.0%} of the frames while signing.",
    "ok": "Looks good.",
}


def check_clip(landmarks: np.ndarray, info: VideoInfo, config: PrepConfig, signing: bool = True, notes: dict[str, str] = NOTES) -> tuple[bool, str, float]:  # fmt: skip
    """Whether a recording (its extracted landmarks) can be used, why not, and how steadily a hand was
    detected while signing.

    The reason is worded by `notes`, keyed as NOTES is, because the same check serves apps in
    different languages (the practice app is in Swedish, see ROADMAP-takk.md).

    A clip of not signing (`signing` false, the no_event prompts) only has to hold a recording: no
    hands, or hands moving for longer than any sign, is what it is meant to show.

    Runs the preparation the clip would go through later, so that an unusable
    recording is caught while the signer can still redo it. The hand share is measured over the
    signing itself (the first to the last frame with a hand), not over the whole recording: the rest
    before and after the sign has no hands in view by design, so over the whole clip even a clean
    recording scores around 0.4.
    """
    aspect = info.width / info.height if info.height else 1.0
    present = hand_presence(hide_low_hands(landmarks, aspect, config.max_hand_y)).any(axis=1)
    usable = prepare_clip(landmarks, info.fps, aspect, config) is not None
    with_hands = np.flatnonzero(present)
    seconds = (with_hands[-1] - with_hands[0] + 1) / info.fps if len(with_hands) else 0.0
    hand_share = float(present[with_hands[0] : with_hands[-1] + 1].mean()) if len(with_hands) else 0.0
    if not len(landmarks):
        return False, notes["empty"], 0.0
    if not signing:
        return True, notes["recorded"], hand_share
    if not len(with_hands):
        return False, notes["no_hands"], hand_share
    if seconds < config.min_hands:
        return False, notes["short"].format(seconds=seconds), hand_share
    if seconds > config.max_hands:
        return False, notes["long"].format(seconds=seconds), hand_share
    if not usable:
        return False, notes["no_body"], hand_share
    if hand_share < 0.8:
        return True, notes["lost_hand"].format(lost=1 - hand_share), hand_share
    return True, notes["ok"], hand_share
