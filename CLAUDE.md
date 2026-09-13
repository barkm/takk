# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

## Project Context

This system takes as input a sequence of human pose landmarks and a sign, and outputs yes/no depending on whether the landmark sequence is signing that sign. See README.md for the approach, datasets, and evaluation protocol.

**Read ROADMAP.md at the start of a session.** It holds the current plan, step status, open decisions and findings. Keep it updated: mark steps done, record decisions and new findings, and change the plan there when it changes.

Key constraints:
- **Open vocabulary.** The model must validate signs not seen during training, from few (possibly one) reference clips. Frame it as embedding + similarity verification, not closed-set classification.
- **Dataset-agnostic data pipeline.** Each dataset (Kaggle ASL Signs, ASL Citizen, WLASL, ...) gets an adapter into a common landmark format. Nothing downstream may depend on a specific dataset.
- **Consistent landmarks.** All video datasets must be extracted with one fixed MediaPipe setup (same version and settings) into the common landmark layout: `extraction.py`, the Tasks API HolisticLandmarker on the CPU. ASL Citizen is the primary dataset; Kaggle ASL Signs (landmarks from the removed legacy MediaPipe Holistic) is only an optional extra.
- **No leakage in evaluation.** Hold out both signers and signs. A test signer or held-out sign must never appear in training. Use the splits from `splits.py`.
- **Optimize for model quality.** Deployment constraints (model size, latency) and threshold selection are out of scope for now.
- **Use uv, never bare python/pip.** `uv add` for dependencies, `uv run` for scripts, `uvx` for one-off CLI tools.
- **Data lives in the git-ignored `data/` directory** (raw downloads in `data/raw/`, converted landmark stores in `data/processed/<dataset>/`).
- **Code layout.** Library code in `src/isolated_sign_validation/` (common landmark format in `landmarks.py`, one adapter per dataset in `datasets/`), one-off scripts in `scripts/`, tests in `tests/` (`uv run pytest`).
- **Headless machine.** The machine is used remotely over SSH with no display. Use matplotlib's Agg backend and save figures to the git-ignored `outputs/` directory; never `plt.show()`.
- **Long term (not in scope yet):** Swedish Sign Language signs from teckensprakslexikon.su.se.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.