"""Prepare the WLASL clips for training alongside ASL Citizen.

WLASL is ASL, so its glosses are mapped onto ASL Citizen's labels (`wlasl.map_signs`, see ROADMAP.md):
a gloss matching one ASL Citizen gloss joins that class, a gloss matching several variants of one
gloss, a held-out sign, or a new sign the hash split doesn't put in train is left out of training
(prepared with a null split and a "wlasl:" label). Clips that join an ASL Citizen class get that
sign's ASL-LEX phonological features, like the ASL Citizen clips do.

Run from the repo root: uv run scripts/prepare_wlasl.py
Writes data/prepared/<store name>-<preparation config id>/.
"""

import argparse
from pathlib import Path

import polars as pl

from isolated_sign_validation.datasets import asl_citizen, asl_lex, wlasl
from isolated_sign_validation.landmarks import LandmarkStore
from isolated_sign_validation.preparation import PrepConfig, PreparedData, prepare_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=wlasl.STORE_DIR)
    args = parser.parse_args()

    clips = LandmarkStore(args.store).clips.with_row_index("row")
    asl_citizen_signs = LandmarkStore(asl_citizen.STORE_DIR).clips["sign"].unique().to_list()
    mapping = wlasl.map_signs(clips["sign"].unique().to_list(), asl_citizen_signs)
    mapped = pl.col("sign").replace_strict(mapping, return_dtype=pl.String)
    clips = (
        clips.with_columns(mapped=mapped)
        .with_columns(
            split=pl.when(pl.col("mapped").is_not_null()).then(pl.lit("train")),
            sign=pl.col("mapped").fill_null("wlasl:" + pl.col("sign")),
        )
        .drop("mapped")
    )
    trained = clips.filter(pl.col("split").is_not_null())
    known = set(asl_citizen_signs)
    print(f"{trained.height} of {clips.height} clips train, on {trained['sign'].n_unique()} signs of which "
          f"{trained.filter(~pl.col('sign').is_in(list(known)))['sign'].n_unique()} are new to ASL Citizen")  # fmt: skip

    codes = asl_citizen.read_videos(asl_citizen.RAW_DIR).select(sign="Gloss", Code="ASL-LEX Code").unique()
    phonology = codes.join(asl_lex.read_phonology(asl_lex.RAW_DIR), on="Code", how="left").drop("Code")
    clips = clips.join(phonology, on="sign", how="left", maintain_order="left")

    config = PrepConfig()
    out = Path("data/prepared") / f"{args.store.name}-{config.id()}"
    prepare_store(args.store, clips, config, out)

    data = PreparedData(out)
    print(
        data.clips.group_by("split")
        .agg(
            clips=pl.len(),
            signs=pl.col("sign").n_unique(),
            mirrored=(pl.col("dominant") == "left").mean().round(3),
            frames_median=pl.col("n_frames").median(),
            frames_p95=pl.col("n_frames").quantile(0.95),
            frames_max=pl.col("n_frames").max(),
        )
        .sort("split")
    )
    print(f"{data.frames.nbytes / 2**30:.2f} GB of frames")


if __name__ == "__main__":
    main()
