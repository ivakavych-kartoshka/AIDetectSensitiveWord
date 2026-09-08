"""Augment Vietnamese training data - heavily oversample hate_speech and
insult samples using existing augmentation techniques (homoglyph, leetspeak,
abbreviation, case variation, special chars, zero-width).

Usage:
  python run_augment_vi.py --out dataset/clean/train_vi_aug.csv --hs-variants 5 --insult-variants 2
"""

import argparse
import sys

import pandas as pd

from src.config.config import Config
from src.data.augmentation import augment_text

config = Config()

LABEL_COLS = [
    "insult",
    "hate_speech",
    "threat",
    "harassment",
    "sexual",
    "spam",
]


def main():

    parser = argparse.ArgumentParser(description="Augment Vietnamese training data")

    parser.add_argument("--out", default="dataset/clean/train_vi_aug.csv")
    parser.add_argument("--hs-variants", type=int, default=5, help="Variants per hate_speech=1 sample")
    parser.add_argument("--insult-variants", type=int, default=2, help="Variants per insult-only sample")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    df = pd.read_csv(config.train_file)

    vi = df[df["language"] == "vi"].copy()

    hs_mask = vi["hate_speech"] == 1
    insult_mask = vi["insult"] == 1
    insult_only_mask = insult_mask & ~hs_mask

    print(f"VI samples           : {len(vi)}")
    print(f"  hate_speech        : {int(hs_mask.sum())}")
    print(f"  insult-only        : {int(insult_only_mask.sum())}")
    print(f"  both               : {int((hs_mask & insult_mask).sum())}")

    rows = []

    # --- Original rows ---
    rows.append(vi)

    n_added = 0

    # --- Augment hate_speech samples (also insult) ---
    hs_df = vi[hs_mask]

    print(f"\nAugmenting {len(hs_df)} hate_speech samples x{args.hs_variants} variants...")

    hs_aug = []

    for _, row in hs_df.iterrows():

        text = row["normalized_text"]

        for variant in augment_text(text, num_variants=args.hs_variants):

            new_row = row.copy()

            new_row["normalized_text"] = variant
            new_row["text"] = variant

            hs_aug.append(new_row)

            n_added += 1

    rows.append(
        pd.DataFrame(hs_aug, columns=vi.columns)
    )

    # --- Augment insult-only samples (fewer variants) ---
    if args.insult_variants > 0:

        insult_only_df = vi[insult_only_mask]

        print(f"Augmenting {len(insult_only_df)} insult-only samples x{args.insult_variants} variants...")

        insult_aug = []

        for _, row in insult_only_df.iterrows():

            text = row["normalized_text"]

            for variant in augment_text(text, num_variants=args.insult_variants):

                new_row = row.copy()

                new_row["normalized_text"] = variant
                new_row["text"] = variant

                insult_aug.append(new_row)

                n_added += 1

        rows.append(
            pd.DataFrame(insult_aug, columns=vi.columns)
        )

    aug_df = pd.concat(rows, ignore_index=True)

    print(f"\nOriginal VI  : {len(vi)}")
    print(f"Augmented    : {n_added}")
    print(f"Total        : {len(aug_df)}")
    print(f"New ratio    : hate_speech={(aug_df['hate_speech']==1).sum()/len(aug_df):.4f} "
          f"insult={(aug_df['insult']==1).sum()/len(aug_df):.4f}")

    aug_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()