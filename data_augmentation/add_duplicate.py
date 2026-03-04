"""
add_duplicate.py

For each unique paragraph ID in the augmented CSVs, appends one additional
duplication row (dup3 for train, or the next unused dup index for dev).
Overwrites the original files in place.

The reason of existence of this file is that I messed up the augmentation counts, 
I should produce 8x the original samples not 7x.
"""

import os
import pandas as pd

FILES = [
    'data_augmentation/augmented_train_data.csv',
    'data_augmentation/augmented_dev_data.csv',
]


def add_one_duplicate(csv_path: str) -> None:
    df = pd.read_csv(csv_path, dtype={'par_id': str})

    # Drop accidental duplicate header row
    df = df[df['label'].astype(str) != 'label'].copy()
    df['label'] = df['label'].astype(int)

    # Derive orig_par_id and aug_type from the par_id suffix
    df['orig_par_id'] = df['par_id'].str.replace(r'_[^_]+$', '', regex=True)
    df['aug_type']    = df['par_id'].str.extract(r'_([^_]+)$')

    # Find rows that are already duplicates (aug_type matches 'dup<number>')
    dup_mask = df['aug_type'].str.match(r'^dup\d+$', na=False)
    dup_df   = df[dup_mask]

    if dup_df.empty:
        print(f'[{os.path.basename(csv_path)}] No existing dup rows found — skipping.')
        return

    # Determine the next dup index
    max_dup_n   = dup_df['aug_type'].str.extract(r'(\d+)')[0].astype(int).max()
    new_dup_tag = f'dup{max_dup_n + 1}'

    # For each orig_par_id pick the lowest-numbered dup as the source row
    source_df = (
        dup_df
        .sort_values('aug_type')
        .groupby('orig_par_id', as_index=False)
        .first()
    )

    # Build new rows
    new_rows = source_df.copy()
    new_rows['par_id']   = new_rows['orig_par_id'] + '_' + new_dup_tag
    new_rows['aug_type'] = new_dup_tag

    # Drop helper columns and concatenate
    helpers = ['orig_par_id', 'aug_type']
    result  = pd.concat(
        [df.drop(columns=helpers), new_rows.drop(columns=helpers)],
        ignore_index=True,
    )

    result.to_csv(csv_path, index=False)

    print(
        f'[{os.path.basename(csv_path)}] '
        f'Added {len(new_rows):,} {new_dup_tag} rows — '
        f'{len(df):,} → {len(result):,} total rows'
    )


if __name__ == '__main__':
    for path in FILES:
        add_one_duplicate(path)
