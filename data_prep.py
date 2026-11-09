"""
Load and clean athletes.csv with the same logic used in Part A.

Returns one pandas DataFrame of encoded features plus the target `total_lift`
(deadlift + candj + snatch + backsq). The four lift columns are removed from the
features so the target does not leak.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PATH = Path.home() / "Downloads" / "athletes.csv"
TARGET = "total_lift"

REQUIRED_COLS = ['athlete_id', 'region', 'age', 'weight', 'height', 'gender',
                 'eat', 'train', 'background', 'experience', 'schedule', 'howlong',
                 'deadlift', 'candj', 'snatch', 'backsq']
NUMERIC_COLS = ['age', 'weight', 'height', 'deadlift', 'candj', 'snatch', 'backsq']
LIFT_COLS = ['deadlift', 'candj', 'snatch', 'backsq']
SURVEY_COLS = ['background', 'experience', 'schedule', 'howlong', 'eat']
MULTISELECT_COLS = ['eat', 'background', 'schedule', 'experience']


def _sanitize_columns(cols):
    """Make column names safe for every library (LightGBM rejects special characters)."""
    seen, out = {}, []
    for c in cols:
        new = re.sub(r'[^0-9a-zA-Z]+', '_', str(c)).strip('_').lower()
        if new in seen:
            seen[new] += 1
            new = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
        out.append(new)
    return out


def load_athletes(path=DEFAULT_PATH, version="v1"):
    """
    v1: keeps athletes who declined survey questions (declines -> all-zero dummies)
    v2: drops athletes who declined any survey question
    """
    assert version in ("v1", "v2")

    df = pd.read_csv(path, usecols=REQUIRED_COLS, low_memory=False)
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Remove missing rows + not relevant columns
    df = df.dropna(subset=REQUIRED_COLS).drop(columns=['train'])

    # Remove outliers
    df = df[df['weight'] < 1500]
    df = df[df['gender'] != '--']
    df = df[df['age'] >= 18]
    df = df[(df['height'] < 96) & (df['height'] > 48)]

    male_ok = (df['gender'] == 'Male') & (df['deadlift'] <= 1105)
    female_ok = (df['gender'] == 'Female') & (df['deadlift'] <= 636)
    df = df[(df['deadlift'] > 0) & (male_ok | female_ok)]

    df = df[(df['candj'] > 0) & (df['candj'] <= 395)]
    df = df[(df['snatch'] > 0) & (df['snatch'] <= 496)]
    df = df[(df['backsq'] > 0) & (df['backsq'] <= 1069)]

    # Clean survey data
    df = df.replace({'Decline to answer|': np.nan})
    if version == "v2":
        df = df.dropna(subset=SURVEY_COLS)

    df = df.drop_duplicates(subset='athlete_id').drop(columns=['athlete_id'])

    # Target
    df[TARGET] = df[LIFT_COLS].sum(axis=1)
    df = df.drop(columns=LIFT_COLS)

    # One-hot encode multiselect survey columns
    for col in MULTISELECT_COLS:
        dummies = df[col].str.get_dummies(sep='|').add_prefix(f'{col}_')
        df = pd.concat([df.drop(columns=col), dummies], axis=1)
    df = df.drop(columns=df.filter(like='Decline to answer').columns)

    df = pd.get_dummies(df, columns=['region', 'gender', 'howlong'],
                        drop_first=True, dtype=int)
    df.columns = _sanitize_columns(df.columns)

    return df.reset_index(drop=True)
