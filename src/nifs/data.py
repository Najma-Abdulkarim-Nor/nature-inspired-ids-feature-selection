"""Dataset loaders and the shared preprocessing pipeline (paper Section III-B)."""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "class", "difficulty",
]

DATASET_NAMES = {"nsl-kdd": "NSL-KDD", "cic-ids-2017": "CIC-IDS-2017"}


def load_nslkdd(nslkdd_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the official NSL-KDD train/test split with a binary label (attack = 1)."""

    def _resolve(name):
        # supports both NSL-KDD/KDDTrain+.txt and the nested
        # NSL-KDD/KDDTrain+.txt/KDDTrain+.txt layout produced by Drive unzips
        p = os.path.join(nslkdd_dir, name)
        return os.path.join(p, name) if os.path.isdir(p) else p

    train_path, test_path = _resolve("KDDTrain+.txt"), _resolve("KDDTest+.txt")
    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p} (see data/README.md)")

    out = []
    for path in (train_path, test_path):
        df = pd.read_csv(path, names=NSLKDD_COLUMNS).drop(columns=["difficulty"])
        df["label"] = (df["class"].str.lower() != "normal").astype(int)
        out.append(df.drop(columns=["class"]))
    return out[0], out[1]


def load_cicids2017(
    cicids_dir: str, row_cap: int | None = 150_000, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate the per-day CIC-IDS-2017 CSVs and make a stratified 80/20 split.

    If the data has more than ``row_cap`` rows it is stratified-downsampled first,
    so the wrapper search stays tractable.
    """
    csv_files = sorted(glob.glob(os.path.join(cicids_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cicids_dir} (see data/README.md)")

    frames = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    full = full.rename(columns={"Label": "class"})
    full["class"] = full["class"].astype(str).str.strip()
    full["label"] = (full["class"].str.upper() != "BENIGN").astype(int)
    full = full.drop(columns=["class"])

    # CICFlowMeter produces inf for some rate features (e.g. Flow Bytes/s)
    full = full.replace([np.inf, -np.inf], np.nan)

    if row_cap is not None and len(full) > row_cap:
        full, _ = train_test_split(
            full, train_size=row_cap, stratify=full["label"], random_state=random_state
        )

    train_df, test_df = train_test_split(
        full, test_size=0.20, stratify=full["label"], random_state=random_state
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame, label_col: str = "label"):
    """Impute, one-hot encode and min-max scale, fitting statistics on train only.

    Returns ``(X_train, X_test, y_train, y_test, feature_names)``.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    thresh = int(0.5 * train_df.shape[1])
    train_df = train_df.dropna(thresh=thresh)
    test_df = test_df.dropna(thresh=thresh)

    y_train = train_df.pop(label_col).values
    y_test = test_df.pop(label_col).values

    cat_cols = train_df.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in train_df.columns if c not in cat_cols]

    for c in num_cols:
        med = train_df[c].median()
        train_df[c] = train_df[c].fillna(med)
        test_df[c] = test_df[c].fillna(med)
    for c in cat_cols:
        mode = train_df[c].mode()
        fill = mode.iloc[0] if not mode.empty else "missing"
        train_df[c] = train_df[c].fillna(fill)
        test_df[c] = test_df[c].fillna(fill)

    if cat_cols:
        train_oh = pd.get_dummies(train_df, columns=cat_cols)
        test_oh = pd.get_dummies(test_df, columns=cat_cols)
        test_oh = test_oh.reindex(columns=train_oh.columns, fill_value=0)
    else:
        train_oh, test_oh = train_df, test_df

    train_oh = train_oh.astype(float)
    test_oh = test_oh.astype(float)

    mins = train_oh.min()
    ranges = (train_oh.max() - mins).replace(0, 1)
    train_norm = (train_oh - mins) / ranges
    test_norm = ((test_oh - mins) / ranges).clip(0, 1)

    return (
        train_norm.values.astype(np.float32),
        test_norm.values.astype(np.float32),
        y_train,
        y_test,
        train_norm.columns.tolist(),
    )


def load_dataset(name: str, data_dir: str = "data", cic_row_cap: int | None = 150_000,
                 random_state: int = 42):
    """Load and preprocess a dataset by key: ``"nsl-kdd"`` or ``"cic-ids-2017"``."""
    key = name.lower()
    if key == "nsl-kdd":
        train_df, test_df = load_nslkdd(os.path.join(data_dir, "NSL-KDD"))
    elif key == "cic-ids-2017":
        train_df, test_df = load_cicids2017(
            os.path.join(data_dir, "CIC_IDS_2017", "MachineLearningCVE"),
            row_cap=cic_row_cap, random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown dataset {name!r}; choose from {sorted(DATASET_NAMES)}")
    return preprocess(train_df, test_df)
