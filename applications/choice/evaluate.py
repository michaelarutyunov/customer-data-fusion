"""Choice model evaluation — M1 lift gate over no-CDT baselines (bead 16r).

Reframes M1 success as **lift over a product-only baseline** plus an
**oracle-attainability** check, replacing the absolute 0.65 AUC floor as the
binding criterion. All predictors are scored on the SAME participant-level val
split used to train M1 (random_state=42, test_size=0.3), so there is no leakage.

Reference predictors:
  - ORACLE        : the choice set's own generative choice_probabilities
                    (a lookup, the Bayes-optimal predictor — not fit).
  - PRODUCT_ONLY  : sklearn LogisticRegression on product_features alone.
  - PERSONA_ONEHOT: sklearn LogisticRegression on a one-hot of the persona id
                    (participant_id with the trailing _NNNN index stripped).

PASS gate: lift_over_product >= 0.05 AND M1_AUC >= 0.85 * oracle_AUC.
Brier <= 0.25 and calibration slope in [0.8, 1.2] are Tier-2 informational.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from applications.choice.data import PRODUCT_DIM
from applications.choice.model import ChoiceModel

LIFT_GATE = 0.05  # CDT lift over product-only baseline (Tier-1, critical)
ORACLE_FRACTION = 0.85  # M1 must reach this fraction of the oracle ceiling
BRIER_GATE = 0.25
CAL_SLOPE_LO, CAL_SLOPE_HI = 0.8, 1.2
ABSOLUTE_FLOOR = 0.65  # informational only — NOT part of the gate


def _load_oracle_probabilities(
    choice_sets_path: Path,
) -> dict[tuple[str, str], float]:
    """Map (choice_set_id, slot) -> generative choice probability."""
    lookup: dict[tuple[str, str], float] = {}
    with open(choice_sets_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cs = json.loads(line)
            csid = cs["choice_set_id"]
            for slot, prob in cs["choice_probabilities"].items():
                lookup[(csid, slot)] = float(prob)
    return lookup


def _calibration_slope(labels: list[int], probs: list[float]) -> float:
    prob_true = np.array(labels)
    prob_pred = np.array(probs)
    try:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            prob_true, prob_pred, n_bins=10
        )
    except ValueError:
        return 1.0
    valid = (fraction_of_positives > 0) & (fraction_of_positives < 1)
    if np.sum(valid) > 1:
        slope, _ = np.polyfit(
            mean_predicted_value[valid], fraction_of_positives[valid], 1
        )
        return float(slope)
    return 1.0


def _persona_onehot(df: pd.DataFrame, encoder: OneHotEncoder | None = None):
    """One-hot the persona id (participant_id with trailing _NNNN stripped)."""
    # 'price_lex_0000' -> 'price_lex'
    persona = df["participant_id"].str.rsplit("_", n=1).str[0].to_frame(name="persona")
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        return encoder.fit_transform(persona), encoder
    return encoder.transform(persona), encoder


def evaluate(
    model_path: Path = Path("applications/choice/choice_head.pt"),
    data_path: Path = Path("applications/choice/choice_training.parquet"),
    choice_sets_path: Path = Path("data/synthetic/choice_sets.jsonl"),
    device: str = "cpu",
) -> dict:
    """Evaluate M1 against the oracle + no-CDT baselines on the held-out val split."""
    print(f"Loading model from {model_path}")
    print(f"Loading data from {data_path}")
    print(f"Loading oracle probabilities from {choice_sets_path}")

    df = pq.read_table(data_path).to_pandas()
    print(f"Loaded {len(df)} choice rows")

    # Attach oracle probabilities (join on choice_set_id + slot).
    oracle = _load_oracle_probabilities(choice_sets_path)
    df["oracle_prob"] = [
        oracle.get((csid, slot), np.nan)
        for csid, slot in zip(df["choice_set_id"], df["slot"])
    ]
    missing = int(df["oracle_prob"].isna().sum())
    if missing:
        print(f"⚠️  {missing} rows have no oracle probability (stale data?) — dropping them")
        df = df.dropna(subset=["oracle_prob"]).reset_index(drop=True)

    product_dim = len(df["product_features"].iloc[0])
    assert product_dim == PRODUCT_DIM, (
        f"product_features width {product_dim} != §0.1 board width {PRODUCT_DIM}; "
        "rebuild choice_training.parquet via applications/choice/data.py"
    )
    model = ChoiceModel(cdt_dim=128, product_dim=product_dim, hidden_dim=64, dropout=0.1).to(
        device
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Identical participant-level split used to train M1 (random_state=42, 0.3).
    participants = df["participant_id"].unique()
    train_pids, val_pids = train_test_split(participants, test_size=0.3, random_state=42)
    train_df = df[df["participant_id"].isin(train_pids)]
    val_df = df[df["participant_id"].isin(val_pids)]
    print(f"Split: {len(train_pids)} train / {len(val_pids)} val participants")
    print(f"       {len(train_df)} train / {len(val_df)} val rows")

    labels_val = val_df["chosen"].astype(int).tolist()

    # --- M1 (CDT two-tower) ---
    with torch.no_grad():
        cdt = torch.tensor(np.stack(val_df["cdt_embedding"].tolist())).float().to(device)
        prod = torch.tensor(np.stack(val_df["product_features"].tolist())).float().to(device)
        m1_probs = model(cdt, prod).squeeze(1).cpu().tolist()
    m1_auc = roc_auc_score(labels_val, m1_probs)

    # --- ORACLE (the choice set's own probabilities; not fit) ---
    oracle_auc = roc_auc_score(labels_val, val_df["oracle_prob"].tolist())

    # --- PRODUCT_ONLY baseline (product_features as-is, whatever width) ---
    x_train_prod = np.stack(train_df["product_features"].tolist())
    x_val_prod = np.stack(val_df["product_features"].tolist())
    prod_model = LogisticRegression(max_iter=1000, random_state=42).fit(
        x_train_prod, train_df["chosen"].astype(int)
    )
    product_only_auc = roc_auc_score(
        labels_val, prod_model.predict_proba(x_val_prod)[:, 1]
    )

    # --- PERSONA_ONEHOT baseline (one-hot persona id) ---
    x_train_ph, encoder = _persona_onehot(train_df)
    x_val_ph, _ = _persona_onehot(val_df, encoder)
    persona_model = LogisticRegression(max_iter=1000, random_state=42).fit(
        x_train_ph, train_df["chosen"].astype(int)
    )
    persona_onehot_auc = roc_auc_score(
        labels_val, persona_model.predict_proba(x_val_ph)[:, 1]
    )

    lift_over_product = m1_auc - product_only_auc

    # Tier-2 informational (M1 on val)
    brier = brier_score_loss(labels_val, m1_probs)
    cal_slope = _calibration_slope(labels_val, m1_probs)

    # Gate (Tier-1): lift over product-only AND oracle-attainability.
    gate_pass = (lift_over_product >= LIFT_GATE) and (
        m1_auc >= ORACLE_FRACTION * oracle_auc
    )

    print(f"\n{'=' * 60}")
    print("M1 Choice Model Evaluation — lift gate")
    print(f"{'=' * 60}")
    print(f"M1_AUC (CDT two-tower) : {m1_auc:.4f}")
    print(f"oracle_AUC (ceiling)   : {oracle_auc:.4f}   (0.85·oracle = {ORACLE_FRACTION * oracle_auc:.4f})")
    print(f"product_only_AUC       : {product_only_auc:.4f}")
    print(f"persona_onehot_AUC     : {persona_onehot_auc:.4f}")
    print(f"lift_over_product      : {lift_over_product:+.4f}   (gate ≥ {LIFT_GATE})")
    print(f"{'-' * 60}")
    print(f"absolute 0.65 floor    : {'✅' if m1_auc >= ABSOLUTE_FLOOR else 'ℹ️'} {m1_auc:.4f} (informational only)")
    print(f"Brier                  : {brier:.4f}   (Tier-2 ≤ {BRIER_GATE})")
    print(f"calibration slope      : {cal_slope:.4f}   (Tier-2 ∈ [{CAL_SLOPE_LO}, {CAL_SLOPE_HI}])")
    print(f"{'=' * 60}")

    lift_ok = lift_over_product >= LIFT_GATE
    attain_ok = m1_auc >= ORACLE_FRACTION * oracle_auc
    print(f"lift ≥ {LIFT_GATE}?           {'✅' if lift_ok else '❌'} ({lift_over_product:+.4f})")
    print(f"M1 ≥ 0.85·oracle?       {'✅' if attain_ok else '❌'} ({m1_auc:.4f} vs {ORACLE_FRACTION * oracle_auc:.4f})")
    print(f"\n{'✅ M1 PASSED the lift gate' if gate_pass else '❌ M1 FAILED the lift gate'}")

    return {
        "M1_AUC": m1_auc,
        "oracle_AUC": oracle_auc,
        "product_only_AUC": product_only_auc,
        "persona_onehot_AUC": persona_onehot_auc,
        "lift_over_product": lift_over_product,
        "brier": brier,
        "calibration_slope": cal_slope,
        "absolute_floor_0_65_met": m1_auc >= ABSOLUTE_FLOOR,
        "passed": gate_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M1 choice model (lift gate)")
    parser.add_argument("--model", type=Path, default=Path("applications/choice/choice_head.pt"))
    parser.add_argument("--data", type=Path, default=Path("applications/choice/choice_training.parquet"))
    parser.add_argument(
        "--choice-sets",
        type=Path,
        default=Path("data/synthetic/choice_sets.jsonl"),
        help="choice_sets.jsonl for oracle probabilities",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        results = evaluate(
            model_path=args.model,
            data_path=args.data,
            choice_sets_path=args.choice_sets,
            device=args.device,
        )
        exit(0 if results["passed"] else 1)
    except Exception as e:  # noqa: BLE001
        print(f"❌ ERROR: {e}")
        exit(1)


if __name__ == "__main__":
    main()
