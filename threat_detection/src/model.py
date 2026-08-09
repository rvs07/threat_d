# src/model.py

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics  import (classification_report,
                               confusion_matrix,
                               roc_auc_score)
from sklearn.decomposition import PCA
from config import (FEATURE_FILE, MODEL_FILE, MODEL_DIR,
                    CONTAMINATION, RANDOM_STATE, N_ESTIMATORS)


from src.features import MODEL_FEATURES


# ── Data loader ────────────────────────────────────────────────────────────

def _load_features(path: str):
    """Load feature CSV and return X matrix + metadata."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"  [+] Loaded feature data : {df.shape}")

    # Drop rows where any model feature is NaN
    before = len(df)
    df.dropna(subset=MODEL_FEATURES, inplace=True)
    if len(df) < before:
        print(f"  [!] Dropped {before - len(df)} rows with NaN features.")

    X   = df[MODEL_FEATURES].values
    y   = (df["label"] != "normal").astype(int).values   # 0=normal 1=attack
    return df, X, y


# ── Model training ─────────────────────────────────────────────────────────

def _train_model(X: np.ndarray) -> IsolationForest:
    """Fit Isolation Forest — fully unsupervised (no y needed)."""
    print(f"\n  [+] Training Isolation Forest …")
    print(f"      n_estimators  = {N_ESTIMATORS}")
    print(f"      contamination = {CONTAMINATION}")
    print(f"      random_state  = {RANDOM_STATE}")

    model = IsolationForest(
        n_estimators  = N_ESTIMATORS,
        contamination = CONTAMINATION,
        random_state  = RANDOM_STATE,
        n_jobs        = -1,
    )
    model.fit(X)
    print("  [✔] Model training complete.")
    return model


# ── Scoring & thresholding ─────────────────────────────────────────────────

def _score_and_predict(model: IsolationForest,
                       X    : np.ndarray,
                       df   : pd.DataFrame) -> pd.DataFrame:
    """
    Add anomaly scores and binary predictions to the dataframe.

    score_raw  : raw decision function output (more negative = more anomalous)
    anomaly_score : normalised to 0-1  (1 = most anomalous)
    prediction : 1 = anomaly, 0 = normal
    """
    raw_scores = model.decision_function(X)   # range ≈ -0.5 … +0.5
    predictions = model.predict(X)             # -1=anomaly, +1=normal

    # Normalise score to 0-1  (invert so HIGH score = HIGH risk)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    norm_scores  = 1 - (raw_scores - min_s) / (max_s - min_s)

    df = df.copy()
    df["score_raw"]     = raw_scores.round(6)
    df["anomaly_score"] = norm_scores.round(4)
    df["prediction"]    = (predictions == -1).astype(int)  # 1=anomaly

    flagged = df["prediction"].sum()
    print(f"\n  [+] Anomalies flagged : {flagged:,} "
          f"/ {len(df):,}  "
          f"({flagged/len(df)*100:.1f}%)")
    return df


# ── Evaluation ─────────────────────────────────────────────────────────────

def _evaluate(df: pd.DataFrame):
    """Print classification metrics against the known labels."""
    y_true = (df["label"] != "normal").astype(int)
    y_pred = df["prediction"]
    y_prob = df["anomaly_score"]

    print("\n  📊 Classification Report:")
    print("  " + "-"*50)
    report = classification_report(
        y_true, y_pred,
        target_names=["Normal", "Attack"],
        digits=3
    )
    for line in report.splitlines():
        print("  " + line)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  🔢 Confusion Matrix:")
    print(f"     True  Negatives (correct normal) : {tn:>5}")
    print(f"     False Positives (false alarms)   : {fp:>5}")
    print(f"     False Negatives (missed attacks) : {fn:>5}")
    print(f"     True  Positives (caught attacks) : {tp:>5}")

    # ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
        print(f"\n  🎯 ROC-AUC Score : {auc:.4f}")
    except Exception:
        pass

    # Per-attack-type detection rate
    print("\n  🏷️  Detection Rate by Attack Type:")
    for label in df["label"].unique():
        if label == "normal":
            continue
        subset = df[df["label"] == label]
        detected = subset["prediction"].sum()
        total    = len(subset)
        rate     = detected / total * 100 if total else 0
        bar      = "█" * int(rate / 5)
        print(f"     {label:<15} {detected:>3}/{total:<3}  "
              f"({rate:5.1f}%)  {bar}")


# ── Visualisations ─────────────────────────────────────────────────────────

def _save_plots(df: pd.DataFrame, model: IsolationForest,
                X: np.ndarray, out_dir: str):
    """Generate and save 4 diagnostic plots."""
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")

    # ── 1. Anomaly Score Distribution ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, grp in df.groupby("label"):
        ax.hist(grp["anomaly_score"], bins=50, alpha=0.6, label=label)
    ax.axvline(df[df["prediction"]==1]["anomaly_score"].min(),
               color="red", linestyle="--", label="decision threshold")
    ax.set_title("Anomaly Score Distribution by Label")
    ax.set_xlabel("Anomaly Score  (1 = most anomalous)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_distribution.png"), dpi=120)
    plt.close()
    print("  [+] Plot saved: score_distribution.png")

    # ── 2. Confusion Matrix Heatmap ────────────────────────────────────
    y_true = (df["label"] != "normal").astype(int)
    cm     = confusion_matrix(y_true, df["prediction"])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Attack"],
                yticklabels=["Normal","Attack"], ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=120)
    plt.close()
    print("  [+] Plot saved: confusion_matrix.png")

    # ── 3. PCA 2-D scatter ─────────────────────────────────────────────
    pca  = PCA(n_components=2, random_state=RANDOM_STATE)
    X2d  = pca.fit_transform(X)
    colors = df["prediction"].map({0: "steelblue", 1: "crimson"})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(X2d[:, 0], X2d[:, 1],
               c=colors, alpha=0.4, s=10)
    ax.scatter([], [], c="steelblue", label="Normal")
    ax.scatter([], [], c="crimson",   label="Anomaly")
    ax.set_title("PCA Projection — Normal vs Anomaly")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_scatter.png"), dpi=120)
    plt.close()
    print("  [+] Plot saved: pca_scatter.png")

    # ── 4. Feature Importance (mean anomaly score per feature quartile) ─
    importances = np.abs(
        np.mean([tree.feature_importances_
                 for tree in model.estimators_
                 if hasattr(tree, "feature_importances_")],
                axis=0)
    ) if hasattr(model.estimators_[0], "feature_importances_") else \
        np.ones(len(MODEL_FEATURES))

    # Use anomaly_score correlation as proxy importance
    corr_imp = (
        df[MODEL_FEATURES + ["anomaly_score"]]
          .corr()["anomaly_score"]
          .drop("anomaly_score")
          .abs()
          .sort_values(ascending=True)
          .tail(15)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_imp.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Top 15 Features — Correlation with Anomaly Score")
    ax.set_xlabel("Absolute Correlation")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=120)
    plt.close()
    print("  [+] Plot saved: feature_importance.png")


# ── Model persistence ──────────────────────────────────────────────────────

def _save_model(model: IsolationForest, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"\n  [✔] Model saved → {path}  ({size_kb:.1f} KB)")


def load_model(path: str = MODEL_FILE) -> IsolationForest:
    """Public helper — load model from disk (used by later phases)."""
    return joblib.load(path)


# ── Public runner ──────────────────────────────────────────────────────────

def run() -> pd.DataFrame:
    plots_dir = os.path.join(MODEL_DIR, "plots")

    print("\n[Phase 4] Model Building — Isolation Forest")
    print("-" * 40)

    df, X, y = _load_features(FEATURE_FILE)

    model   = _train_model(X)
    df      = _score_and_predict(model, X, df)

    _evaluate(df)
    _save_plots(df, model, X, plots_dir)
    _save_model(model, MODEL_FILE)

    # Save scored dataset for downstream phases
    scored_path = FEATURE_FILE.replace("features.csv", "scored_logs.csv")
    df.to_csv(scored_path, index=False)
    print(f"  [✔] Scored dataset saved → {scored_path}")

    return df


# ── Direct execution ───────────────────────────────────────────────────────

if __name__ == "__main__":
    run()