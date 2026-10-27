# Preprocessing, training, evaluation, and artifact persistence.
from pathlib import Path
import json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier

ARTIFACT_DIR = Path("artifacts")
TARGET = "Revenue"

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)], remainder="drop")

def model_candidates():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=250, class_weight="balanced", min_samples_leaf=2, n_jobs=-1, random_state=42),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=.08, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=.06, max_depth=2, random_state=42),
    }

def _scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1/(1+np.exp(-np.clip(raw, -30, 30)))
    return model.predict(X).astype(float)

def train_all(df: pd.DataFrame, save: bool = True) -> tuple[Pipeline, dict, pd.DataFrame]:
    if TARGET not in df: raise ValueError(f"Missing target column {TARGET}")
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    # Use a validation split for a simple threshold choice; keep test set for final evaluation.
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=.2, random_state=43, stratify=y_train)
    results, fitted = [], {}
    for name, estimator in model_candidates().items():
        try:
            pipe = Pipeline([("preprocess", build_preprocessor(X)), ("model", estimator)])
            pipe.fit(X_fit, y_fit)
            val_probs = _scores(pipe, X_val)
            thresholds = np.linspace(.1, .9, 33)
            f1s = [f1_score(y_val, val_probs >= t, zero_division=0) for t in thresholds]
            threshold = float(thresholds[int(np.argmax(f1s))])
            probs = _scores(pipe, X_test)
            pred = probs >= threshold
            row = {"model": name, "threshold": threshold, "validation_pr_auc": float(average_precision_score(y_val, val_probs)),
                   "accuracy": float(accuracy_score(y_test, pred)),
                   "precision": float(precision_score(y_test, pred, zero_division=0)),
                   "recall": float(recall_score(y_test, pred, zero_division=0)),
                   "f1": float(f1_score(y_test, pred, zero_division=0)),
                   "roc_auc": float(roc_auc_score(y_test, probs)),
                   "pr_auc": float(average_precision_score(y_test, probs)),
                   "tn": int(confusion_matrix(y_test, pred, labels=[0,1])[0,0]),
                   "fp": int(confusion_matrix(y_test, pred, labels=[0,1])[0,1]),
                   "fn": int(confusion_matrix(y_test, pred, labels=[0,1])[1,0]),
                   "tp": int(confusion_matrix(y_test, pred, labels=[0,1])[1,1])}
            results.append(row); fitted[name] = (pipe, row)
        except Exception as exc:
            warnings.warn(f"Skipping {name}: {exc}")
    if not results: raise RuntimeError("No model trained successfully")
    # PR-AUC is the primary ranking metric for the usually imbalanced purchase target.
    best_row = max(results, key=lambda r: r["validation_pr_auc"])
    best_model = fitted[best_row["model"]][0]
    # Refit the selected model on all training rows after model selection.
    best_estimator = model_candidates()[best_row["model"]]
    best_model = Pipeline([("preprocess", build_preprocessor(X)), ("model", best_estimator)])
    best_model.fit(X_train, y_train)
    metrics = {"best_model": best_row["model"], "selection_metric": "PR-AUC", "test_metrics": best_row,
               "all_models": sorted(results, key=lambda r: r["pr_auc"], reverse=True),
               "dataset_rows": int(len(df)), "positive_rate": float(y.mean()), "synthetic_demo_data": bool(df.attrs.get("synthetic", False))}
    if save:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, ARTIFACT_DIR / "model.joblib")
        (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
        pd.DataFrame(results).sort_values("pr_auc", ascending=False).to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
        _save_importance(best_model, X.columns)
        write_model_card(metrics)
    return best_model, metrics, pd.DataFrame(results).sort_values("pr_auc", ascending=False)

def _save_importance(pipe, columns):
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"): return
    try:
        names = pipe.named_steps["preprocess"].get_feature_names_out()
        imp = pd.DataFrame({"feature": names, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        imp.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    except Exception: pass

def write_model_card(metrics):
    row = metrics["test_metrics"]
    content = f'''# Model Card: E-Commerce Purchase Intent\n\n## Intended use\nEducational prediction of whether a historical online shopping session resulted in a purchase. Not intended for high-stakes or fully automated decisions.\n\n## Data\nRows: {metrics['dataset_rows']}; observed positive class rate: {metrics['positive_rate']:.3f}. Synthetic demo data: {metrics['synthetic_demo_data']}.\n\n## Selected model\n{metrics['best_model']} selected by test-set PR-AUC in this educational baseline. For rigorous model selection, compare candidates on validation/cross-validation and use the test set only once.\n\n## Metrics (holdout baseline)\n- PR-AUC: {row['pr_auc']:.4f}\n- ROC-AUC: {row['roc_auc']:.4f}\n- Precision: {row['precision']:.4f}\n- Recall: {row['recall']:.4f}\n- F1: {row['f1']:.4f}\n- Threshold: {row['threshold']:.2f}\n\n## Limitations and risks\nHistorical data may not represent current customers. Thresholds should reflect intervention costs and conversion value. PageValues may only be known late in a session and can leak information for early-session scoring. Feature importance is not causation. Check performance across relevant traffic/device/visitor groups before real use.\n'''
    (ARTIFACT_DIR / "model_card.md").write_text(content, encoding="utf-8")

def load_model(path="artifacts/model.joblib"):
    p = Path(path)
    if not p.exists(): raise FileNotFoundError("Model not found. Run `python scripts/train.py` first.")
    return joblib.load(p)
