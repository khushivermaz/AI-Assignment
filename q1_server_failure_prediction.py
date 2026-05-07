"""
===============================================================================
QUESTION 1: Predicting Enterprise Server Failure Across Global Data Centers
===============================================================================
Author     : Data Science & AI Assignment
Description: Predictive maintenance system for 10 global clients, each with
             5-20 data centers and 5000+ servers. Handles missing data (25%),
             predicts failure in < 5 seconds per server, auto-adapts to new
             infrastructure.
===============================================================================
"""

import numpy as np
import pandas as pd
import time
import warnings
from datetime import datetime, timedelta

# ─── Suppress sklearn version warnings ────────────────────────────────────────
warnings.filterwarnings("ignore")

# ─── Attempt to import sklearn; install if missing ────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, f1_score, precision_score,
                                  recall_score)
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARNING] scikit-learn not found. Using NumPy-only fallback mode.\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ─ DATA GENERATION  (Simulates real monitoring telemetry)
# ══════════════════════════════════════════════════════════════════════════════

class ServerMonitoringDataGenerator:
    """
    Generates realistic server monitoring data for multiple clients /
    data-centers. Injects:
      • 25 % random missing values  (per constraint)
      • Correlated anomaly patterns (pre-failure signatures)
      • Multiple environment types  (cloud, on-prem, hybrid)
    """

    ENVIRONMENT_TYPES = ["cloud", "on_premise", "hybrid", "colocation"]
    SERVER_ROLES       = ["web", "database", "cache", "compute", "storage"]

    def __init__(self, n_clients: int = 10, seed: int = 42):
        self.n_clients = n_clients
        np.random.seed(seed)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _inject_failure_pattern(row: dict) -> dict:
        """Gradually degrade metrics for servers about to fail."""
        row["memory_usage_pct"]  = min(100, row["memory_usage_pct"]  + np.random.uniform(20, 40))
        row["disk_io_latency_ms"]= row["disk_io_latency_ms"] * np.random.uniform(3, 8)
        row["network_latency_ms"]= row["network_latency_ms"] * np.random.uniform(2, 5)
        row["error_rate_per_hr"] = row["error_rate_per_hr"]  + np.random.uniform(50, 200)
        row["cpu_usage_pct"]     = min(100, row["cpu_usage_pct"]      + np.random.uniform(15, 35))
        row["crash_count_24h"]   = int(np.random.uniform(3, 15))
        row["temp_celsius"]      = row["temp_celsius"] + np.random.uniform(15, 30)
        return row

    @staticmethod
    def _inject_missing(df: pd.DataFrame, missing_rate: float = 0.25) -> pd.DataFrame:
        """Randomly blank ~25 % of metric cells (MCAR pattern)."""
        metric_cols = ["memory_usage_pct", "disk_io_latency_ms", "network_latency_ms",
                       "error_rate_per_hr", "cpu_usage_pct", "temp_celsius",
                       "disk_free_pct", "swap_usage_pct", "packet_loss_pct"]
        for col in metric_cols:
            mask = np.random.random(len(df)) < missing_rate
            df.loc[mask, col] = np.nan
        return df

    # ── public ───────────────────────────────────────────────────────────────

    def generate(self, n_servers_per_client: int = 500) -> pd.DataFrame:
        records = []
        for client_id in range(1, self.n_clients + 1):
            env_type   = np.random.choice(self.ENVIRONMENT_TYPES)
            n_dc       = np.random.randint(5, 21)          # 5-20 data centers
            n_servers  = n_servers_per_client

            for dc_id in range(1, n_dc + 1):
                dc_servers = max(10, n_servers // n_dc)
                for server_id in range(dc_servers):
                    will_fail = np.random.random() < 0.15  # 15 % failure rate

                    row = {
                        "client_id"          : client_id,
                        "datacenter_id"      : f"C{client_id:02d}-DC{dc_id:02d}",
                        "server_id"          : f"SRV-{client_id:02d}-{dc_id:02d}-{server_id:04d}",
                        "environment_type"   : env_type,
                        "server_role"        : np.random.choice(self.SERVER_ROLES),
                        "server_age_days"    : np.random.randint(30, 2000),
                        # ── metrics ──
                        "memory_usage_pct"   : np.random.uniform(20, 75),
                        "disk_io_latency_ms" : np.random.uniform(1, 50),
                        "network_latency_ms" : np.random.uniform(0.5, 30),
                        "error_rate_per_hr"  : np.random.uniform(0, 20),
                        "cpu_usage_pct"      : np.random.uniform(10, 70),
                        "temp_celsius"       : np.random.uniform(35, 60),
                        "disk_free_pct"      : np.random.uniform(15, 90),
                        "swap_usage_pct"     : np.random.uniform(0, 40),
                        "packet_loss_pct"    : np.random.uniform(0, 2),
                        "crash_count_24h"    : int(np.random.poisson(0.3)),
                        "maintenance_days_ago": np.random.randint(0, 365),
                        "will_fail_24h"      : int(will_fail),
                    }

                    if will_fail:
                        row = self._inject_failure_pattern(row)

                    records.append(row)

        df = pd.DataFrame(records)
        df = self._inject_missing(df)          # inject 25 % missing
        print(f"[DataGen] Generated {len(df):,} server records across {self.n_clients} clients.")
        print(f"          Failure rate : {df['will_fail_24h'].mean():.1%}")
        print(f"          Missing data : ~{df.isnull().mean().mean():.1%} per metric column\n")
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ─ FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    Derives composite risk indicators from raw telemetry.
    All transformations are deterministic and vectorised.
    """

    @staticmethod
    def build(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # ── composite scores (handle NaN safely) ──────────────────────────
        out["memory_pressure_score"] = (
            out["memory_usage_pct"].fillna(50) / 100 +
            out["swap_usage_pct"].fillna(0)   / 100
        ) / 2

        out["io_stress_score"] = (
            np.log1p(out["disk_io_latency_ms"].fillna(out["disk_io_latency_ms"].median())) *
            (1 + out["packet_loss_pct"].fillna(0) / 100)
        )

        out["error_severity"] = (
            np.log1p(out["error_rate_per_hr"].fillna(0)) +
            out["crash_count_24h"].fillna(0) * 2
        )

        out["thermal_risk"] = np.clip(
            (out["temp_celsius"].fillna(50) - 40) / 40, 0, 1
        )

        out["age_risk"] = np.clip(
            out["server_age_days"].fillna(365) / 2000, 0, 1
        )

        out["maintenance_lag_risk"] = np.clip(
            out["maintenance_days_ago"].fillna(90) / 365, 0, 1
        )

        # ── overall composite risk ─────────────────────────────────────────
        out["composite_risk_score"] = (
            0.30 * out["memory_pressure_score"] +
            0.20 * out["io_stress_score"].clip(0, 1) +
            0.20 * (out["error_severity"] / out["error_severity"].max().clip(1)) +
            0.15 * out["thermal_risk"] +
            0.10 * out["age_risk"] +
            0.05 * out["maintenance_lag_risk"]
        )

        print(f"[FeatureEng] Built {out.shape[1] - df.shape[1]} engineered features.")
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ─ MODEL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

NUMERIC_FEATURES = [
    "memory_usage_pct", "disk_io_latency_ms", "network_latency_ms",
    "error_rate_per_hr", "cpu_usage_pct", "temp_celsius", "disk_free_pct",
    "swap_usage_pct", "packet_loss_pct", "crash_count_24h",
    "maintenance_days_ago", "server_age_days",
    "memory_pressure_score", "io_stress_score", "error_severity",
    "thermal_risk", "age_risk", "maintenance_lag_risk", "composite_risk_score",
]

TARGET = "will_fail_24h"


class ServerFailurePredictor:
    """
    End-to-end predictor that:
      • Imputes missing values (KNN imputation)
      • Scales features
      • Trains Random Forest + Gradient Boosting ensemble
      • Predicts failure probability in < 5 s per server
      • Auto-adapts to new environments via environment-type encoding
    """

    def __init__(self):
        self.pipeline    = None
        self.is_fitted   = False
        self.feature_names = NUMERIC_FEATURES
        self._le         = {}

    # ── private ──────────────────────────────────────────────────────────────

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        out = df.copy()
        for col in ["environment_type", "server_role"]:
            if col not in out.columns:
                continue
            if fit:
                le = LabelEncoder()
                out[col + "_enc"] = le.fit_transform(out[col].fillna("unknown"))
                self._le[col] = le
            else:
                le = self._le.get(col)
                if le:
                    vals = out[col].fillna("unknown")
                    # handle unseen labels → map to 0
                    vals = vals.apply(
                        lambda v: v if v in le.classes_ else le.classes_[0]
                    )
                    out[col + "_enc"] = le.transform(vals)
        return out

    def _get_X(self, df: pd.DataFrame) -> np.ndarray:
        extra = [c for c in ["environment_type_enc", "server_role_enc"] if c in df.columns]
        cols  = self.feature_names + extra
        cols  = [c for c in cols if c in df.columns]
        return df[cols].values

    # ── public ───────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "ServerFailurePredictor":
        df = self._encode_categoricals(df, fit=True)
        X  = self._get_X(df)
        y  = df[TARGET].values

        if SKLEARN_OK:
            self.pipeline = Pipeline([
                ("imputer", KNNImputer(n_neighbors=5)),
                ("scaler",  StandardScaler()),
                ("clf",     GradientBoostingClassifier(
                    n_estimators=150, max_depth=5,
                    learning_rate=0.08, subsample=0.85,
                    random_state=42
                )),
            ])
            self.pipeline.fit(X, y)
        else:
            # NumPy-only fallback: threshold on composite_risk_score
            self._threshold = 0.45

        self.is_fitted = True
        print("[Model] Training complete.")
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df = self._encode_categoricals(df, fit=False)
        X  = self._get_X(df)
        if SKLEARN_OK and self.pipeline:
            return self.pipeline.predict_proba(X)[:, 1]
        # fallback
        return df["composite_risk_score"].fillna(0).values

    def predict(self, df: pd.DataFrame, threshold: float = 0.40) -> np.ndarray:
        return (self.predict_proba(df) >= threshold).astype(int)

    def evaluate(self, df: pd.DataFrame) -> dict:
        y_true = df[TARGET].values
        y_prob  = self.predict_proba(df)
        y_pred  = self.predict(df)

        metrics = {
            "f1_score"       : f1_score(y_true, y_pred)       if SKLEARN_OK else None,
            "precision"      : precision_score(y_true, y_pred) if SKLEARN_OK else None,
            "recall"         : recall_score(y_true, y_pred)    if SKLEARN_OK else None,
            "roc_auc"        : roc_auc_score(y_true, y_prob)   if SKLEARN_OK else None,
            "failure_detected": int((y_pred * y_true).sum()),
            "total_failures" : int(y_true.sum()),
        }
        if metrics["total_failures"]:
            metrics["downtime_reduction_pct"] = (
                metrics["failure_detected"] / metrics["total_failures"]
            ) * 100

        return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ─ ANOMALY DETECTION (Unsupervised layer)
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """Isolation Forest for detecting novel failure patterns not in training."""

    def __init__(self, contamination: float = 0.12):
        self.contamination = contamination
        self.model = None

    def fit(self, df: pd.DataFrame):
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        X = df[cols].fillna(df[cols].median())
        if SKLEARN_OK:
            self.model = IsolationForest(
                contamination=self.contamination, n_estimators=100, random_state=42
            )
            self.model.fit(X)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        X = df[cols].fillna(df[cols].median())
        if SKLEARN_OK and self.model:
            raw = self.model.decision_function(X)   # higher = more normal
            return 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return df["composite_risk_score"].fillna(0).values


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ─ ALERTING & PRIORITISATION
# ══════════════════════════════════════════════════════════════════════════════

class AlertEngine:
    """
    Merges supervised failure probability with unsupervised anomaly score
    and emits a priority-ranked alert list.
    """

    CRITICAL = 0.70
    HIGH     = 0.50
    MEDIUM   = 0.35

    @staticmethod
    def classify(prob: float) -> str:
        if prob >= AlertEngine.CRITICAL: return "CRITICAL"
        if prob >= AlertEngine.HIGH:     return "HIGH"
        if prob >= AlertEngine.MEDIUM:   return "MEDIUM"
        return "LOW"

    def generate_alerts(
        self,
        df: pd.DataFrame,
        fail_probs: np.ndarray,
        anomaly_scores: np.ndarray
    ) -> pd.DataFrame:

        df = df.copy()
        df["failure_probability"] = fail_probs
        df["anomaly_score"]       = anomaly_scores
        df["combined_risk"]       = 0.65 * fail_probs + 0.35 * anomaly_scores
        df["alert_level"]         = df["combined_risk"].apply(self.classify)
        df["alert_time"]          = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        alerts = df[df["alert_level"] != "LOW"].copy()
        alerts = alerts.sort_values("combined_risk", ascending=False)

        return alerts[["server_id", "datacenter_id", "client_id",
                        "alert_level", "failure_probability",
                        "anomaly_score", "combined_risk", "alert_time"]]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 ─ MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_server_failure_system():
    print("=" * 70)
    print("  SERVER FAILURE PREDICTION SYSTEM  —  Question 1")
    print("=" * 70)
    print()

    # 1. Generate data
    gen  = ServerMonitoringDataGenerator(n_clients=10)
    raw  = gen.generate(n_servers_per_client=300)   # 300 × n_dc per client

    # 2. Feature engineering
    fe   = FeatureEngineer()
    data = fe.build(raw)

    # 3. Train / test split
    train_df, test_df = train_test_split(data, test_size=0.20, random_state=42,
                                          stratify=data[TARGET]) if SKLEARN_OK \
                        else (data.iloc[:int(len(data)*0.8)], data.iloc[int(len(data)*0.8):])

    print(f"[Split] Train: {len(train_df):,} | Test: {len(test_df):,}\n")

    # 4. Train supervised model
    predictor = ServerFailurePredictor()
    predictor.fit(train_df)

    # 5. Train anomaly detector
    anomaly = AnomalyDetector()
    anomaly.fit(train_df)

    # 6. Evaluate on test set
    print("\n── Evaluation on Test Set ──────────────────────────────────────────")
    metrics = predictor.evaluate(test_df)
    for k, v in metrics.items():
        if v is not None:
            label = f"{k:30s}"
            val   = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(f"  {label}: {val}")

    downtime_r = metrics.get("downtime_reduction_pct", 0)
    target_met = "✅ TARGET MET" if downtime_r >= 40 else "⚠️  BELOW TARGET"
    print(f"\n  Downtime Reduction    : {downtime_r:.1f}%   {target_met}")

    # 7. Latency check
    print("\n── Latency Check (per-server prediction time) ──────────────────────")
    sample = test_df.sample(min(100, len(test_df)), random_state=1)
    t0 = time.time()
    _ = predictor.predict_proba(sample)
    elapsed = (time.time() - t0) / len(sample) * 1000
    latency_ok = "✅ PASS" if elapsed < 5000 else "❌ FAIL"
    print(f"  Avg prediction time   : {elapsed:.2f} ms/server   {latency_ok}")

    # 8. Generate alerts
    print("\n── Alert Summary ────────────────────────────────────────────────────")
    fail_probs     = predictor.predict_proba(test_df)
    anomaly_scores = anomaly.score(test_df)
    alert_engine   = AlertEngine()
    alerts         = alert_engine.generate_alerts(test_df, fail_probs, anomaly_scores)

    for level in ["CRITICAL", "HIGH", "MEDIUM"]:
        count = (alerts["alert_level"] == level).sum()
        print(f"  {level:10s}: {count:5,} servers")

    print(f"\n  Top 5 at-risk servers:")
    top5 = alerts.head(5)[["server_id", "datacenter_id", "alert_level",
                             "failure_probability", "combined_risk"]]
    print(top5.to_string(index=False))

    # 9. Cross-client adaptability check
    print("\n── Cross-Client Adaptability ────────────────────────────────────────")
    for cid in sorted(data["client_id"].unique())[:5]:
        client_data = test_df[test_df["client_id"] == cid]
        if len(client_data) < 10:
            continue
        f1 = f1_score(client_data[TARGET], predictor.predict(client_data)) if SKLEARN_OK else "N/A"
        print(f"  Client {cid:02d}  |  Servers: {len(client_data):4d}  |  F1: {f1:.4f}" if SKLEARN_OK
              else f"  Client {cid:02d}  |  Servers: {len(client_data):4d}")

    print("\n[✓] Question 1 — Server Failure Prediction System complete.")
    return {
        "metrics"  : metrics,
        "alerts"   : alerts,
        "predictor": predictor,
        "anomaly"  : anomaly,
    }


if __name__ == "__main__":
    run_server_failure_system()
