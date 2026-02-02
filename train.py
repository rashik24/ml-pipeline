# ======================================================================
# Nested Consensus CV:
#   A) LR + L1 (inner GridSearch over C & threshold)  → eval LR
#   B) LR + SFS (inner CV selects k features)         → eval LR
#   C) ML (RF/GB/XGB/SVM) full features (inner HP tuning) → eval
# Per-fold Youden threshold; SMOTE in-pipeline; optional VIF pruning.
# macOS-safe threading (n_jobs=1 inside estimators/search).
# ======================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter
from joblib import parallel_backend

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Optional: VIF pruning (set vif_threshold to a number to enable)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

RANDOM_STATE = 0

# -----------------------------
# 0) Load data & feature engineering (edit paths/columns as needed)
# -----------------------------
df = pd.read_csv("data/dataset.csv")


df['IQR_over_Median'] = df['IQR'] / df['Median']
df['P90_over_Median'] = df['P90'] / df['Median']
df['Q3_over_Q1']      = df['Q3'] / df['Q1']
df['Excess_Supply']   = df['Avg_Weekly_Supply'] - (df['Average_monthly_served'] * 32 / 12)
df['PPIN']            = df['Avg_Weekly_Supply'] / df['Average_monthly_served']
df['WTS_binary']      = np.where(df[['C_pantry','C_Other','Partners_webpage']].max(axis=1)==1, 1, 0)

X_vars = [
    "P90_over_Median","Q3_over_Q1","PPIN","Avg_Weekly_Supply","REFRIG",
    "FROZEN","Is_multi_FB","PRODUCE","IQR_over_Median","Excess_Supply",
    "Monthly_Open_Hours","Average_monthly_served","Is_Disaster_Relief",
    "Is_Non_Church","Is_TEFAP","Is_CSFP","Is_Other","Is_Meal_Services",
    "Is_White","Is_Multi","social_media","Is_Pantries","Is_Black",
    "Is_Hispanic","Is_Elderly_Services","Is_Children_Services","Is_Shelters_Group_Homes"
]
df = df.dropna(subset=X_vars + ["WTS_binary"])
X = df[X_vars].copy()
y = df["WTS_binary"].astype(int).copy()

# -----------------------------
# 1) CV helpers
# -----------------------------
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE+1)

def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    return thr[np.argmax(tpr - fpr)]

def summarize_perf(rows):
    d = pd.DataFrame(rows)
    return d.mean()[["AUC","Accuracy","F1_yes","Recall_yes","Precision_yes"]].rename({
        "AUC":"Mean AUC","Accuracy":"Mean Accuracy","F1_yes":"Mean F1_yes",
        "Recall_yes":"Mean Recall_yes","Precision_yes":"Mean Precision_yes"
    })

# -----------------------------
# 2) Optional VIF pruning utilities
# -----------------------------
def compute_vif(X_df):
    Xc = sm.add_constant(X_df, has_constant='add')
    vif_vals = []
    for i, col in enumerate(X_df.columns, start=1):  # skip constant
        vif_vals.append((col, variance_inflation_factor(Xc.values, i)))
    return pd.DataFrame(vif_vals, columns=["feature","VIF"]).sort_values("VIF", ascending=False)

def vif_prune(X_df, thresh=5.0, max_iter=50):
    cols = list(X_df.columns)
    dropped = []
    for _ in range(max_iter):
        vif_tbl = compute_vif(X_df[cols])
        max_feat, max_vif = vif_tbl.iloc[0]["feature"], vif_tbl.iloc[0]["VIF"]
        if max_vif <= thresh:  # stop
            return cols, dropped, vif_tbl
        cols.remove(max_feat)
        dropped.append((max_feat, float(max_vif)))
    # safety stop
    return cols, dropped, compute_vif(X_df[cols])

# -----------------------------
# 3) LR + L1 with inner GridSearch over C & threshold (FIXED)
# -----------------------------
def nested_lr_with_L1_grid(vif_threshold=None):
    """
    L1-embedded selection with inner GridSearchCV over C (and threshold),
    nested in outer CV. Optional VIF prune on outer-train only.
    Returns: performance summary, stability table, best params list.
    """
    perf, picked, best_params = [], [], []

    with parallel_backend("threading"):
        for tr, te in outer_cv.split(X, y):
            Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

            # --- VIF prune on outer-training only (optional) ---
            if vif_threshold is not None:
                kept_cols, _, _ = vif_prune(Xtr, thresh=vif_threshold)
            else:
                kept_cols = list(Xtr.columns)

            # Pipeline: scale -> SMOTE -> SelectFromModel(L1) -> final LR
            l1_est = LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000)
            pipe = ImbPipeline([
                ('scale', StandardScaler()),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('sfrom', SelectFromModel(l1_est, threshold="median")),
                ('clf', LogisticRegression(max_iter=2000, solver='liblinear'))
            ])

            # Inner grid over L1 strength and selection threshold
            param_grid = {
                'sfrom__estimator__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                'sfrom__threshold':   ['median', 'mean', '1.25*median', '0.75*median']
            }

            search = GridSearchCV(
                estimator=pipe,
                param_grid=param_grid,
                scoring='roc_auc',
                cv=inner_cv,
                n_jobs=1,
                refit=True,
                verbose=0
            )

            # Fit inner search on the full VIF-pruned set (NOT pre-sliced to feats)
            search.fit(Xtr[kept_cols], ytr)

            # Best pipeline for this outer fold (already refit on full outer-train)
            best = search.best_estimator_
            best_params.append(search.best_params_)

            # Which features got selected? Use the fitted selector mask mapped to kept_cols
            selector = best.named_steps['sfrom']
            feats = pd.Index(kept_cols)[selector.get_support()].tolist()
            picked.append(feats)

            # ===== FIX: pass the SAME columns used at fit time =====
            y_prob = best.predict_proba(Xte[kept_cols])[:, 1]
            thr = youden_threshold(yte, y_prob)
            y_hat = (y_prob >= thr).astype(int)

            rep = classification_report(yte, y_hat, output_dict=True, zero_division=0)
            perf.append({
                "AUC": roc_auc_score(yte, y_prob),
                "Accuracy": accuracy_score(yte, y_hat),
                "F1_yes": rep['1']['f1-score'],
                "Recall_yes": rep['1']['recall'],
                "Precision_yes": rep['1']['precision']
            })

    # Stability across outer folds
    stab = (
        pd.Series([f for s in picked for f in s])
        .value_counts()
        .rename_axis("feature")
        .reset_index(name="count")
    )
    stab["pct_folds"] = stab["count"] / outer_cv.get_n_splits()
    return summarize_perf(perf), stab, best_params

# -----------------------------
# 4) LR + SFS (inner CV selects k features)
# -----------------------------
def nested_lr_with_SFS(k=10, vif_threshold=None):
    perf, picked = [], []
    with parallel_backend("threading"):
        for tr, te in outer_cv.split(X, y):
            Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]

            if vif_threshold is not None:
                kept_cols, _, _ = vif_prune(Xtr, thresh=vif_threshold)
            else:
                kept_cols = list(Xtr.columns)

            base_lr = LogisticRegression(max_iter=2000, solver='liblinear')
            sfs = SequentialFeatureSelector(
                base_lr, n_features_to_select=min(k, len(kept_cols)),
                direction='forward', scoring='roc_auc', cv=inner_cv, n_jobs=1
            )
            sel_pipe = ImbPipeline([
                ('scale', StandardScaler()),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('sfs', sfs)
            ])
            sel_pipe.fit(Xtr[kept_cols], ytr)
            mask = sel_pipe.named_steps['sfs'].get_support()
            feats = pd.Index(kept_cols)[mask].tolist()
            picked.append(feats)

            lr_pipe = ImbPipeline([
                ('scale', StandardScaler()),
                ('smote', SMOTE(random_state=RANDOM_STATE)),
                ('clf', LogisticRegression(max_iter=2000, solver='liblinear'))
            ])
            lr_pipe.fit(Xtr[feats], ytr)
            y_prob = lr_pipe.predict_proba(Xte[feats])[:, 1]
            thr = youden_threshold(yte, y_prob)
            y_hat = (y_prob >= thr).astype(int)

            rep = classification_report(yte, y_hat, output_dict=True, zero_division=0)
            perf.append({
                "AUC": roc_auc_score(yte, y_prob),
                "Accuracy": accuracy_score(yte, y_hat),
                "F1_yes": rep['1']['f1-score'],
                "Recall_yes": rep['1']['recall'],
                "Precision_yes": rep['1']['precision']
            })

    stab = (
        pd.Series([f for s in picked for f in s])
        .value_counts()
        .rename_axis("feature")
        .reset_index(name="count")
    )
    stab["pct_folds"] = stab["count"] / outer_cv.get_n_splits()
    return summarize_perf(perf), stab

# -----------------------------
# 5) Final LR refit & interpretation (coeffs / OR)
# -----------------------------
from sklearn.pipeline import Pipeline

def fit_and_interpret(X, y, features, label="LR"):
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(max_iter=2000, solver='liblinear'))
    ])
    pipe.fit(X[features], y)
    coef = pd.Series(pipe.named_steps['clf'].coef_[0], index=features).sort_values(key=abs, ascending=False)
    odds = np.exp(coef)
    print(f"\n=== Final {label} Logistic Regression ===")
    print("Features:", features)
    print("\nCoefficients (std):\n", coef.round(3))
    print("\nOdds Ratios:\n", odds.round(3))
    return coef, odds

# -----------------------------
# 6) ML models: inner HP tuning + outer eval (all features)
# -----------------------------
param_spaces = {
    "Random Forest": {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 3, 5, 10],
        "clf__min_samples_split": [2, 5, 10]
    },
    "Gradient Boosting": {
        "clf__n_estimators": [150, 250, 400],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth": [2, 3, 4]
    },
    "XGBoost": {
        "clf__n_estimators": [200, 400, 600],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth": [2, 3, 4],
        "clf__subsample": [0.7, 1.0],
        "clf__colsample_bytree": [0.7, 1.0]
    },
    "SVM (RBF)": {
        "clf__C": [0.5, 1, 2, 4],
        "clf__gamma": ["scale", 0.1, 0.01]
    }
}

base_models = {
    "Random Forest": ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE))
    ]),
    "Gradient Boosting": ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', GradientBoostingClassifier(random_state=RANDOM_STATE))
    ]),
    "XGBoost": ImbPipeline([
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=1))
    ]),
    "SVM (RBF)": ImbPipeline([
        ('scale', StandardScaler()),
        ('smote', SMOTE(random_state=RANDOM_STATE)),
        ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE))
    ])
}

def nested_hp_tuned_model(name, pipe, space, n_iter=12):
    perfs, best_params = [], []
    with parallel_backend("threading"):
        for tr, te in outer_cv.split(X, y):
            Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
            search = RandomizedSearchCV(
                estimator=pipe, param_distributions=space, n_iter=n_iter,
                scoring='roc_auc', cv=inner_cv, random_state=RANDOM_STATE, n_jobs=1, refit=True
            )
            search.fit(Xtr, ytr)
            best = search.best_estimator_
            y_prob = best.predict_proba(Xte)[:, 1]
            thr = youden_threshold(yte, y_prob)
            y_hat = (y_prob >= thr).astype(int)
            rep = classification_report(yte, y_hat, output_dict=True, zero_division=0)
            perfs.append({
                "AUC": roc_auc_score(yte, y_prob),
                "Accuracy": accuracy_score(yte, y_hat),
                "F1_yes": rep['1']['f1-score'],
                "Recall_yes": rep['1']['recall'],
                "Precision_yes": rep['1']['precision']
            })
            best_params.append(search.best_params_)
    return summarize_perf(perfs), best_params

def consensus_params(best_params_list):
    keys = set().union(*[p.keys() for p in best_params_list])
    cons = {}
    for k in keys:
        vals = [p[k] for p in best_params_list if k in p]
        cons[k] = Counter(vals).most_common(1)[0][0]
    return cons

# -----------------------------
# 7) RUN ALL
# -----------------------------
print("\n=== LR + L1 (nested CV with inner grid on C) ===")
perf_L1_grid, stab_L1_grid, l1_best_params = nested_lr_with_L1_grid(vif_threshold=None)  # set e.g., 5.0 to enable VIF
print(perf_L1_grid.round(3))
print("\nTop stable features (L1 grid):\n", stab_L1_grid.head(20))
print("\nBest inner parameters per outer fold (L1):\n", l1_best_params)

print("\n=== LR + SFS (nested CV) ===")
perf_SFS, stab_SFS = nested_lr_with_SFS(k=10, vif_threshold=None)
print(perf_SFS.round(3))
print("\nTop stable features (SFS):\n", stab_SFS.head(20))

# Choose stable features (e.g., ≥60% of outer folds)
stable_L1 = stab_L1_grid.query("pct_folds >= 0.6")["feature"].tolist()
stable_SFS = stab_SFS.query("pct_folds >= 0.6")["feature"].tolist()

# Final LR refits for interpretation
coef_L1, or_L1   = fit_and_interpret(X, y, stable_L1,  label="L1-selected (grid)")
coef_SFS, or_SFS = fit_and_interpret(X, y, stable_SFS, label="SFS-selected")

print("\n=== ML (nested CV + HP tuning) — Mean performance ===")
ml_summaries, ml_params = {}, {}
for name, pipe in base_models.items():
    summ, params = nested_hp_tuned_model(name, pipe, param_spaces[name], n_iter=12)
    ml_summaries[name] = summ.round(3)
    ml_params[name] = params

perf_ML = pd.DataFrame(ml_summaries).T.sort_values("Mean AUC", ascending=False)
print(perf_ML)

# Pick best ML, refit on ALL data with consensus params, prep for SHAP
best_ml_name = perf_ML["Mean AUC"].idxmax()
print(f"\nSelected best ML model: {best_ml_name}")
best_consensus_params = consensus_params(ml_params[best_ml_name])
print("Consensus params:", best_consensus_params)

final_pipe = base_models[best_ml_name].set_params(**best_consensus_params)
final_pipe.fit(X, y)

try:
    import shap
    if best_ml_name in ["Random Forest", "Gradient Boosting", "XGBoost"]:
        bg = shap.sample(X, 50, random_state=0)
        explainer = shap.Explainer(final_pipe.named_steps['clf'], bg)
        sv = explainer(X)
        # In a notebook you can visualize:
        # shap.summary_plot(sv, X)
        # shap.summary_plot(sv, X, plot_type="bar")
        print("\nSHAP ready for tree model (use summary_plot in your notebook).")
    else:
        print("\nBest model is SVM; SHAP KernelExplainer is slower. Consider tree model for SHAP visuals.")
except Exception as e:
    print("\nSHAP not available or failed to compute:", e)
import joblib
import json
from pathlib import Path

ART = Path("ml/artifacts")
ART.mkdir(parents=True, exist_ok=True)

# Save final model
joblib.dump(final_pipe, ART / "model.joblib")

# Save performance table
perf_ML.to_json(ART / "metrics.json")

print("Saved model and metrics.")
