from pathlib import Path
import json
import math
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch import nn


class PSCNet(nn.Module):
    def __init__(self, in_dim, h=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
        )
        self.mu = nn.Linear(h, 2)
        self.logvar = nn.Linear(h, 2)

    def forward(self, x):
        h = self.body(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        logvar = torch.clamp(logvar, -5.0, 5.0)
        return mu, logvar


class PredictRequest(BaseModel):
    bandgap_eV: Optional[float] = None
    JV_default_Jsc_mAcm2: Optional[float] = None
    JV_default_Voc_V: Optional[float] = None
    JV_default_FF: Optional[float] = None
    Perovskite_deposition_annealing_temperature_C: Optional[float] = None
    Perovskite_deposition_annealing_time_s: Optional[float] = None
    Cell_architecture: Optional[str] = None
    Perovskite_composition_short_form: Optional[str] = None
    ETL_material: Optional[str] = None
    HTL_material: Optional[str] = None
    Perovskite_deposition_method: Optional[str] = None
    Additive_type: Optional[str] = None
    Encapsulation: Optional[str] = None

class FullPredictRequest(BaseModel):
    pass

class ScoreToFeaturesRequest(BaseModel):
    target_score: float
    initial_features: Optional[dict] = None


BASE_DIR = Path(__file__).resolve().parent

USE_ROBUST_MODEL = False

if USE_ROBUST_MODEL and (BASE_DIR / "model_robust_subset.pt").exists():
    print("Using robust model trained on form features subset")
    artifacts_path = BASE_DIR / "artifacts_robust_subset.json"
    preprocess_path = BASE_DIR / "preprocess_robust_subset.joblib"
    model_path = BASE_DIR / "model_robust_subset.pt"
else:
    print("Using original full-feature model")
    artifacts_path = BASE_DIR / "artifacts.json"
    preprocess_path = BASE_DIR / "preprocess.joblib"
    model_path = BASE_DIR / "model_best.pt"

with open(artifacts_path) as f:
    artifacts = json.load(f)

num_cols = artifacts["num_cols"]
cat_cols = artifacts["cat_cols"]
p_min = artifacts["p_min"]
p_max = artifacts["p_max"]
s_min = artifacts["s_min"]
s_max = artifacts["s_max"]
sig_ref_p = artifacts["sig_ref_p"]
sig_ref_s = artifacts["sig_ref_s"]
input_dim = artifacts["input_dim"]

preprocess = joblib.load(preprocess_path)
model = PSCNet(input_dim)
state = torch.load(model_path, map_location="cpu")
model.load_state_dict(state)
model.eval()

numeric_alias = {
    "JV_default_Jsc_mAcm2": "JV_default_Jsc",
    "JV_default_Voc_V": "JV_default_Voc",
    "JV_default_FF": "JV_default_FF",
}

categorical_alias = {
    "Cell_architecture": "Cell_architecture",
    "Perovskite_composition_short_form": "Perovskite_composition_short_form",
    "ETL_material": "ETL_stack_sequence",
    "HTL_material": "HTL_stack_sequence",
    "Perovskite_deposition_method": "Perovskite_deposition_procedure",
    "Additive_type": "Perovskite_additives_compounds",
    "Encapsulation": "Encapsulation",
    "bandgap_eV": "Perovskite_band_gap",
    "Perovskite_deposition_annealing_temperature_C": "Perovskite_deposition_thermal_annealing_temperature",
    "Perovskite_deposition_annealing_time_s": "Perovskite_deposition_thermal_annealing_time",
}

app = FastAPI()

# Get frontend URL from environment variable or use defaults
FRONTEND_URL = os.getenv("FRONTEND_URL", "")
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
]

# Add production frontend URL if provided
if FRONTEND_URL:
    CORS_ORIGINS.append(FRONTEND_URL)

# CORS configuration - use regex for wildcard matching
# FastAPI doesn't support wildcards in allow_origins, so we use allow_origin_regex
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.(vercel\.app|netlify\.app|railway\.app|onrender\.com)$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_row(payload: PredictRequest) -> pd.DataFrame:
    data = payload.model_dump(exclude_none=True)
    row = {col: np.nan for col in num_cols}
    row.update({col: "" for col in cat_cols})
    for alias, target in numeric_alias.items():
        if alias in data and target in row:
            value = float(data[alias])
            if alias == "JV_default_FF":
                value /= 100.0
            row[target] = value
    for alias, target in categorical_alias.items():
        if alias in data and target in row:
            row[target] = str(data[alias])
    return pd.DataFrame([row])

def build_row_full(features: dict) -> pd.DataFrame:
    row = {col: np.nan for col in num_cols}
    row.update({col: "" for col in cat_cols})
    for key, value in features.items():
        if key in num_cols:
            try:
                row[key] = float(value) if value != "" and value is not None else np.nan
            except:
                row[key] = np.nan
        elif key in cat_cols:
            row[key] = str(value) if value != "" and value is not None else ""
    return pd.DataFrame([row])


def compute_score(mu_pce, mu_sta, sigma_pce, sigma_sta):
    p_norm_raw = (mu_pce - p_min) / (p_max - p_min + 1e-8)
    s_norm_raw = (mu_sta - s_min) / (s_max - s_min + 1e-8)
    
    if p_norm_raw < 0:
        p_norm = 0.0
        p_penalty = abs(p_norm_raw)
    else:
        p_norm = min(1.0, p_norm_raw)
        p_penalty = 0.0
    
    if s_norm_raw < 0:
        s_norm = 0.0
        s_penalty = abs(s_norm_raw)
    else:
        s_norm = min(1.0, s_norm_raw)
        s_penalty = 0.0
    
    if p_norm > 0 and p_norm < 0.05:
        p_norm_scaled = p_norm * 8.0
    elif p_norm >= 0.05:
        p_norm_scaled = 0.4 + (p_norm - 0.05) * 0.6 / 0.95
    else:
        p_norm_scaled = 0.0
    
    if s_norm > 0 and s_norm < 0.05:
        s_norm_scaled = s_norm * 8.0
    elif s_norm >= 0.05:
        s_norm_scaled = 0.4 + (s_norm - 0.05) * 0.6 / 0.95
    else:
        s_norm_scaled = 0.0
    
    q = 0.7 * min(1.0, p_norm_scaled) + 0.3 * min(1.0, s_norm_scaled)
    alpha = beta = 0.4
    u = 1.0 - alpha * (sigma_pce / (sig_ref_p + 1e-8)) - beta * (sigma_sta / (sig_ref_s + 1e-8))
    u = float(np.clip(u, 0.0, 1.0))
    
    base_score = 100.0 * q * u
    
    if p_penalty > 0 or s_penalty > 0:
        total_penalty = min(0.9, p_penalty * 0.7 + s_penalty * 0.3)
        base_score = max(0, base_score * (1.0 - total_penalty))
    
    return float(np.clip(base_score, 0.0, 100.0)), float(p_norm_raw), float(s_norm_raw)


def compute_feature_importance(x_tensor, model, preprocess, row_df, req_data, mu_pce, mu_sta):
    req_dict = req_data.model_dump(exclude_none=True)
    user_features = []
    
    if "JV_default_Jsc_mAcm2" in req_dict:
        jsc_val = float(req_dict["JV_default_Jsc_mAcm2"])
        user_features.append({"name": "Short-Circuit Current (Jsc)", "value": jsc_val, "unit": "mA/cm²", "impact": "high", "suggestion": f"Increase to {jsc_val * 1.15:.1f} mA/cm² for better efficiency" if jsc_val < 25 else "Good value"})
    
    if "JV_default_Voc_V" in req_dict:
        voc_val = float(req_dict["JV_default_Voc_V"])
        user_features.append({"name": "Open-Circuit Voltage (Voc)", "value": voc_val, "unit": "V", "impact": "high", "suggestion": f"Increase to {voc_val * 1.08:.2f} V for better efficiency" if voc_val < 1.15 else "Good value"})
    
    if "JV_default_FF" in req_dict:
        ff_val = float(req_dict["JV_default_FF"])
        user_features.append({"name": "Fill Factor", "value": ff_val, "unit": "%", "impact": "high", "suggestion": f"Increase to {min(100, ff_val * 1.05):.1f}% for better efficiency" if ff_val < 80 else "Good value"})
    
    suggestions = []
    score_target = 90.0
    
    num_provided = sum(1 for k in ["JV_default_Jsc_mAcm2", "JV_default_Voc_V", "JV_default_FF"] if k in req_dict)
    if num_provided == 3 and mu_pce < p_min * 1.2:
        suggestions.append(f"⚠️ Your JV parameters (Jsc={req_dict.get('JV_default_Jsc_mAcm2', 'N/A')}, Voc={req_dict.get('JV_default_Voc_V', 'N/A')}, FF={req_dict.get('JV_default_FF', 'N/A')}) are excellent, but the predicted efficiency ({mu_pce:.2f}%) is low. This may be because other material/processing features are at defaults. The model is conservative when information is incomplete.")
    
    if mu_pce < p_min:
        suggestions.append(f"Efficiency ({mu_pce:.2f}%) is below typical range ({p_min:.2f}%-{p_max:.2f}%). Focus on improving Jsc, Voc, and Fill Factor.")
    elif mu_pce < p_min * 1.5:
        suggestions.append(f"Efficiency ({mu_pce:.2f}%) is in the lower range. Consider optimizing material composition and processing conditions.")
    elif mu_pce < p_max * 0.8:
        pce_gap = p_max - mu_pce
        suggestions.append(f"To reach 90+ score: Need efficiency ~{p_max:.1f}% (currently {mu_pce:.2f}%, gap: {pce_gap:.2f}%). This requires optimal material composition, processing conditions, and layer thicknesses - not just JV parameters.")
    
    if mu_sta < s_min:
        suggestions.append(f"Stability ({mu_sta:.1f}h) is below typical range ({s_min:.1f}h-{s_max:.0f}h). Consider better encapsulation or material quality.")
    elif mu_sta < s_min * 5:
        suggestions.append(f"Stability ({mu_sta:.1f}h) could be improved. Better encapsulation and material quality may help.")
    elif mu_sta < s_max * 0.8:
        sta_gap = s_max - mu_sta
        suggestions.append(f"To reach 90+ score: Need stability ~{s_max:.0f}h (currently {mu_sta:.1f}h, gap: {sta_gap:.0f}h). Requires excellent encapsulation, stable materials, and proper device architecture.")
    
    for feat in user_features:
        if "Increase" in feat["suggestion"]:
            suggestions.append(feat["suggestion"])
    
    if mu_pce >= p_max * 0.9 and mu_sta >= s_max * 0.9:
        suggestions.append("🎯 Excellent! You're in the top 10% range. For 90+ score, ensure low uncertainty by providing complete material and processing details.")
    elif not suggestions:
        suggestions.append("Recipe looks good! Consider fine-tuning for even better performance.")
    
    if score_target > 0 and (mu_pce < p_max * 0.8 or mu_sta < s_max * 0.8):
        suggestions.append(f"💡 To achieve 90+ score: The model needs efficiency ≥{p_max*0.9:.1f}% AND stability ≥{s_max*0.9:.0f}h. Your excellent JV parameters alone aren't enough - the model also considers material properties, layer thicknesses, processing conditions, and device architecture that aren't in the current form. Consider using the full feature set from training data.")
    
    try:
        x_tensor.requires_grad = True
        model.eval()
        mu, _ = model(x_tensor)
        mu_flat = mu.squeeze() if mu.dim() > 1 else mu
        if len(mu_flat) >= 2:
            score_target = mu_flat[0] * 0.7 + mu_flat[1] * 0.3
        else:
            score_target = mu_flat[0] if len(mu_flat) > 0 else torch.tensor(0.0)
        score_target.backward()
        grads = x_tensor.grad.numpy().flatten()
        top_grad_features = sorted(enumerate(grads), key=lambda x: abs(x[1]), reverse=True)[:5]
        top_features = [{"name": f"Feature {i+1}", "impact": float(v)} for i, v in top_grad_features]
    except Exception:
        top_features = []
    
    return {
        "top_features": top_features,
        "suggestions": suggestions,
        "user_features": user_features,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    row = build_row(req)
    encoded = preprocess.transform(row)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    encoded_np = encoded.astype(np.float32)
    x = torch.from_numpy(encoded_np)
    with torch.no_grad():
        mu, logvar = model(x)
    mu = mu.numpy().flatten()
    logvar = logvar.numpy().flatten()
    sigma = np.sqrt(np.exp(logvar))
    mu_pce = float(mu[0])
    mu_sta_log = float(mu[1])
    sigma_pce = float(sigma[0])
    sigma_sta_log = float(sigma[1])
    mu_sta = math.exp(mu_sta_log) - 1.0
    sigma_sta = sigma_sta_log * (mu_sta + 1e-8)

    if (
        req.JV_default_Jsc_mAcm2 is not None
        and req.JV_default_Voc_V is not None
        and req.JV_default_FF is not None
    ):
        jsc = float(req.JV_default_Jsc_mAcm2)
        voc = float(req.JV_default_Voc_V)
        ff_pct = float(req.JV_default_FF)
        naive_pce = jsc * voc * ff_pct / 100.0
        if 0.0 < naive_pce < 35.0:
            mu_pce = naive_pce
            sigma_pce = max(sigma_pce, 0.5)

    if req.Encapsulation:
        enc = str(req.Encapsulation).strip().lower()
        if enc in {"yes", "y"}:
            mu_sta = max(mu_sta, 0.8 * s_max)
        elif enc in {"no", "n"}:
            mu_sta = min(mu_sta, 0.3 * s_max)

    score, p_norm_raw, s_norm_raw = compute_score(mu_pce, mu_sta, sigma_pce, sigma_sta)
    x_grad = torch.from_numpy(encoded_np.copy()).requires_grad_(True)
    importance_data = compute_feature_importance(x_grad, model, preprocess, row, req, mu_pce, mu_sta)
    return {
        "pce_pred": mu_pce,
        "t80_pred_hours": mu_sta,
        "score": score,
        "sigma_pce": sigma_pce,
        "sigma_sta_log": sigma_sta_log,
        "p_norm": p_norm_raw,
        "s_norm": s_norm_raw,
        "feature_importance": importance_data["top_features"],
        "suggestions": importance_data["suggestions"],
        "user_features": importance_data.get("user_features", []),
    }

@app.get("/features")
def get_features():
    return {
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "total_features": len(num_cols) + len(cat_cols)
    }

@app.post("/predict/full")
def predict_full(features: dict):
    row = build_row_full(features)
    encoded = preprocess.transform(row)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    encoded_np = encoded.astype(np.float32)
    x = torch.from_numpy(encoded_np)
    with torch.no_grad():
        mu, logvar = model(x)
    mu = mu.numpy().flatten()
    logvar = logvar.numpy().flatten()
    sigma = np.sqrt(np.exp(logvar))
    mu_pce = float(mu[0])
    mu_sta_log = float(mu[1])
    sigma_pce = float(sigma[0])
    sigma_sta_log = float(sigma[1])
    mu_sta = math.exp(mu_sta_log) - 1.0
    sigma_sta = sigma_sta_log * (mu_sta + 1e-8)
    score, p_norm_raw, s_norm_raw = compute_score(mu_pce, mu_sta, sigma_pce, sigma_sta)
    return {
        "pce_pred": mu_pce,
        "t80_pred_hours": mu_sta,
        "score": score,
        "sigma_pce": sigma_pce,
        "sigma_sta_log": sigma_sta_log,
    }

@app.post("/score-to-features")
def score_to_features(req: ScoreToFeaturesRequest):
    target_score = req.target_score
    initial = req.initial_features or {}
    
    row = build_row_full(initial)
    encoded = preprocess.transform(row)
    if hasattr(encoded, "toarray"):
        encoded = encoded.toarray()
    encoded_np = encoded.astype(np.float32)
    
    recommendations = []
    if target_score > 80:
        recommendations.append("For high scores (80+), aim for: Efficiency ≥18%, Stability ≥1000h")
        recommendations.append("Key features: High Jsc (≥25 mA/cm²), High Voc (≥1.15V), High FF (≥80%)")
        recommendations.append("Material: Use stable compositions, proper encapsulation, optimized layer thicknesses")
    elif target_score > 50:
        recommendations.append("For moderate scores (50-80), aim for: Efficiency ≥12%, Stability ≥100h")
        recommendations.append("Key features: Moderate Jsc (20-25 mA/cm²), Voc (1.0-1.15V), FF (75-80%)")
    else:
        recommendations.append("For lower scores, focus on improving basic JV parameters first")
        recommendations.append("Start with: Jsc ≥20 mA/cm², Voc ≥1.0V, FF ≥70%")
    
    feature_suggestions = {}
    if target_score > 70:
        feature_suggestions["JV_default_Jsc"] = 25.0
        feature_suggestions["JV_default_Voc"] = 1.15
        feature_suggestions["JV_default_FF"] = 0.80
        feature_suggestions["Encapsulation"] = "Yes"
    elif target_score > 40:
        feature_suggestions["JV_default_Jsc"] = 22.0
        feature_suggestions["JV_default_Voc"] = 1.10
        feature_suggestions["JV_default_FF"] = 0.75
    else:
        feature_suggestions["JV_default_Jsc"] = 20.0
        feature_suggestions["JV_default_Voc"] = 1.05
        feature_suggestions["JV_default_FF"] = 0.70
    
    return {
        "target_score": target_score,
        "recommended_features": feature_suggestions,
        "recommendations": recommendations
    }

