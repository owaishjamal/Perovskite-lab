import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import joblib
import streamlit as st

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("artifacts.json", "r") as f:
    a = json.load(f)

num_cols = a["num_cols"]
cat_cols = a["cat_cols"]
p_min = a["p_min"]
p_max = a["p_max"]
s_min = a["s_min"]
s_max = a["s_max"]
sig_ref_p = a["sig_ref_p"]
sig_ref_s = a["sig_ref_s"]
input_dim = a["input_dim"]

preprocess = joblib.load("preprocess.joblib")

class PSCNet(nn.Module):
    def __init__(self, in_dim, h=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU()
        )
        self.mu = nn.Linear(h, 2)
        self.logvar = nn.Linear(h, 2)
    def forward(self, x):
        h = self.body(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        logvar = torch.clamp(logvar, -5.0, 5.0)
        return mu, logvar

model = PSCNet(input_dim).to(DEVICE)
state = torch.load("model_best.pt", map_location=DEVICE)
model.load_state_dict(state)
model.eval()

def perfection_score(mu_pce, mu_sta_log, sigma_pce, sigma_sta_log, alpha=0.4, beta=0.4):
    mu_sta = np.expm1(mu_sta_log)
    sigma_sta = sigma_sta_log * (mu_sta + 1e-8)
    p = (mu_pce - p_min) / (p_max - p_min + 1e-8)
    s = (mu_sta - s_min) / (s_max - s_min + 1e-8)
    p = float(max(0.0, min(1.0, p)))
    s = float(max(0.0, min(1.0, s)))
    q = 0.7 * p + 0.3 * s
    u = 1.0 - alpha * (sigma_pce / sig_ref_p) - beta * (sigma_sta / sig_ref_s)
    u = float(max(0.0, min(1.0, u)))
    f = 1.0
    return 100.0 * q * u * f, mu_sta

def predict_and_score(df_row):
    X_enc = preprocess.transform(df_row[num_cols + cat_cols])
    if hasattr(X_enc, "toarray"):
        X_enc = X_enc.toarray()
    X_enc = X_enc.astype(np.float32)
    X_tensor = torch.from_numpy(X_enc).to(DEVICE)
    with torch.no_grad():
        mu, logvar = model(X_tensor)
        mu_np = mu.cpu().numpy()
        var_np = np.exp(logvar.cpu().numpy())
        sigma_np = np.sqrt(var_np)
    scores = []
    mu_sta_real = []
    for i in range(mu_np.shape[0]):
        s, sta_real = perfection_score(
            mu_pce=mu_np[i, 0],
            mu_sta_log=mu_np[i, 1],
            sigma_pce=sigma_np[i, 0],
            sigma_sta_log=sigma_np[i, 1]
        )
        scores.append(s)
        mu_sta_real.append(sta_real)
    return mu_np[:, 0], np.array(mu_sta_real), np.array(scores), sigma_np

st.set_page_config(page_title="Perovskite Designer", layout="centered")

st.title("Perovskite Recipe Perfection Score")

st.sidebar.header("Input features")

default_num = {c: 0.0 for c in num_cols}
default_cat = {c: "" for c in cat_cols}

num_vals = {}
cat_vals = {}

for c in num_cols:
    num_vals[c] = st.sidebar.number_input(c, value=float(default_num[c]))

for c in cat_cols:
    cat_vals[c] = st.sidebar.text_input(c, value=default_cat[c])

if st.sidebar.button("Compute score"):
    data = {}
    for c in num_cols:
        data[c] = [num_vals[c]]
    for c in cat_cols:
        data[c] = [cat_vals[c]]
    df_input = pd.DataFrame(data)
    mu_pce, mu_sta, scores, sigma = predict_and_score(df_input)
    score = float(scores[0])
    pce_pred = float(mu_pce[0])
    sta_pred = float(mu_sta[0])
    st.subheader(f"Perfection Score: {score:.1f} / 100")
    st.metric("Predicted PCE (%)", f"{pce_pred:.2f}")
    st.metric("Predicted T80 (h)", f"{sta_pred:.0f}")
    st.write("Uncertainty (σ PCE, σ log T80):", float(sigma[0, 0]), float(sigma[0, 1]))