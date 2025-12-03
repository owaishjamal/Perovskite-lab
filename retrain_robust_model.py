import numpy as np
import pandas as pd
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

DATA_PATH = "/Users/rishuraj7581gmail.com/Downloads/Perovskite_database_content_all_data.csv"

form_features = {
    "numeric": [
        "Perovskite_band_gap",
        "JV_default_Jsc",
        "JV_default_Voc",
        "JV_default_FF",
        "Perovskite_deposition_thermal_annealing_temperature",
        "Perovskite_deposition_thermal_annealing_time",
    ],
    "categorical": [
        "Cell_architecture",
        "Perovskite_composition_short_form",
        "ETL_stack_sequence",
        "HTL_stack_sequence",
        "Perovskite_deposition_procedure",
        "Perovskite_additives_compounds",
        "Encapsulation",
    ]
}

print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)

df["PCE_target"] = df["Stabilised_performance_PCE"].fillna(df["JV_default_PCE"])
df = df.replace([np.inf, -np.inf], np.nan)

mask = df["PCE_target"].notna() & df["Stability_PCE_T80"].notna()
df_ml = df.loc[mask].copy()

for col in df_ml.columns:
    if df_ml[col].dtype == "object":
        df_ml[col] = df_ml[col].astype(str)
        df_ml[col] = df_ml[col].replace(["nan", "NaN", "NONE", "None", "unknown", "Unknown"], np.nan, regex=False)

sta_raw = df_ml["Stability_PCE_T80"].values.astype(np.float32)
sta_raw = np.where(sta_raw < 0, 0, sta_raw)
sta_cap = np.percentile(sta_raw, 99)
sta_clipped = np.clip(sta_raw, 0, sta_cap)
sta_log = np.log1p(sta_clipped)
df_ml["Sta_log"] = sta_log

print(f"Total rows: {len(df_ml)}")

available_num = [f for f in form_features["numeric"] if f in df_ml.columns]
available_cat = [f for f in form_features["categorical"] if f in df_ml.columns]

print(f"Available numeric features: {len(available_num)}/{len(form_features['numeric'])}")
print(f"Available categorical features: {len(available_cat)}/{len(form_features['categorical'])}")

X_subset = df_ml[available_num + available_cat].copy()
y = df_ml[["PCE_target", "Sta_log"]].values.astype(np.float32)

for col in available_num:
    if col in X_subset.columns:
        if X_subset[col].dtype == 'object':
            def convert_to_numeric(val):
                if pd.isna(val):
                    return np.nan
                if isinstance(val, str):
                    if ';' in val:
                        parts = val.split(';')
                        try:
                            return float(parts[0].strip())
                        except:
                            return np.nan
                    try:
                        return float(val)
                    except:
                        return np.nan
                try:
                    return float(val)
                except:
                    return np.nan
            X_subset[col] = X_subset[col].apply(convert_to_numeric)
        
        X_subset[col] = pd.to_numeric(X_subset[col], errors='coerce')

valid_mask = X_subset.notna().any(axis=1)
X_subset = X_subset.loc[valid_mask]
y = y[valid_mask]

print(f"Rows with at least one feature: {len(X_subset)}")

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_subset, y, test_size=0.3, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

num_cols_subset = [c for c in available_num if c in X_train.columns and X_train[c].dtype in ['float64', 'float32', 'int64', 'int32']]
cat_cols_subset = [c for c in available_cat if c in X_train.columns]

print(f"Using {len(num_cols_subset)} numeric and {len(cat_cols_subset)} categorical features")

num_transform = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

cat_transform = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ]
)

preprocess_subset = ColumnTransformer(
    transformers=[
        ("num", num_transform, num_cols_subset),
        ("cat", cat_transform, cat_cols_subset)
    ]
)

preprocess_subset.fit(X_train)

X_train_enc = preprocess_subset.transform(X_train)
X_val_enc = preprocess_subset.transform(X_val)
X_test_enc = preprocess_subset.transform(X_test)

if hasattr(X_train_enc, "toarray"):
    X_train_enc = X_train_enc.toarray()
    X_val_enc = X_val_enc.toarray()
    X_test_enc = X_test_enc.toarray()

X_train_enc = X_train_enc.astype(np.float32)
X_val_enc = X_val_enc.astype(np.float32)
X_test_enc = X_test_enc.astype(np.float32)

input_dim = X_train_enc.shape[1]
print(f"Encoded input dimension: {input_dim}")

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

def hetero_loss(mu, logvar, y):
    var = torch.exp(logvar)
    precision = 1.0 / (var + 1e-8)
    mse = (mu - y) ** 2
    loss = precision * mse + torch.log(var + 1e-8)
    return loss.mean()

class PerovskiteDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(PerovskiteDataset(X_train_enc, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(PerovskiteDataset(X_val_enc, y_val), batch_size=64, shuffle=False)
test_loader = DataLoader(PerovskiteDataset(X_test_enc, y_test), batch_size=64, shuffle=False)

model = PSCNet(input_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

def run_epoch(loader, train=True):
    if train:
        model.train()
    else:
        model.eval()
    total = 0.0
    n = 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            mu, logvar = model(xb)
            loss = hetero_loss(mu, logvar, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
    return total / n

def eval_metrics(loader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            mu, _ = model(xb)
            preds.append(mu.cpu().numpy())
            trues.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    e_pce = preds[:, 0] - trues[:, 0]
    mae_pce = float(np.mean(np.abs(e_pce)))
    rmse_pce = math.sqrt(float(np.mean(e_pce ** 2)))
    
    t_true = np.expm1(trues[:, 1])
    t_pred = np.expm1(preds[:, 1])
    e_sta = t_pred - t_true
    mae_sta = float(np.mean(np.abs(e_sta)))
    rmse_sta = math.sqrt(float(np.mean(e_sta ** 2)))
    
    return {
        "pce_mae": mae_pce,
        "pce_rmse": rmse_pce,
        "sta_mae": mae_sta,
        "sta_rmse": rmse_sta
    }

best_val_loss = float('inf')
patience_counter = 0
max_patience = 30
n_epochs = 200

print("\nTraining model on subset of features...")
for epoch in range(n_epochs):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    scheduler.step(val_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "backend/model_robust_subset.pt")
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print("\nLoading best model for evaluation...")
model.load_state_dict(torch.load("backend/model_robust_subset.pt"))

test_metrics = eval_metrics(test_loader)
print("\nTest Metrics:")
print(f"  PCE - MAE: {test_metrics['pce_mae']:.3f}%, RMSE: {test_metrics['pce_rmse']:.3f}%")
print(f"  T80 - MAE: {test_metrics['sta_mae']:.1f}h, RMSE: {test_metrics['sta_rmse']:.1f}h")

pce_train = y_train[:, 0]
sta_log_train = y_train[:, 1]
sta_train = np.expm1(sta_log_train)

p_min = float(np.percentile(pce_train, 5))
p_max = float(np.percentile(pce_train, 95))
s_min = float(np.percentile(sta_train, 5))
s_max = float(np.percentile(sta_train, 95))

sig_ref_p = float(np.median(np.abs(pce_train - np.median(pce_train))))
sig_ref_s = float(np.median(np.abs(sta_train - np.median(sta_train))))
if sig_ref_p <= 0:
    sig_ref_p = 1.0
if sig_ref_s <= 0:
    sig_ref_s = 1.0

artifacts_subset = {
    "num_cols": num_cols_subset,
    "cat_cols": cat_cols_subset,
    "p_min": p_min,
    "p_max": p_max,
    "s_min": s_min,
    "s_max": s_max,
    "sig_ref_p": sig_ref_p,
    "sig_ref_s": sig_ref_s,
    "input_dim": input_dim
}

joblib.dump(preprocess_subset, "backend/preprocess_robust_subset.joblib")
with open("backend/artifacts_robust_subset.json", "w") as f:
    json.dump(artifacts_subset, f, indent=2)

print("\n✅ Model saved!")
print(f"  - Model: backend/model_robust_subset.pt")
print(f"  - Preprocessor: backend/preprocess_robust_subset.joblib")
print(f"  - Artifacts: backend/artifacts_robust_subset.json")
print(f"\nThis model is trained specifically on the {len(available_num + available_cat)} features")
print("available in your form, so it should perform better with limited inputs!")

