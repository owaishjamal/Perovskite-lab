# Perovskite Lab - Complete Project Report

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [System Architecture](#system-architecture)
4. [Machine Learning Model](#machine-learning-model)
5. [Backend API](#backend-api)
6. [Frontend Application](#frontend-application)
7. [Features in Detail](#features-in-detail)
8. [Code Structure](#code-structure)
9. [Data Flow](#data-flow)
10. [Technical Implementation](#technical-implementation)
11. [File-by-File Breakdown](#file-by-file-breakdown)
12. [API Endpoints](#api-endpoints)
13. [UI Components](#ui-components)
14. [Scoring Algorithm](#scoring-algorithm)
15. [Deployment](#deployment)

---

## Executive Summary

**Project Name:** Perovskite Lab

**Purpose:** A web-based application that uses deep learning to predict the performance of perovskite solar cells based on material properties, device architecture, and processing conditions. The system provides a "Perfection Score" (0-100) that combines efficiency, stability, and prediction uncertainty.

**Technology Stack:**
- **Frontend:** React 18 + TypeScript + Vite
- **Backend:** FastAPI (Python)
- **ML Framework:** PyTorch
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Server:** Uvicorn (ASGI)

**Key Features:**
1. Quick Predict (13 key features)
2. Full Predict (105 features)
3. Score-to-Features (reverse engineering)
4. Real-time predictions with uncertainty quantification
5. Feature importance analysis
6. Improvement suggestions

---

## Project Overview

### Problem Statement
Designing high-performance perovskite solar cells requires extensive experimentation. Researchers need to:
- Screen hundreds of material combinations
- Optimize device architectures
- Balance efficiency and stability
- Understand which features matter most

### Solution
A machine learning-powered web application that:
- Predicts PCE (Power Conversion Efficiency) and T80 stability
- Provides uncertainty estimates for each prediction
- Calculates a composite "Perfection Score"
- Offers actionable improvement suggestions
- Supports reverse engineering (target score → features)

### Dataset
- **Training Data:** 1,570 curated perovskite solar cell devices
- **Features:** 105 total (25 numeric + 80 categorical)
- **Outputs:** PCE (%), T80 stability (hours), uncertainty estimates

---

## System Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Web Browser   │
│   (React App)   │
└────────┬────────┘
         │ HTTP/REST API
         │ (Port 5173)
         ▼
┌─────────────────┐
│  FastAPI Server │
│   (Port 8000)   │
└────────┬────────┘
         │
         ├───► Preprocessing Pipeline
         │     (scikit-learn)
         │
         ├───► PSCNet Model
         │     (PyTorch)
         │
         └───► Scoring Algorithm
               (Custom)
```

### Component Breakdown

1. **Frontend (React + TypeScript)**
   - User interface
   - Form handling
   - API communication
   - State management
   - Theme switching (dark/light)

2. **Backend (FastAPI)**
   - REST API endpoints
   - Request validation (Pydantic)
   - Model inference
   - Preprocessing
   - CORS handling

3. **ML Model (PyTorch)**
   - PSCNet neural network
   - Trained weights (model_best.pt)
   - Preprocessing pipeline (preprocess.joblib)
   - Model metadata (artifacts.json)

---

## Machine Learning Model

### Model Architecture: PSCNet

**Type:** Heteroscedastic Neural Network

**Architecture:**
```
Input Layer (Variable dimensions, typically ~2952 after encoding)
    ↓
Hidden Layer 1: Linear(2952 → 256) + ReLU
    ↓
Hidden Layer 2: Linear(256 → 256) + ReLU
    ↓
Hidden Layer 3: Linear(256 → 256) + ReLU
    ↓
    ├───► Mean Head: Linear(256 → 2)
    │     Outputs: [μ_PCE, μ_log(T80)]
    │
    └───► Uncertainty Head: Linear(256 → 2)
          Outputs: [log(σ²_PCE), log(σ²_log(T80))]
```

**Key Characteristics:**
- **Heteroscedastic Design:** Predicts both mean and variance for each output
- **Shared Representation:** Three hidden layers learn a compact device fingerprint
- **Dual Output Heads:** Separate heads for mean predictions and uncertainty
- **Log-variance Clamping:** logvar clamped to [-5.0, 5.0] for numerical stability

### Training Details

**Loss Function:** Heteroscedastic Loss
```
L = mean(Σ[(μ - y)² * exp(-logvar) + logvar])
```

**Optimizer:** Adam (Learning Rate: 1e-3)

**Scheduler:** ReduceLROnPlateau (factor=0.5, patience=5)

**Training Configuration:**
- Batch Size: 128 (train), 256 (val/test)
- Max Epochs: 80
- Early Stopping: Patience 10
- Train/Val/Test Split: 70%/15%/15%
- Random State: 42

**Preprocessing:**
- **Numerical Features (25):**
  - Median imputation for missing values
  - StandardScaler (zero mean, unit variance)
  
- **Categorical Features (80):**
  - Most frequent imputation
  - OneHotEncoder (handle_unknown="ignore")
  
- **Combined:** ColumnTransformer combines both pipelines

**Final Input Dimension:** 2,952 (after one-hot encoding)

### Model Outputs

1. **μ_PCE:** Mean predicted Power Conversion Efficiency (%)
2. **μ_log(T80):** Mean predicted log(1 + T80) where T80 is stability in hours
3. **σ_PCE:** Uncertainty in PCE prediction (standard deviation)
4. **σ_log(T80):** Uncertainty in log(T80) prediction

**Post-processing:**
- T80 = exp(μ_log(T80)) - 1
- σ_T80 = σ_log(T80) * (T80 + 1e-8)

---

## Backend API

### Technology Stack
- **Framework:** FastAPI
- **Server:** Uvicorn (ASGI)
- **Validation:** Pydantic
- **CORS:** Enabled for frontend communication

### API Endpoints

#### 1. POST `/predict`
**Purpose:** Quick prediction using 13 key features

**Request Body:**
```json
{
  "bandgap_eV": 1.55,
  "JV_default_Jsc_mAcm2": 23.4,
  "JV_default_Voc_V": 1.10,
  "JV_default_FF": 78.5,
  "Perovskite_deposition_annealing_temperature_C": 100.0,
  "Perovskite_deposition_annealing_time_s": 600.0,
  "Cell_architecture": "n-i-p",
  "Perovskite_composition_short_form": "FA0.8MA0.2PbI3",
  "ETL_material": "TiO2",
  "HTL_material": "Spiro-OMeTAD",
  "Perovskite_deposition_method": "Spin-coating",
  "Additive_type": "MACl",
  "Encapsulation": "Yes"
}
```

**Response:**
```json
{
  "pce_pred": 15.23,
  "t80_pred_hours": 450.5,
  "score": 65.8,
  "sigma_pce": 1.2,
  "sigma_sta_log": 0.3,
  "p_norm": 0.65,
  "s_norm": 0.31,
  "feature_importance": [...],
  "suggestions": [...],
  "user_features": [...]
}
```

**Processing Steps:**
1. Map API field names to internal feature names
2. Build DataFrame with defaults
3. Preprocess using saved pipeline
4. Run model inference
5. Calculate Perfection Score
6. Generate feature importance (gradient-based)
7. Generate improvement suggestions
8. Return comprehensive results

#### 2. GET `/features`
**Purpose:** Get list of all available features

**Response:**
```json
{
  "numeric_features": ["JV_default_Jsc", ...],
  "categorical_features": ["Cell_architecture", ...],
  "total_features": 105
}
```

#### 3. POST `/predict/full`
**Purpose:** Full prediction using all 105 features

**Request Body:**
```json
{
  "JV_default_Jsc": 23.4,
  "JV_default_Voc": 1.10,
  "Cell_architecture": "n-i-p",
  ...
}
```

**Response:**
```json
{
  "pce_pred": 15.23,
  "t80_pred_hours": 450.5,
  "score": 65.8,
  "sigma_pce": 1.2,
  "sigma_sta_log": 0.3
}
```

#### 4. POST `/score-to-features`
**Purpose:** Reverse engineering - get feature recommendations for target score

**Request Body:**
```json
{
  "target_score": 80.0,
  "initial_features": {}
}
```

**Response:**
```json
{
  "target_score": 80.0,
  "recommended_features": {
    "JV_default_Jsc": 25.0,
    "JV_default_Voc": 1.15,
    "JV_default_FF": 0.80,
    "Encapsulation": "Yes"
  },
  "recommendations": [...]
}
```

### Key Backend Functions

#### `build_row(payload: PredictRequest)`
- Maps API field names to internal feature names
- Handles special cases (e.g., FF conversion from % to decimal)
- Creates DataFrame with proper defaults

#### `compute_score(mu_pce, mu_sta, sigma_pce, sigma_sta)`
- Normalizes PCE and stability to [0, 1]
- Applies scaling for low values (< 0.05)
- Calculates quality score (70% PCE + 30% stability)
- Applies uncertainty penalty
- Returns final score (0-100)

#### `compute_feature_importance(...)`
- Uses gradient-based feature importance
- Generates improvement suggestions based on predictions
- Analyzes user-provided features
- Returns actionable recommendations

---

## Frontend Application

### Technology Stack
- **Framework:** React 18
- **Language:** TypeScript
- **Build Tool:** Vite
- **Routing:** React Router DOM v7
- **Styling:** CSS with CSS Variables (theme support)

### Application Structure

```
src/
├── main.tsx          # Entry point
├── App.tsx           # Main app component (routing, theme)
├── styles.css        # Global styles and theme
└── pages/
    ├── Home.tsx              # Landing page
    ├── SimplePredict.tsx     # Quick predict form
    ├── FeaturesToScore.tsx   # Full predict form
    └── ScoreToFeatures.tsx   # Reverse engineering
```

### Key Components

#### 1. App.tsx
**Purpose:** Root component with routing and theme management

**Features:**
- React Router setup
- Theme state management (dark/light)
- LocalStorage persistence
- Navigation bar
- Theme toggle button

**Routes:**
- `/` → Home page
- `/simple-predict` → Quick Predict
- `/full-predict` → Full Predict
- `/score-to-features` → Score → Features

#### 2. Home.tsx
**Purpose:** Landing page with project overview

**Sections:**
- Hero section with animated statistics
- Key benefits (6 cards)
- "What is this?" explanation
- Model architecture overview
- Detailed PSCNet architecture visualization
- Feature cards (3 main features)
- Perfection Score explanation
- Use cases
- Call-to-action

**Animations:**
- Animated counters for statistics
- Fade-in animations
- Hover effects
- Gradient text

#### 3. SimplePredict.tsx
**Purpose:** Quick prediction form (13 features)

**Features:**
- 6 numeric fields (bandgap, Jsc, Voc, FF, annealing temp/time)
- 7 categorical fields (architecture, composition, ETL, HTL, method, additive, encapsulation)
- Field descriptions for each input
- Preset configurations (3 presets)
- Real-time form validation
- Loading states
- Error handling
- Results display with:
  - Perfection Score (large display)
  - Progress bar
  - PCE and T80 predictions
  - Uncertainty values
  - Improvement suggestions
  - Feature importance
  - User feature analysis
  - Prediction history (localStorage)
  - Baseline comparison

**State Management:**
- Form state (controlled inputs)
- Prediction results
- Loading/error states
- History (localStorage)
- Baseline for comparison

#### 4. FeaturesToScore.tsx
**Purpose:** Full prediction form (all 105 features)

**Features:**
- Dynamically loads feature list from API
- Separate numeric and categorical inputs
- Scrollable form (max-height)
- Real-time validation
- Results panel with:
  - Perfection Score
  - PCE prediction
  - T80 prediction
  - Uncertainty values

**Data Flow:**
1. On mount: Fetch feature metadata from `/features`
2. Initialize form with all features
3. User fills form
4. Submit: Send all non-empty features to `/predict/full`
5. Display results

#### 5. ScoreToFeatures.tsx
**Purpose:** Reverse engineering (target score → features)

**Features:**
- Single input: target score (0-100)
- Validation (must be 1-100)
- Recommendations based on score:
  - High (80+): Efficiency ≥18%, Stability ≥1000h
  - Medium (50-80): Efficiency ≥12%, Stability ≥100h
  - Low (<50): Basic JV parameters
- Feature suggestions with target values

---

## Features in Detail

### Feature 1: Quick Predict

**Purpose:** Fast screening using 13 essential features

**Input Fields:**
1. **Bandgap Energy (eV)** - Energy gap between electron bands
2. **Short-Circuit Current (mA/cm²)** - Current when cell is shorted
3. **Open-Circuit Voltage (V)** - Voltage when no current flows
4. **Fill Factor (%)** - How well the cell uses its power
5. **Annealing Temperature (°C)** - Heating temperature during crystal formation
6. **Annealing Time (seconds)** - How long to heat the material
7. **Cell Architecture** - n-i-p or p-i-n
8. **Perovskite Composition** - Chemical formula
9. **ETL Material** - Electron transport layer
10. **HTL Material** - Hole transport layer
11. **Deposition Method** - How material is applied
12. **Additive Type** - Extra chemicals
13. **Encapsulation** - Protective coating

**Outputs:**
- Perfection Score (0-100)
- Predicted PCE (%)
- Predicted T80 (hours)
- Uncertainty estimates
- Feature importance
- Improvement suggestions
- User feature analysis

**Special Features:**
- Preset configurations
- Prediction history
- Baseline comparison
- Real-time suggestions

### Feature 2: Full Predict

**Purpose:** Most accurate predictions using all 105 features

**Input:** All features from training data

**Outputs:**
- Perfection Score
- Predicted PCE
- Predicted T80
- Uncertainty estimates

**Use Case:** Research-grade predictions when you have complete device information

### Feature 3: Score → Features

**Purpose:** Reverse engineering - what features do I need for a target score?

**Input:** Target Perfection Score (0-100)

**Outputs:**
- Recommended feature values
- Target ranges for PCE and stability
- Key parameters to focus on
- General recommendations

**Logic:**
- Score > 80: High-performance targets
- Score 50-80: Moderate targets
- Score < 50: Basic improvement targets

### Feature 4: Improvement Suggestions

**Purpose:** Actionable recommendations based on predictions

**Types of Suggestions:**
1. **Efficiency-based:**
   - If PCE < p_min: "Focus on improving Jsc, Voc, and Fill Factor"
   - If PCE in lower range: "Consider optimizing material composition"
   - If PCE < 80% of max: "Need efficiency ~X% to reach 90+ score"

2. **Stability-based:**
   - If T80 < s_min: "Consider better encapsulation"
   - If T80 could be improved: "Better encapsulation and material quality"
   - If T80 < 80% of max: "Need stability ~Xh to reach 90+ score"

3. **Feature-specific:**
   - Jsc < 25: "Increase to X mA/cm²"
   - Voc < 1.15: "Increase to X V"
   - FF < 80: "Increase to X%"

4. **Context-aware:**
   - Excellent JV but low prediction: "Model is conservative with incomplete info"
   - High performance: "You're in top 10% range"

### Feature 5: Feature Importance

**Purpose:** Understand which features matter most

**Method:** Gradient-based importance
- Enable gradients on input tensor
- Forward pass through model
- Backward pass to compute gradients
- Top 5 features by absolute gradient value

**Display:** Shows top features with impact scores

### Feature 6: Theme Switching

**Purpose:** User preference for dark/light mode

**Implementation:**
- CSS Variables for colors
- LocalStorage persistence
- Smooth transitions
- All components theme-aware

**Theme Variables:**
- `--bg`: Background color
- `--bg-elevated`: Elevated surfaces
- `--text`: Primary text color
- `--muted`: Secondary text color
- `--primary`: Primary accent color
- `--border-subtle`: Border color

### Feature 7: Prediction History

**Purpose:** Track previous predictions

**Storage:** LocalStorage (browser)

**Features:**
- Stores last 8 predictions
- Shows score and PCE
- Clickable to review

### Feature 8: Baseline Comparison

**Purpose:** Compare current prediction with a baseline

**Features:**
- Set any prediction as baseline
- Load baseline values
- Side-by-side comparison (PCE, T80, Score)

---

## Code Structure

### Project Directory Structure

```
mini_project_22222/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── model_best.pt              # Trained model weights
│   ├── preprocess.joblib          # Preprocessing pipeline
│   ├── artifacts.json             # Model metadata
│   ├── test_predict.py            # API test script
│   └── venv/                      # Virtual environment
│
├── frontend/
│   ├── src/
│   │   ├── main.tsx               # Entry point
│   │   ├── App.tsx                # Root component
│   │   ├── styles.css             # Global styles
│   │   └── pages/
│   │       ├── Home.tsx
│   │       ├── SimplePredict.tsx
│   │       ├── FeaturesToScore.tsx
│   │       └── ScoreToFeatures.tsx
│   ├── index.html                 # HTML template
│   ├── package.json               # Dependencies
│   ├── vite.config.ts             # Vite configuration
│   └── tsconfig.json              # TypeScript config
│
├── app.py                         # Streamlit alternative (not used)
├── perovskite_gamified_ml_xai.ipynb  # Model training notebook
├── retrain_robust_model.py        # Retraining script
├── README.md
├── RUN_INSTRUCTIONS.md
├── RETRAIN_GUIDE.md
└── SCORE_90_GUIDE.md
```

---

## Data Flow

### Quick Predict Flow

```
User Input (13 features)
    ↓
SimplePredict.tsx
    ↓
POST /predict
    ↓
Backend: build_row()
    ↓
Preprocessing Pipeline
    ↓
PSCNet Model Inference
    ↓
Post-processing (exp, calculations)
    ↓
compute_score()
    ↓
compute_feature_importance()
    ↓
Generate suggestions
    ↓
JSON Response
    ↓
SimplePredict.tsx displays results
```

### Full Predict Flow

```
User Input (105 features)
    ↓
FeaturesToScore.tsx
    ↓
POST /predict/full
    ↓
Backend: build_row_full()
    ↓
Preprocessing Pipeline
    ↓
PSCNet Model Inference
    ↓
compute_score()
    ↓
JSON Response
    ↓
FeaturesToScore.tsx displays results
```

### Score-to-Features Flow

```
User Input (target score)
    ↓
ScoreToFeatures.tsx
    ↓
POST /score-to-features
    ↓
Backend: score_to_features()
    ↓
Rule-based recommendations
    ↓
JSON Response
    ↓
ScoreToFeatures.tsx displays recommendations
```

---

## Technical Implementation

### Backend Implementation Details

#### Model Loading
```python
# Load artifacts
with open(artifacts_path) as f:
    artifacts = json.load(f)

# Load preprocessor
preprocess = joblib.load(preprocess_path)

# Load model
model = PSCNet(input_dim)
state = torch.load(model_path, map_location="cpu")
model.load_state_dict(state)
model.eval()  # Set to evaluation mode
```

#### Preprocessing
```python
# Transform input
encoded = preprocess.transform(row)
if hasattr(encoded, "toarray"):
    encoded = encoded.toarray()  # Convert sparse to dense
encoded_np = encoded.astype(np.float32)
```

#### Model Inference
```python
x = torch.from_numpy(encoded_np)
with torch.no_grad():  # Disable gradient computation
    mu, logvar = model(x)
    
# Convert to numpy
mu = mu.numpy().flatten()
logvar = logvar.numpy().flatten()
sigma = np.sqrt(np.exp(logvar))
```

#### Score Calculation
```python
# Normalize PCE
p_norm_raw = (mu_pce - p_min) / (p_max - p_min + 1e-8)

# Apply scaling for low values
if p_norm > 0 and p_norm < 0.05:
    p_norm_scaled = p_norm * 8.0  # Boost low values
elif p_norm >= 0.05:
    p_norm_scaled = 0.4 + (p_norm - 0.05) * 0.6 / 0.95

# Calculate quality
q = 0.7 * p_norm_scaled + 0.3 * s_norm_scaled

# Uncertainty penalty
u = 1.0 - 0.4 * (sigma_pce / sig_ref_p) - 0.4 * (sigma_sta / sig_ref_s)

# Final score
score = 100.0 * q * u
```

### Frontend Implementation Details

#### State Management
- **React Hooks:** useState, useEffect, useMemo
- **LocalStorage:** For theme and prediction history
- **No external state management:** Pure React

#### API Communication
```typescript
const res = await fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});
const data = await res.json();
```

#### Form Handling
- Controlled components (React)
- Real-time validation
- Type-safe with TypeScript

#### Theme System
```css
.theme-dark {
  --bg: #070a12;
  --text: #f3f4f6;
  ...
}

.theme-light {
  --bg: #f3f4f6;
  --text: #0f172a;
  ...
}
```

---

## File-by-File Breakdown

### Backend Files

#### `backend/main.py`
**Purpose:** FastAPI application with all endpoints

**Key Components:**
1. **PSCNet Class:** Neural network architecture
2. **Pydantic Models:** Request/response validation
3. **FastAPI App:** Main application instance
4. **CORS Middleware:** Enable cross-origin requests
5. **Helper Functions:**
   - `build_row()`: Convert API request to DataFrame
   - `build_row_full()`: Handle full feature set
   - `compute_score()`: Calculate Perfection Score
   - `compute_feature_importance()`: Gradient-based importance
6. **API Endpoints:**
   - `/predict`: Quick predict
   - `/features`: Get feature list
   - `/predict/full`: Full predict
   - `/score-to-features`: Reverse engineering

**Lines of Code:** ~409

#### `backend/artifacts.json`
**Purpose:** Model metadata

**Contents:**
- `num_cols`: List of 25 numeric feature names
- `cat_cols`: List of 80 categorical feature names
- `p_min`, `p_max`: PCE normalization bounds (5th, 95th percentile)
- `s_min`, `s_max`: Stability normalization bounds
- `sig_ref_p`, `sig_ref_s`: Reference uncertainties for penalty
- `input_dim`: Input dimension after encoding (2952)

#### `backend/model_best.pt`
**Purpose:** Trained PyTorch model weights

**Format:** PyTorch state dict
**Size:** ~2-5 MB (depends on model)

#### `backend/preprocess.joblib`
**Purpose:** Scikit-learn preprocessing pipeline

**Contains:**
- ColumnTransformer
- Numerical pipeline (imputer + scaler)
- Categorical pipeline (imputer + encoder)

### Frontend Files

#### `frontend/src/main.tsx`
**Purpose:** Application entry point

**Code:**
```typescript
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
```

#### `frontend/src/App.tsx`
**Purpose:** Root component with routing

**Features:**
- BrowserRouter setup
- Theme state management
- Navigation bar
- Route definitions
- Theme toggle

**Lines of Code:** ~135

#### `frontend/src/pages/Home.tsx`
**Purpose:** Landing page

**Components:**
- AnimatedCounter: Counts up statistics
- Hero section
- Benefits section
- Model architecture visualization
- Feature cards
- Use cases
- CTA section

**Lines of Code:** ~950+

#### `frontend/src/pages/SimplePredict.tsx`
**Purpose:** Quick predict form

**Components:**
- Form state management
- Field definitions (numeric + categorical)
- Preset configurations
- API integration
- Results display
- History management
- Baseline comparison

**Lines of Code:** ~610

#### `frontend/src/pages/FeaturesToScore.tsx`
**Purpose:** Full predict form

**Components:**
- Dynamic feature loading
- Form generation
- API integration
- Results display

**Lines of Code:** ~289

#### `frontend/src/pages/ScoreToFeatures.tsx`
**Purpose:** Reverse engineering

**Components:**
- Target score input
- API integration
- Recommendations display

**Lines of Code:** ~196

#### `frontend/src/styles.css`
**Purpose:** Global styles and theme

**Contents:**
- CSS Variables (theme colors)
- Theme classes (.theme-dark, .theme-light)
- Animations (@keyframes)
- Utility classes (.hover-lift, .fade-in, etc.)

---

## API Endpoints

### 1. POST `/predict`

**Request:**
```json
{
  "bandgap_eV": 1.55,
  "JV_default_Jsc_mAcm2": 23.4,
  "JV_default_Voc_V": 1.10,
  "JV_default_FF": 78.5,
  "Perovskite_deposition_annealing_temperature_C": 100.0,
  "Perovskite_deposition_annealing_time_s": 600.0,
  "Cell_architecture": "n-i-p",
  "Perovskite_composition_short_form": "FA0.8MA0.2PbI3",
  "ETL_material": "TiO2",
  "HTL_material": "Spiro-OMeTAD",
  "Perovskite_deposition_method": "Spin-coating",
  "Additive_type": "MACl",
  "Encapsulation": "Yes"
}
```

**Response:**
```json
{
  "pce_pred": 15.23,
  "t80_pred_hours": 450.5,
  "score": 65.8,
  "sigma_pce": 1.2,
  "sigma_sta_log": 0.3,
  "p_norm": 0.65,
  "s_norm": 0.31,
  "feature_importance": [
    {"name": "Feature 1", "impact": 0.0234},
    ...
  ],
  "suggestions": [
    "Efficiency (15.23%) is in the lower range...",
    ...
  ],
  "user_features": [
    {
      "name": "Short-Circuit Current (Jsc)",
      "value": 23.4,
      "unit": "mA/cm²",
      "impact": "high",
      "suggestion": "Good value"
    },
    ...
  ]
}
```

### 2. GET `/features`

**Response:**
```json
{
  "numeric_features": [
    "Cell_area_measured",
    "JV_default_Jsc",
    ...
  ],
  "categorical_features": [
    "Cell_architecture",
    "Perovskite_composition_short_form",
    ...
  ],
  "total_features": 105
}
```

### 3. POST `/predict/full`

**Request:**
```json
{
  "JV_default_Jsc": 23.4,
  "JV_default_Voc": 1.10,
  "Cell_architecture": "n-i-p",
  ...
}
```

**Response:**
```json
{
  "pce_pred": 15.23,
  "t80_pred_hours": 450.5,
  "score": 65.8,
  "sigma_pce": 1.2,
  "sigma_sta_log": 0.3
}
```

### 4. POST `/score-to-features`

**Request:**
```json
{
  "target_score": 80.0,
  "initial_features": {}
}
```

**Response:**
```json
{
  "target_score": 80.0,
  "recommended_features": {
    "JV_default_Jsc": 25.0,
    "JV_default_Voc": 1.15,
    "JV_default_FF": 0.80,
    "Encapsulation": "Yes"
  },
  "recommendations": [
    "For high scores (80+), aim for: Efficiency ≥18%, Stability ≥1000h",
    "Key features: High Jsc (≥25 mA/cm²), High Voc (≥1.15V), High FF (≥80%)",
    ...
  ]
}
```

---

## UI Components

### Navigation Bar
- Logo/Brand name
- Navigation links (Home, Quick Predict, Full Predict, Score → Features)
- Theme toggle button

### Form Components
- **Input Fields:**
  - Number inputs (with units)
  - Text inputs
  - Select dropdowns
  - Field descriptions
  
- **Buttons:**
  - Primary action buttons
  - Preset buttons
  - Theme toggle

### Results Display
- **Score Display:**
  - Large score number
  - Progress bar (color-coded)
  - Score message
  
- **Metrics:**
  - PCE prediction
  - T80 prediction
  - Uncertainty values
  
- **Suggestions:**
  - Improvement recommendations
  - Feature importance
  - User feature analysis

### Cards
- Feature cards (hover effects)
- Benefit cards
- Stat cards (animated)

### Animations
- Fade-in on load
- Counter animations
- Hover effects
- Progress bar animations

---

## Scoring Algorithm

### Perfection Score Formula

**Step 1: Normalize PCE**
```
p_norm_raw = (mu_pce - p_min) / (p_max - p_min + 1e-8)
```

**Step 2: Apply Scaling (for low values)**
```
if p_norm > 0 and p_norm < 0.05:
    p_norm_scaled = p_norm * 8.0  # Boost low values
elif p_norm >= 0.05:
    p_norm_scaled = 0.4 + (p_norm - 0.05) * 0.6 / 0.95
else:
    p_norm_scaled = 0.0
```

**Step 3: Normalize Stability**
```
s_norm_raw = (mu_sta - s_min) / (s_max - s_min + 1e-8)
# Apply same scaling as PCE
```

**Step 4: Calculate Quality**
```
q = 0.7 * p_norm_scaled + 0.3 * s_norm_scaled
```

**Step 5: Calculate Uncertainty Penalty**
```
u = 1.0 - 0.4 * (sigma_pce / sig_ref_p) - 0.4 * (sigma_sta / sig_ref_s)
u = clip(u, 0.0, 1.0)
```

**Step 6: Apply Penalties (if values below range)**
```
if p_norm_raw < 0 or s_norm_raw < 0:
    penalty = min(0.9, abs(p_norm_raw) * 0.7 + abs(s_norm_raw) * 0.3)
    base_score = base_score * (1.0 - penalty)
```

**Step 7: Final Score**
```
score = 100.0 * q * u
score = clip(score, 0.0, 100.0)
```

### Score Interpretation

- **90-100:** Excellent recipe, top-tier performance
- **75-90:** High-quality recipe, research-grade
- **50-75:** Good recipe, can be optimized
- **25-50:** Decent recipe, needs improvement
- **0-25:** Poor recipe, significant changes needed

---

## Deployment

### Development Setup

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install fastapi uvicorn torch pandas numpy joblib pydantic scikit-learn
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### Production Build

**Frontend:**
```bash
cd frontend
npm run build
# Output in dist/ folder
```

**Backend:**
```bash
# Use production ASGI server
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Environment Variables
- Backend port: 8000 (default)
- Frontend port: 5173 (default)
- CORS origins: Configured in main.py

---

## Key Technical Decisions

### 1. Why Heteroscedastic Model?
- Provides uncertainty estimates
- More informative than point predictions
- Allows confidence-based scoring

### 2. Why FastAPI?
- Fast and modern
- Automatic API documentation
- Type validation with Pydantic
- Async support

### 3. Why React + TypeScript?
- Type safety
- Component reusability
- Large ecosystem
- Good developer experience

### 4. Why CSS Variables for Theming?
- No runtime overhead
- Easy to maintain
- Smooth transitions
- No JavaScript needed

### 5. Why LocalStorage for History?
- No backend needed
- Fast access
- Persists across sessions
- Simple implementation

---

## Future Enhancements

1. **User Authentication:** Save predictions to user accounts
2. **Batch Predictions:** Upload CSV for multiple predictions
3. **Export Results:** Download predictions as CSV/PDF
4. **Advanced Visualizations:** Charts for predictions over time
5. **Model Retraining:** Web interface for retraining
6. **SHAP Integration:** Better feature importance visualization
7. **Comparison Tool:** Compare multiple recipes side-by-side
8. **Mobile App:** React Native version

---

## Conclusion

This project demonstrates a complete machine learning application stack:
- **Data Science:** Model training and evaluation
- **Backend Development:** REST API with FastAPI
- **Frontend Development:** Modern React application
- **ML Engineering:** Model deployment and inference
- **UX Design:** Intuitive interface with helpful features

The application successfully bridges the gap between research and practical application, making advanced ML predictions accessible to researchers and engineers working on perovskite solar cells.

---

## Appendix: Dependencies

### Backend Dependencies
```
fastapi==0.123.4
uvicorn==0.38.0
torch==2.9.1
pandas==2.3.3
numpy==2.3.5
joblib==1.5.2
pydantic==2.12.5
scikit-learn==1.7.2
```

### Frontend Dependencies
```
react==18.2.0
react-dom==18.2.0
react-router-dom==7.9.6
typescript==5.4.0
vite==5.0.0
@vitejs/plugin-react==4.0.0
```

---

**Report Generated:** 2025
**Project Version:** 1.0
**Authors:** Owaish Jamal, Rishu Raj, Vikas Singh, Trisha Bharti, Prachi Kumari
**Institution:** Indian Institute of Information Technology Allahabad
**Supervisor:** Dr. Upendra Kumar

