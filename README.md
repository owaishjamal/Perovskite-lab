# 🔬 Perovskite Lab

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18.2-blue.svg)
![TypeScript](https://img.shields.io/badge/TypeScript-5.4-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.123-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**AI-Powered Design Assistant for High-Efficiency, Stable Perovskite Solar Cells**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Deployment](#-deployment) • [API Documentation](#-api-documentation) • [Project Structure](#-project-structure)

</div>

---

## 📖 About

**Perovskite Lab** is a web-based application that uses deep learning to predict the performance of perovskite solar cells based on material properties, device architecture, and processing conditions. The system provides a **Perfection Score (0-100)** that combines efficiency, stability, and prediction uncertainty into a single interpretable metric.

### 🎯 Key Highlights

- 🚀 **Fast Predictions** - Get results in seconds, not hours
- 🎯 **Highly Accurate** - Trained on 1,570+ real devices with 92% prediction accuracy
- 🔬 **Research-Grade** - Built on peer-reviewed data and validated against experimental results
- 💡 **Actionable Insights** - Get feature recommendations to achieve target performance
- 📊 **Uncertainty Quantification** - Know how confident the model is with uncertainty estimates
- 🎨 **Modern UI** - Beautiful, responsive interface with dark/light mode

---

## ✨ Features

### 🎯 Quick Predict
Fast prediction using **13 key features**. Perfect for quick screening of recipe ideas with instant results and improvement suggestions.

### 🔬 Full Predict
Comprehensive prediction using all **105 training features**. Most accurate results for research-grade predictions when you have complete device information.

### 📈 Score → Features
Reverse engineering feature - Enter a target perfection score and get feature recommendations to achieve it. Perfect for goal-oriented design.

### 💡 Smart Suggestions
- **Improvement Recommendations** - Actionable suggestions based on predictions
- **Feature Importance** - Understand which features matter most (gradient-based)
- **User Feature Analysis** - Analysis of your input parameters with optimization tips

### 🎨 User Experience
- **Dark/Light Mode** - Toggle between themes
- **Prediction History** - Track your previous predictions (localStorage)
- **Baseline Comparison** - Compare current prediction with a baseline
- **Preset Configurations** - Quick-start with pre-configured recipes
- **Responsive Design** - Works on all screen sizes

---

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **PyTorch** - Deep learning framework for model inference
- **scikit-learn** - Machine learning preprocessing pipeline
- **Pandas & NumPy** - Data manipulation and numerical computing
- **Uvicorn** - ASGI server for FastAPI

### Frontend
- **React 18** - UI library
- **TypeScript** - Type-safe JavaScript
- **Vite** - Fast build tool and dev server
- **React Router DOM** - Client-side routing
- **CSS Variables** - Theme system

### Machine Learning
- **PSCNet** - Heteroscedastic neural network
- **Training Data** - 1,570 curated perovskite solar cell devices
- **Features** - 105 total (25 numeric + 80 categorical)
- **Outputs** - PCE (%), T80 stability (hours), uncertainty estimates

---

## 🚀 Installation

### Prerequisites

- **Python 3.8+** with pip
- **Node.js 16+** with npm
- **Git** (for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/perovskite-lab.git
cd perovskite-lab
```

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install fastapi uvicorn torch pandas numpy joblib pydantic scikit-learn
```

**Note:** If you encounter issues installing PyTorch, install it separately:
```bash
# For CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA (GPU support):
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies
npm install
```

---

## 🎮 Usage

### Starting the Application

#### Terminal 1: Backend Server

```bash
cd backend
venv\Scripts\activate  # Windows (or source venv/bin/activate on Linux/Mac)
uvicorn main:app --reload --port 8000
```

The backend API will be available at: **http://localhost:8000**

- **API Documentation:** http://localhost:8000/docs (Swagger UI)
- **Alternative Docs:** http://localhost:8000/redoc (ReDoc)

#### Terminal 2: Frontend Development Server

```bash
cd frontend
npm run dev
```

The frontend will be available at: **http://localhost:5173**

### Using the Application

1. **Open your browser** and navigate to `http://localhost:5173`
2. **Choose a feature:**
   - **Quick Predict** - Fast screening with 13 key features
   - **Full Predict** - Comprehensive prediction with all 105 features
   - **Score → Features** - Reverse engineering for target scores
3. **Fill in the form** with your perovskite recipe parameters
4. **Get predictions** including:
   - Perfection Score (0-100)
   - Predicted PCE (%)
   - Predicted T80 stability (hours)
   - Uncertainty estimates
   - Improvement suggestions

### Example: Quick Predict

```javascript
// Example API call
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    bandgap_eV: 1.55,
    JV_default_Jsc_mAcm2: 23.4,
    JV_default_Voc_V: 1.10,
    JV_default_FF: 78.5,
    Cell_architecture: "n-i-p",
    Perovskite_composition_short_form: "FA0.8MA0.2PbI3",
    ETL_material: "TiO2",
    HTL_material: "Spiro-OMeTAD",
    Encapsulation: "Yes"
  })
});

const result = await response.json();
console.log(result);
// {
//   pce_pred: 15.23,
//   t80_pred_hours: 450.5,
//   score: 65.8,
//   suggestions: [...],
//   ...
// }
```

---

## 📡 API Documentation

### Endpoints

#### 1. POST `/predict`
Quick prediction using 13 key features.

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
  "feature_importance": [...],
  "suggestions": [...],
  "user_features": [...]
}
```

#### 2. GET `/features`
Get list of all available features.

**Response:**
```json
{
  "numeric_features": ["JV_default_Jsc", ...],
  "categorical_features": ["Cell_architecture", ...],
  "total_features": 105
}
```

#### 3. POST `/predict/full`
Full prediction using all 105 features.

**Request:**
```json
{
  "JV_default_Jsc": 23.4,
  "JV_default_Voc": 1.10,
  "Cell_architecture": "n-i-p",
  ...
}
```

#### 4. POST `/score-to-features`
Get feature recommendations for a target score.

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
  "recommendations": [...]
}
```

For detailed API documentation, visit **http://localhost:8000/docs** when the backend is running.

---

## 📁 Project Structure

```
perovskite-lab/
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── model_best.pt              # Trained PyTorch model
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
│   │       ├── Home.tsx           # Landing page
│   │       ├── SimplePredict.tsx  # Quick predict
│   │       ├── FeaturesToScore.tsx # Full predict
│   │       └── ScoreToFeatures.tsx # Reverse engineering
│   ├── index.html                 # HTML template
│   ├── package.json               # Dependencies
│   └── vite.config.ts             # Vite configuration
│
├── perovskite_gamified_ml_xai.ipynb  # Model training notebook
├── retrain_robust_model.py        # Retraining script
├── PROJECT_REPORT.md              # Comprehensive project documentation
├── RUN_INSTRUCTIONS.md            # Detailed setup guide
├── RETRAIN_GUIDE.md               # Model retraining guide
├── SCORE_90_GUIDE.md              # Guide for achieving high scores
└── README.md                      # This file
```

---

## 🧠 Model Architecture

### PSCNet: Heteroscedastic Neural Network

```
Input Layer (2952 dimensions after encoding)
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

### Training Details

- **Dataset:** 1,570 curated perovskite solar cell devices
- **Features:** 105 total (25 numeric + 80 categorical)
- **Loss Function:** Heteroscedastic Loss
- **Optimizer:** Adam (LR: 1e-3)
- **Batch Size:** 128 (train), 256 (val/test)
- **Max Epochs:** 80 with early stopping

### Perfection Score Formula

The Perfection Score combines:
- **Efficiency (70% weight)** - Normalized PCE
- **Stability (30% weight)** - Normalized T80
- **Uncertainty Penalty** - Reduces score for high uncertainty

```
Score = 100 × Quality × Uncertainty_Penalty
where Quality = 0.7 × P_norm + 0.3 × S_norm
```

---

## 🧪 Testing

### Test Backend API

```bash
cd backend
python test_predict.py
```

Or use curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "bandgap_eV": 1.55,
    "JV_default_Jsc_mAcm2": 23.4,
    "JV_default_Voc_V": 1.10,
    "JV_default_FF": 78.5,
    "Cell_architecture": "n-i-p",
    "Perovskite_composition_short_form": "FA0.8MA0.2PbI3",
    "ETL_material": "TiO2",
    "HTL_material": "Spiro-OMeTAD",
    "Encapsulation": "Yes"
  }'
```

---

## 🐛 Troubleshooting

### Backend Issues

**Problem:** `ModuleNotFoundError: No module named 'sklearn'`
- **Solution:** Install scikit-learn: `pip install scikit-learn`

**Problem:** Port 8000 already in use
- **Solution:** Use a different port: `uvicorn main:app --reload --port 8001`
- **Note:** Update CORS origins in `backend/main.py` if using a different port

**Problem:** CUDA/GPU errors
- **Solution:** The model runs on CPU by default. Install CPU-only PyTorch if needed.

### Frontend Issues

**Problem:** `npm install` fails
- **Solution:** 
  ```bash
  rm -rf node_modules package-lock.json
  npm install
  ```

**Problem:** Cannot connect to backend
- **Solution:** 
  - Ensure backend is running on port 8000
  - Check CORS settings in `backend/main.py`
  - Check browser console for errors

**Problem:** Port 5173 already in use
- **Solution:** Vite will automatically use the next available port

---

## 🚀 Deployment

Ready to deploy? We've got you covered!

### Quick Deploy (Railway + Vercel)

**Backend (Railway):**
1. Sign up at [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Set root directory: `backend`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy! 🎉

**Frontend (Vercel):**
1. Sign up at [vercel.com](https://vercel.com)
2. Import from GitHub
3. Set root directory: `frontend`
4. Add env var: `VITE_API_URL` = your Railway backend URL
5. Deploy! 🎉

### Detailed Guide

For complete deployment instructions, see **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)**

Includes:
- Step-by-step guides for multiple platforms
- Environment variable setup
- CORS configuration
- Troubleshooting
- Production optimizations

---

## 📚 Documentation

- **[PROJECT_REPORT.md](PROJECT_REPORT.md)** - Comprehensive project documentation
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete deployment instructions
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Quick deployment checklist
- **[RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md)** - Detailed setup and run instructions
- **[RETRAIN_GUIDE.md](RETRAIN_GUIDE.md)** - Guide for retraining the model
- **[SCORE_90_GUIDE.md](SCORE_90_GUIDE.md)** - Guide for achieving 90+ scores

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Owaish Jamal**
- **Rishu Raj**
- **Vikas Singh**
- **Trisha Bharti**
- **Prachi Kumari**

**Supervisor:** Dr. Upendra Kumar

**Institution:** Indian Institute of Information Technology Allahabad

---

## 🙏 Acknowledgments

- Perovskite solar cell research community
- Open-source ML libraries (PyTorch, scikit-learn, FastAPI, React)
- All contributors and testers

---

## 📊 Project Statistics

- **Training Devices:** 1,570
- **Features:** 105 (25 numeric + 80 categorical)
- **Model Accuracy:** 92%
- **Input Dimension:** 2,952 (after encoding)
- **Model Parameters:** ~750K
- **Prediction Time:** < 100ms

---

## 🔮 Future Enhancements

- [ ] User authentication and saved predictions
- [ ] Batch prediction (CSV upload)
- [ ] Export results (CSV/PDF)
- [ ] Advanced visualizations and charts
- [ ] SHAP integration for better feature importance
- [ ] Comparison tool for multiple recipes
- [ ] Mobile app (React Native)
- [ ] Model retraining web interface

---

## 📞 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

<div align="center">

**Made with ❤️ for the perovskite solar cell research community**

⭐ Star this repo if you find it helpful!

</div>
