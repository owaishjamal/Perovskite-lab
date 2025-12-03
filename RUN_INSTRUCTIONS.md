# Perovskite Lab - Run Instructions

## Project Overview

This is **Perovskite Lab** - a web application for predicting perovskite solar cell performance with:
- **Backend**: FastAPI server (Python) that runs a PyTorch ML model for predicting perovskite solar cell performance
- **Frontend**: React + TypeScript web application (Vite) for user interface

## Prerequisites

### Required Software

1. **Python 3.8+** (with pip)
   - Check: `python --version` or `python3 --version`
   - Download: https://www.python.org/downloads/

2. **Node.js 16+** (with npm)
   - Check: `node --version` and `npm --version`
   - Download: https://nodejs.org/

### Python Dependencies

The backend requires these Python packages:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `torch` - PyTorch for ML model
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `joblib` - Model preprocessing
- `pydantic` - Data validation
- `scikit-learn` - Machine learning preprocessing (required for loading preprocessor)

### Frontend Dependencies

The frontend uses:
- React 18
- TypeScript
- Vite (build tool)
- React Router DOM

---

## Step-by-Step Setup Instructions

### Step 1: Navigate to Project Directory

```bash
cd mini_project_22222
```

### Step 2: Setup Backend

#### 2.1 Install Python Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install fastapi uvicorn torch pandas numpy joblib pydantic scikit-learn
```

**Note**: If you encounter issues installing PyTorch, you may need to install it separately:
```bash
# For CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA (GPU support):
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

#### 2.2 Verify Backend Files

Ensure these files exist in the `backend/` directory:
- `main.py` - FastAPI application
- `model_best.pt` - Trained ML model
- `preprocess.joblib` - Preprocessing pipeline
- `artifacts.json` - Model metadata

#### 2.3 Start Backend Server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
Using original full-feature model
INFO:     Application startup complete.
```

The backend API will be available at: **http://localhost:8000**

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

**Keep this terminal window open!**

---

### Step 3: Setup Frontend

#### 3.1 Open a New Terminal Window

Keep the backend running, and open a **new terminal** for the frontend.

#### 3.2 Navigate to Frontend Directory

```bash
cd mini_project_22222/frontend
```

#### 3.3 Install Frontend Dependencies

```bash
npm install
```

This will install all required packages listed in `package.json`.

#### 3.4 Start Frontend Development Server

```bash
npm run dev
```

**Expected output:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

The frontend will be available at: **http://localhost:5173**

**Keep this terminal window open!**

---

## Step 4: Access the Application

1. Open your web browser
2. Navigate to: **http://localhost:5173**
3. You should see the Perovskite Score application homepage

---

## Application Features

The application has several pages:

1. **Home** (`/`) - Landing page with overview
2. **Quick Predict** (`/simple-predict`) - Simple prediction form with 13 key features
3. **Full Predict** (`/full-predict`) - Full feature prediction form
4. **Score → Features** (`/score-to-features`) - Reverse prediction (target score to features)

---

## Testing the Backend API

You can test the backend API directly using the provided test script:

```bash
# From the backend directory
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
    "Perovskite_deposition_annealing_temperature_C": 100.0,
    "Perovskite_deposition_annealing_time_s": 600.0,
    "Cell_architecture": "n-i-p",
    "Perovskite_composition_short_form": "FA0.8MA0.2PbI3",
    "ETL_material": "TiO2",
    "HTL_material": "Spiro-OMeTAD",
    "Perovskite_deposition_method": "Spin-coating",
    "Additive_type": "MACl",
    "Encapsulation": "Yes"
  }'
```

---

## Troubleshooting

### Backend Issues

**Problem: ModuleNotFoundError (especially sklearn)**
- **Solution**: Make sure you've installed all dependencies: `pip install fastapi uvicorn torch pandas numpy joblib pydantic scikit-learn`
- The preprocessor files require scikit-learn to be installed

**Problem: CUDA/GPU errors**
- **Solution**: The model runs on CPU by default. If you see CUDA errors, ensure PyTorch CPU version is installed.

**Problem: Port 8000 already in use**
- **Solution**: Use a different port: `uvicorn main:app --reload --port 8001`
- **Note**: You'll need to update the CORS origins in `backend/main.py` if using a different port.

**Problem: Model files not found**
- **Solution**: Ensure you're running from the `backend/` directory, or update file paths in `main.py`

### Frontend Issues

**Problem: npm install fails**
- **Solution**: 
  - Delete `node_modules` folder and `package-lock.json`
  - Run `npm install` again
  - If issues persist, try `npm cache clean --force`

**Problem: Port 5173 already in use**
- **Solution**: Vite will automatically use the next available port, or you can specify: `npm run dev -- --port 3000`

**Problem: Cannot connect to backend**
- **Solution**: 
  - Ensure backend is running on port 8000
  - Check CORS settings in `backend/main.py` (should allow `http://localhost:5173`)
  - Check browser console for errors

**Problem: Frontend shows connection errors**
- **Solution**: Verify backend is running and accessible at http://localhost:8000/docs

---

## Production Build (Optional)

### Build Frontend for Production

```bash
cd frontend
npm run build
```

This creates a `dist/` folder with optimized production files.

### Preview Production Build

```bash
npm run preview
```

---

## Project Structure

```
mini_project_22222/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── model_best.pt           # Trained ML model
│   ├── model_robust_subset.pt  # Alternative robust model
│   ├── preprocess.joblib       # Preprocessing pipeline
│   ├── artifacts.json          # Model metadata
│   └── test_predict.py         # API test script
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main React component
│   │   ├── main.tsx           # Entry point
│   │   ├── pages/             # Page components
│   │   └── styles.css         # Global styles
│   ├── package.json           # Frontend dependencies
│   └── vite.config.ts         # Vite configuration
├── app.py                     # Streamlit app (alternative UI)
├── README.md                  # Basic readme
├── RETRAIN_GUIDE.md          # Model retraining guide
└── SCORE_90_GUIDE.md         # Guide for achieving high scores
```

---

## Quick Start Summary

**Terminal 1 (Backend):**
```bash
cd mini_project_22222/backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install fastapi uvicorn torch pandas numpy joblib pydantic scikit-learn
uvicorn main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd mini_project_22222/frontend
npm install
npm run dev
```

**Browser:**
- Open http://localhost:5173

---

## Additional Resources

- **Model Retraining**: See `RETRAIN_GUIDE.md` for instructions on retraining the model
- **Achieving High Scores**: See `SCORE_90_GUIDE.md` for tips on getting 90+ scores
- **API Documentation**: Visit http://localhost:8000/docs when backend is running

---

## Support

If you encounter issues:
1. Check that both backend and frontend are running
2. Verify all dependencies are installed
3. Check terminal output for error messages
4. Review the troubleshooting section above

