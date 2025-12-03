# 🚀 Deployment Guide - Perovskite Lab

Complete instructions for deploying the application to production.

## Table of Contents

1. [Deployment Overview](#deployment-overview)
2. [Backend Deployment](#backend-deployment)
3. [Frontend Deployment](#frontend-deployment)
4. [Environment Variables](#environment-variables)
5. [CORS Configuration](#cors-configuration)
6. [Platform-Specific Guides](#platform-specific-guides)
7. [Troubleshooting](#troubleshooting)

---

## Deployment Overview

### Architecture

```
┌─────────────────┐
│   Frontend      │  (Vercel/Netlify)
│   (Static)      │
└────────┬────────┘
         │ HTTPS
         │ API Calls
         ▼
┌─────────────────┐
│   Backend API   │  (Railway/Render/Heroku)
│   (FastAPI)     │
└─────────────────┘
```

### Recommended Platforms

**Backend:**
- 🚂 **Railway** (Recommended - Easy, free tier)
- 🎨 **Render** (Free tier available)
- 🔵 **Heroku** (Paid, but reliable)
- ☁️ **AWS EC2/Elastic Beanstalk** (Advanced)

**Frontend:**
- ▲ **Vercel** (Recommended - Best for React)
- 🟢 **Netlify** (Great free tier)
- 📦 **GitHub Pages** (Free, but limited)

---

## Backend Deployment

### Prerequisites

1. **Create `requirements.txt`** in backend directory
2. **Create `Procfile`** (for Heroku/Railway)
3. **Update CORS** for production domain
4. **Environment variables** setup

### Step 1: Create requirements.txt

Create `backend/requirements.txt`:

```txt
fastapi==0.123.4
uvicorn[standard]==0.38.0
torch==2.9.1
pandas==2.3.3
numpy==2.3.5
joblib==1.5.2
pydantic==2.12.5
scikit-learn==1.7.2
python-multipart==0.0.9
```

**Note:** For smaller deployments, you might want CPU-only PyTorch:
```txt
torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Create Procfile

Create `backend/Procfile`:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Step 3: Create .gitignore (if not exists)

Create `.gitignore` in project root:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
*.egg-info/
dist/
build/

# Node
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.pnp
.pnp.js
dist/
build/

# Environment
.env
.env.local
.env.production

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Model files (if too large for git)
# Uncomment if needed:
# *.pt
# *.joblib
```

### Step 4: Update CORS in main.py

Update `backend/main.py` to allow your frontend domain:

```python
# Replace this line:
allow_origins=["http://localhost:5173", "http://localhost:3000"],

# With your production frontend URL:
allow_origins=[
    "http://localhost:5173",  # Keep for local dev
    "http://localhost:3000",
    "https://your-frontend-domain.vercel.app",  # Add your frontend URL
    "https://your-frontend-domain.netlify.app",
],
```

---

## Frontend Deployment

### Step 1: Update API URL

Create `frontend/.env.production`:

```env
VITE_API_URL=https://your-backend-api.railway.app
```

Or update API calls in code to use environment variable:

```typescript
// In your API calls, replace:
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Then use:
const res = await fetch(`${API_URL}/predict`, { ... });
```

### Step 2: Update All API Calls

Update all fetch calls in frontend files:

**SimplePredict.tsx, FeaturesToScore.tsx, ScoreToFeatures.tsx:**

```typescript
// Replace hardcoded URLs:
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Then in fetch calls:
const res = await fetch(`${API_URL}/predict`, { ... });
```

### Step 3: Build Frontend

```bash
cd frontend
npm run build
```

This creates a `dist/` folder with production-ready files.

---

## Platform-Specific Guides

### Option 1: Railway (Recommended - Easiest)

#### Backend on Railway

1. **Sign up** at [railway.app](https://railway.app)

2. **Create New Project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Connect your GitHub account
   - Select your repository

3. **Configure Backend Service**
   - Railway auto-detects Python
   - Set **Root Directory** to `backend`
   - Set **Start Command** to: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Add Environment Variables** (if needed)
   - Usually not required for this project

5. **Deploy**
   - Railway automatically deploys on push
   - Get your backend URL: `https://your-app.railway.app`

6. **Update CORS**
   - Add Railway URL to CORS origins in `main.py`
   - Redeploy

#### Frontend on Vercel

1. **Sign up** at [vercel.com](https://vercel.com)

2. **Import Project**
   - Click "Add New Project"
   - Import from GitHub
   - Select your repository

3. **Configure**
   - **Framework Preset:** Vite
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`

4. **Environment Variables**
   - Add: `VITE_API_URL` = `https://your-backend.railway.app`

5. **Deploy**
   - Click "Deploy"
   - Get your frontend URL: `https://your-app.vercel.app`

---

### Option 2: Render

#### Backend on Render

1. **Sign up** at [render.com](https://render.com)

2. **Create New Web Service**
   - Connect GitHub repository
   - Select your repo

3. **Configure**
   - **Name:** perovskite-backend
   - **Environment:** Python 3
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Root Directory:** `backend`

4. **Environment Variables**
   - `PYTHON_VERSION` = `3.12`

5. **Deploy**
   - Render will build and deploy
   - Get URL: `https://your-app.onrender.com`

**Note:** Render free tier spins down after inactivity. First request may be slow.

#### Frontend on Netlify

1. **Sign up** at [netlify.com](https://netlify.com)

2. **Add New Site**
   - "Import from Git"
   - Connect GitHub
   - Select repository

3. **Build Settings**
   - **Base directory:** `frontend`
   - **Build command:** `npm run build`
   - **Publish directory:** `frontend/dist`

4. **Environment Variables**
   - Add: `VITE_API_URL` = `https://your-backend.onrender.com`

5. **Deploy**
   - Netlify auto-deploys on push
   - Get URL: `https://your-app.netlify.app`

---

### Option 3: Heroku

#### Backend on Heroku

1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login**
   ```bash
   heroku login
   ```

3. **Create App**
   ```bash
   cd backend
   heroku create your-app-name
   ```

4. **Set Buildpacks**
   ```bash
   heroku buildpacks:set heroku/python
   ```

5. **Create requirements.txt** (already done above)

6. **Create Procfile** (already done above)

7. **Deploy**
   ```bash
   git add .
   git commit -m "Prepare for Heroku deployment"
   git push heroku main
   ```

8. **Get URL**
   ```bash
   heroku info
   # URL: https://your-app-name.herokuapp.com
   ```

**Note:** Heroku free tier is no longer available. Paid plans start at $7/month.

---

### Option 4: AWS (Advanced)

#### Backend on AWS Elastic Beanstalk

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize**
   ```bash
   cd backend
   eb init -p python-3.12 your-app-name
   ```

3. **Create Environment**
   ```bash
   eb create your-env-name
   ```

4. **Deploy**
   ```bash
   eb deploy
   ```

#### Frontend on AWS S3 + CloudFront

1. **Build frontend**
   ```bash
   cd frontend
   npm run build
   ```

2. **Upload to S3**
   - Create S3 bucket
   - Enable static website hosting
   - Upload `dist/` contents

3. **CloudFront** (optional, for CDN)
   - Create CloudFront distribution
   - Point to S3 bucket

---

## Environment Variables

### Backend Environment Variables

Usually not required, but you can add:

```bash
# In Railway/Render/Heroku dashboard:
PORT=8000  # Usually auto-set
PYTHON_VERSION=3.12
```

### Frontend Environment Variables

**Vercel/Netlify:**
```
VITE_API_URL=https://your-backend-api.railway.app
```

**Build-time variables:**
- Vite uses `VITE_` prefix
- Access with `import.meta.env.VITE_API_URL`

---

## CORS Configuration

### Update main.py for Production

```python
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Get frontend URL from environment or use default
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Local dev
        "http://localhost:3000",
        FRONTEND_URL,  # Production frontend
        "https://your-app.vercel.app",  # Add your actual URLs
        "https://your-app.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Complete Deployment Checklist

### Pre-Deployment

- [ ] Create `requirements.txt` in backend
- [ ] Create `Procfile` in backend
- [ ] Update CORS in `main.py` with production URLs
- [ ] Create `.env.production` for frontend
- [ ] Update all API calls to use environment variable
- [ ] Test build locally: `npm run build` in frontend
- [ ] Test backend locally with production settings

### Backend Deployment

- [ ] Choose platform (Railway/Render/Heroku)
- [ ] Connect GitHub repository
- [ ] Set root directory to `backend`
- [ ] Configure build/start commands
- [ ] Set environment variables (if needed)
- [ ] Deploy and get backend URL
- [ ] Test backend API: `https://your-backend.com/docs`

### Frontend Deployment

- [ ] Choose platform (Vercel/Netlify)
- [ ] Connect GitHub repository
- [ ] Set root directory to `frontend`
- [ ] Configure build settings
- [ ] Set `VITE_API_URL` environment variable
- [ ] Deploy and get frontend URL
- [ ] Test frontend: Open in browser

### Post-Deployment

- [ ] Update CORS with actual frontend URL
- [ ] Test all features:
  - [ ] Quick Predict
  - [ ] Full Predict
  - [ ] Score → Features
- [ ] Check browser console for errors
- [ ] Test on mobile devices
- [ ] Update README with production URLs

---

## Quick Start: Railway + Vercel (Recommended)

### Backend (Railway) - 5 minutes

1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Click on the service → Settings
5. Set **Root Directory** to `backend`
6. Set **Start Command** to: `uvicorn main:app --host 0.0.0.0 --port $PORT`
7. Deploy! Get your URL: `https://your-app.railway.app`

### Frontend (Vercel) - 5 minutes

1. Go to [vercel.com](https://vercel.com) and sign up
2. Click "Add New Project" → Import from GitHub
3. Select your repository
4. Configure:
   - **Framework Preset:** Vite
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
5. Add Environment Variable:
   - Key: `VITE_API_URL`
   - Value: `https://your-app.railway.app` (from step 1)
6. Deploy! Get your URL: `https://your-app.vercel.app`

### Update CORS

1. Edit `backend/main.py`
2. Add Vercel URL to `allow_origins`
3. Commit and push (Railway auto-deploys)

Done! 🎉

---

## Troubleshooting

### Backend Issues

**Problem:** Model files too large for deployment
- **Solution:** Use Git LFS or host model files separately
- **Alternative:** Upload model files after deployment via SSH

**Problem:** Build fails on Railway/Render
- **Solution:** Check build logs, ensure `requirements.txt` is correct
- **Check:** Python version compatibility

**Problem:** API returns CORS errors
- **Solution:** Update CORS origins in `main.py` with exact frontend URL
- **Check:** No trailing slashes in URLs

**Problem:** Slow first request (Render free tier)
- **Solution:** Normal for Render free tier (spins down after inactivity)
- **Alternative:** Use Railway or upgrade Render plan

### Frontend Issues

**Problem:** API calls fail in production
- **Solution:** Check `VITE_API_URL` environment variable
- **Check:** CORS is configured correctly
- **Check:** Backend URL is correct (no trailing slash)

**Problem:** Build fails
- **Solution:** Check Node.js version (should be 16+)
- **Check:** All dependencies in `package.json`

**Problem:** Environment variables not working
- **Solution:** Vite requires `VITE_` prefix
- **Check:** Rebuild after adding environment variables

### General Issues

**Problem:** Can't access deployed site
- **Solution:** Check DNS propagation (can take a few minutes)
- **Check:** SSL certificate (usually auto-generated)

**Problem:** Model predictions are slow
- **Solution:** Normal for first request (cold start)
- **Alternative:** Use GPU instances for faster inference

---

## Production Optimizations

### Backend

1. **Add caching** for feature metadata:
   ```python
   from functools import lru_cache
   
   @lru_cache()
   def get_features():
       return {...}
   ```

2. **Add rate limiting** (optional):
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/predict")
   @limiter.limit("10/minute")
   def predict(...):
       ...
   ```

3. **Add logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)
   ```

### Frontend

1. **Add error boundaries** for better error handling
2. **Add loading states** for better UX
3. **Optimize bundle size** (already done by Vite)
4. **Add analytics** (optional)

---

## Monitoring

### Backend Monitoring

- **Railway:** Built-in metrics dashboard
- **Render:** Built-in metrics
- **Heroku:** Heroku Metrics
- **Custom:** Add Sentry for error tracking

### Frontend Monitoring

- **Vercel:** Built-in analytics
- **Netlify:** Built-in analytics
- **Custom:** Add Google Analytics or similar

---

## Security Considerations

1. **CORS:** Only allow your frontend domain
2. **Rate Limiting:** Prevent abuse (optional)
3. **Input Validation:** Already handled by Pydantic
4. **HTTPS:** Always use HTTPS in production (auto-enabled by platforms)
5. **Environment Variables:** Never commit secrets

---

## Cost Estimates

### Free Tier Options

- **Railway:** $5 free credit/month (usually enough)
- **Render:** Free tier (with limitations)
- **Vercel:** Free tier (generous)
- **Netlify:** Free tier (generous)

### Paid Options

- **Heroku:** $7/month (Hobby plan)
- **AWS:** Pay-as-you-go (can be expensive)
- **Railway:** $5-20/month depending on usage

---

## Support

If you encounter issues:

1. Check platform-specific documentation
2. Check build logs in platform dashboard
3. Test locally first
4. Open an issue on GitHub

---

**Happy Deploying! 🚀**

