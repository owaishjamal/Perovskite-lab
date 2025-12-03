# Railway Deployment Fix

## Problem
Build fails during PyTorch installation because it's too large.

## Solution Options

### Option 1: Use Nixpacks Configuration (Recommended)

The `nixpacks.toml` file is already configured to install PyTorch CPU-only separately.

**In Railway Dashboard:**
1. Go to your service settings
2. Make sure **Root Directory** is set to `backend`
3. The `nixpacks.toml` will be automatically detected
4. Redeploy

### Option 2: Use Dockerfile

If Nixpacks still fails:

1. **In Railway Dashboard:**
   - Go to Settings → Build
   - Change **Builder** from "Nixpacks" to "Dockerfile"
   - Save

2. The `Dockerfile` is already created and will:
   - Install PyTorch CPU-only first (faster)
   - Then install other dependencies
   - Start the server

3. Redeploy

### Option 3: Manual Build Command

If both above fail, set custom build command in Railway:

**Build Command:**
```bash
pip install --upgrade pip && pip install torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
```

**Start Command:**
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Quick Fix Steps

1. **Commit the new files:**
   ```bash
   git add backend/nixpacks.toml backend/Dockerfile backend/requirements.txt
   git commit -m "Fix Railway deployment - separate PyTorch installation"
   git push
   ```

2. **In Railway Dashboard:**
   - Go to your service
   - Click "Redeploy" or wait for auto-deploy
   - Monitor build logs

3. **If still failing:**
   - Check build logs for specific error
   - Try switching to Dockerfile builder
   - Check that model files aren't too large (use Git LFS if needed)

## Model Files Size Check

If model files are causing issues:

```bash
# Check sizes
ls -lh backend/*.pt backend/*.joblib

# If too large (>50MB), consider Git LFS
git lfs install
git lfs track "*.pt"
git lfs track "*.joblib"
```

## Expected Build Time

- With CPU-only PyTorch: ~5-8 minutes
- With full PyTorch: ~15-20 minutes (may timeout)

The CPU-only version is sufficient for inference and much faster to install.

