# Railway Deployment Troubleshooting

## Common Issues and Solutions

### Issue: Build Fails with PyTorch

**Solution:** Use CPU-only PyTorch (already configured in requirements.txt)

If build still fails, try this alternative requirements.txt:

```txt
fastapi==0.123.4
uvicorn[standard]==0.38.0
torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pandas==2.3.3
numpy==2.3.5
joblib==1.5.2
pydantic==2.12.5
scikit-learn==1.7.2
python-multipart==0.0.9
```

### Issue: Model Files Too Large

If model files (.pt, .joblib) are causing issues:

1. **Option 1:** Use Git LFS
   ```bash
   git lfs install
   git lfs track "*.pt"
   git lfs track "*.joblib"
   git add .gitattributes
   git add *.pt *.joblib
   git commit -m "Add model files with LFS"
   ```

2. **Option 2:** Upload models after deployment
   - Deploy without model files
   - Use Railway's volume or external storage
   - Download models on first startup

### Issue: Build Timeout

**Solution:** Railway has a 10-minute build timeout. If PyTorch installation takes too long:

1. Use CPU-only PyTorch (already done)
2. Consider using a lighter model
3. Split requirements into base and optional

### Railway Configuration

Make sure in Railway dashboard:
- **Root Directory:** `backend`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Builder:** Nixpacks (auto-detected)

### Alternative: Use Docker

If Nixpacks continues to fail, create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE $PORT

# Start command
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
```

Then in Railway, set builder to Dockerfile.

