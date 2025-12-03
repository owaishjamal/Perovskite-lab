# 🚀 Quick Deployment Checklist

Use this checklist to ensure everything is ready for deployment.

## Pre-Deployment

### Backend Files
- [x] `requirements.txt` created
- [x] `Procfile` created
- [x] `runtime.txt` created (Python version)
- [x] CORS updated in `main.py` to support production URLs
- [x] `.gitignore` configured

### Frontend Files
- [x] `config.ts` created for API URL
- [x] All API calls updated to use `API_URL` from config
- [x] Environment variable support added (`VITE_API_URL`)

### Testing
- [ ] Test backend locally: `uvicorn main:app --reload`
- [ ] Test frontend build: `npm run build`
- [ ] Test frontend locally: `npm run dev`
- [ ] Verify all features work:
  - [ ] Quick Predict
  - [ ] Full Predict
  - [ ] Score → Features

## Backend Deployment (Choose One)

### Option A: Railway (Recommended)
- [ ] Sign up at railway.app
- [ ] Create new project from GitHub
- [ ] Set root directory: `backend`
- [ ] Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] Deploy and get URL: `https://your-app.railway.app`
- [ ] Test API: Visit `https://your-app.railway.app/docs`

### Option B: Render
- [ ] Sign up at render.com
- [ ] Create new web service
- [ ] Set root directory: `backend`
- [ ] Set build command: `pip install -r requirements.txt`
- [ ] Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- [ ] Deploy and get URL: `https://your-app.onrender.com`

### Option C: Heroku
- [ ] Install Heroku CLI
- [ ] Login: `heroku login`
- [ ] Create app: `heroku create your-app-name`
- [ ] Deploy: `git push heroku main`
- [ ] Get URL: `https://your-app-name.herokuapp.com`

## Frontend Deployment (Choose One)

### Option A: Vercel (Recommended)
- [ ] Sign up at vercel.com
- [ ] Import project from GitHub
- [ ] Set framework: Vite
- [ ] Set root directory: `frontend`
- [ ] Set build command: `npm run build`
- [ ] Set output directory: `dist`
- [ ] Add environment variable: `VITE_API_URL` = your backend URL
- [ ] Deploy and get URL: `https://your-app.vercel.app`

### Option B: Netlify
- [ ] Sign up at netlify.com
- [ ] Import from Git
- [ ] Set base directory: `frontend`
- [ ] Set build command: `npm run build`
- [ ] Set publish directory: `frontend/dist`
- [ ] Add environment variable: `VITE_API_URL` = your backend URL
- [ ] Deploy and get URL: `https://your-app.netlify.app`

## Post-Deployment

### Update CORS
- [ ] Edit `backend/main.py`
- [ ] Add frontend URL to `CORS_ORIGINS` list
- [ ] Commit and push (auto-deploys)

### Final Testing
- [ ] Open frontend URL in browser
- [ ] Test Quick Predict feature
- [ ] Test Full Predict feature
- [ ] Test Score → Features feature
- [ ] Check browser console for errors
- [ ] Test on mobile device (optional)

### Documentation
- [ ] Update README.md with production URLs (optional)
- [ ] Add deployment badges to README (optional)

## Troubleshooting

If something doesn't work:
1. Check platform build logs
2. Verify environment variables are set
3. Check CORS configuration
4. Test API endpoint directly: `https://your-backend.com/docs`
5. Check browser console for errors

---

**Ready to deploy?** Follow the detailed guide in [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

