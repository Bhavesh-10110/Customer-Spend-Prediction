# Streamlit Cloud Deployment Guide

## Step-by-Step Deployment Instructions

### 1. **Prepare Your Repository** ✅
Your code is already pushed to: `https://github.com/Bhavesh-10110/Customer-Spend-Prediction`

### 2. **Update Data Path (IMPORTANT)**
Since Streamlit Cloud won't have access to your local dataset path, you need to modify `knn_streamlit_app.py`:

Replace:
```python
DATA_PATH = Path(r"C:\Clg\TekWorks\Datasets\task1_dataset.csv")
```

With:
```python
DATA_PATH = Path("task1_dataset.csv")  # Place dataset in the same directory
# OR use a URL if hosting the dataset online
# DATA_PATH = "https://raw.githubusercontent.com/Bhavesh-10110/your-repo/main/task1_dataset.csv"
```

### 3. **Deploy to Streamlit Cloud**

1. Go to [Streamlit Cloud](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select:
   - Repository: `Bhavesh-10110/Customer-Spend-Prediction`
   - Branch: `main`
   - Main file path: `knn_streamlit_app.py`
5. Click **"Deploy"**

### 4. **Configure Secrets (Optional)**
If you need environment variables, create `.streamlit/secrets.toml` in your repo:
```toml
# .streamlit/secrets.toml
api_key = "your-api-key"
database_url = "your-database-url"
```

### 5. **Your App is Live!**
Your app will be available at: `https://customer-spend-prediction-[your-username].streamlit.app`

---

## Alternative Deployment Options

### Deploy to Heroku

1. Create `Procfile`:
```
web: streamlit run knn_streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Create `runtime.txt`:
```
python-3.11.5
```

3. Deploy:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

### Deploy to AWS

1. Use EC2 instance with Ubuntu
2. Install Python and dependencies
3. Use systemd or supervisor to run Streamlit
4. Configure nginx as reverse proxy

### Deploy to Google Cloud

1. Create Cloud Run service
2. Push Docker image to Container Registry
3. Deploy with: `gcloud run deploy`

---

## Important Notes

⚠️ **Dataset Management:**
- Option 1: Upload `task1_dataset.csv` to your GitHub repository
- Option 2: Host the dataset on a cloud storage (AWS S3, Google Drive, etc.)
- Option 3: Load data from a database

⚠️ **Memory & Computation:**
- Streamlit Cloud free tier has limited resources
- Model training is cached, so it only trains once per session
- Consider using a more powerful instance if needed

✅ **Best Practices:**
- Keep dataset path relative or use URLs
- Use `@st.cache_resource` for expensive operations (already implemented)
- Monitor app performance in Streamlit Cloud dashboard
- Set up GitHub Actions for CI/CD (optional)

---

## Troubleshooting

### App won't load / FileNotFoundError
**Solution**: Make sure dataset path is correct and file exists in repository or accessible via URL

### Slow predictions
**Solution**: Model is cached after first run. If still slow, consider:
- Reducing polynomial features
- Using a subset of features
- Upgrading to paid Streamlit tier

### ModuleNotFoundError
**Solution**: Ensure all dependencies are in `requirements.txt`
```bash
pip freeze > requirements.txt
```

---

## Monitoring & Updates

After deployment:
1. View logs: Streamlit Cloud → App → Logs
2. Deploy updates: Simply push new commits to `main` branch
3. Streamlit Cloud auto-deploys when you push!

---

## Support

For issues:
- Check Streamlit documentation: https://docs.streamlit.io
- Review app logs on Streamlit Cloud dashboard
- Open an issue on GitHub: https://github.com/Bhavesh-10110/Customer-Spend-Prediction/issues
