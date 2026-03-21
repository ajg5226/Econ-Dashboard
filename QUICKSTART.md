# Quick Start Guide

Get the Recession Prediction Web App running in 5 minutes!

## Prerequisites

- Python 3.11 or higher
- FRED API key (free from https://fred.stlouisfed.org/)

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Run setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Edit .env and add your FRED_API_KEY
nano .env  # or use your preferred editor

# Fetch initial data and train models
python scheduler/update_job.py

# Start the web app
streamlit run app/main.py
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export FRED_API_KEY='your_api_key_here'

# Create data directories
mkdir -p data/models data/reports data/logs

# Fetch initial data
python scheduler/update_job.py

# Start the app
streamlit run app/main.py
```

## First Login

1. Open http://localhost:8501 in your browser
2. Login with:
   - **Username**: `admin`
   - **Password**: `admin123`
3. **IMPORTANT**: Go to Settings → User Management and change the admin password!

## What's Next?

1. **Explore the Dashboard**: View current recession probabilities
2. **Check Indicators**: Explore economic indicators
3. **Review Performance**: See model metrics
4. **Set Up Scheduler**: Configure automatic data updates

## Troubleshooting

### "No prediction data available"
Run: `python scheduler/update_job.py`

### "FRED_API_KEY not set"
Set it: `export FRED_API_KEY='your_key'`

### Import errors
Install dependencies: `pip install -r requirements.txt`

## Need Help?

- See `README_WEB_APP.md` for detailed documentation
- See `DEPLOYMENT.md` for deployment instructions
- Check logs in `data/logs/`

---

**Ready to predict recessions? Let's go! 🚀**

