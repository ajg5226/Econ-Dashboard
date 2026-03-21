# Recession Prediction Web Application

A secure, login-enabled web application for viewing and managing recession probability forecasts. Built with Streamlit and deployed on Streamlit Community Cloud.

## 🎯 Features

- **Secure Authentication**: Login system with role-based access (admin/viewer)
- **Interactive Dashboard**: Real-time recession probability visualizations
- **Indicator Exploration**: Explore 45+ economic indicators and their engineered features
- **Model Performance**: View detailed metrics and confusion matrices
- **Automated Updates**: Background scheduler for data refresh and model retraining
- **Data Downloads**: Export predictions, reports, and charts

## 📋 Prerequisites

- Python 3.11+
- FRED API key (free from https://fred.stlouisfed.org/)
- Git (for deployment)

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd recession_web_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   export FRED_API_KEY='your_fred_api_key_here'
   ```

4. **Run the scheduler (first time)**
   ```bash
   python scheduler/update_job.py
   ```
   This will fetch data, train models, and generate predictions.

5. **Start the Streamlit app**
   ```bash
   streamlit run app/main.py
   ```

6. **Access the app**
   - Open http://localhost:8501 in your browser
   - Login with default credentials:
     - Username: `admin`
     - Password: `admin123`
     - **Change the password immediately after first login!**

## 📁 Project Structure

```
recession_web_app/
├── app/
│   ├── main.py                 # Streamlit entry point
│   ├── auth.py                 # Authentication module
│   ├── config.yaml             # User credentials (hashed)
│   ├── pages/
│   │   ├── dashboard.py        # Main dashboard
│   │   ├── indicators.py       # Indicator exploration
│   │   ├── model_performance.py # Model metrics
│   │   └── settings.py         # Admin settings
│   └── utils/
│       ├── data_loader.py      # Data persistence
│       ├── cache_manager.py    # Caching utilities
│       └── plotting.py         # Chart utilities
├── recession_engine/            # Core prediction engine
│   ├── data_acquisition.py
│   ├── ensemble_model.py
│   └── run_recession_engine.py
├── scheduler/
│   ├── update_job.py          # Background refresh script
│   └── scheduler_config.py    # Scheduler settings
├── data/                       # Persistent storage (created automatically)
│   ├── predictions.csv
│   ├── indicators.csv
│   ├── models/
│   └── reports/
├── tests/                      # Unit tests
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

## 🔐 Authentication

### Default Credentials

- **Username**: `admin`
- **Password**: `admin123`

**⚠️ IMPORTANT**: Change the default password immediately after first login!

### Adding Users

1. Log in as admin
2. Go to Settings page
3. Expand "User Management" section
4. Fill in the form and click "Add User"

Users are stored in `app/config.yaml` with hashed passwords.

## ⚙️ Configuration

### Environment Variables

Create a `.env` file (or set environment variables):

```bash
FRED_API_KEY=your_fred_api_key_here
SECRET_KEY=your_secret_key_here  # For cookie signing
SCHEDULER_INTERVAL=weekly
PREDICTION_HORIZON=6
TRAIN_END_DATE=2015-12-31
```

### Scheduler Configuration

The scheduler can run in several ways:

#### Option 1: Cron Job (Recommended for Linux/Mac)

```bash
# Edit crontab
crontab -e

# Add line for weekly updates (Sunday 3 AM)
0 3 * * 0 /path/to/recession_web_app/scheduler/run_scheduler.sh
```

#### Option 2: GitHub Actions (For Streamlit Cloud)

Create `.github/workflows/scheduler.yml`:

```yaml
name: Weekly Data Refresh

on:
  schedule:
    - cron: '0 3 * * 0'  # Sunday 3 AM UTC
  workflow_dispatch:  # Manual trigger

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python scheduler/update_job.py
        env:
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
```

## 🌐 Deployment

### Streamlit Community Cloud

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `streamlit_app.py`
   - Add secrets:
     - `FRED_API_KEY`: Your FRED API key
     - `SECRET_KEY`: Random secret for cookies

3. **Set up scheduler**
   - Use GitHub Actions (see above) or
   - Use an external server with cron

### Docker Deployment

1. **Build the image**
   ```bash
   docker build -t recession-web-app .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 \
     -e FRED_API_KEY=your_key \
     -e SECRET_KEY=your_secret \
     -v $(pwd)/data:/app/data \
     recession-web-app
   ```

3. **Access the app**
   - Open http://localhost:8501

### Heroku Deployment

1. **Create Procfile**
   ```
   web: streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0
   worker: python scheduler/update_job.py
   ```

2. **Deploy**
   ```bash
   heroku create recession-web-app
   heroku config:set FRED_API_KEY=your_key
   heroku config:set SECRET_KEY=your_secret
   git push heroku main
   ```

## 📊 Usage Guide

### Dashboard Page

- View latest recession probability
- Adjust date range and prediction horizon
- Download data and reports
- See risk level classification

### Indicators Page

- Explore individual economic indicators
- View raw values and engineered features
- Filter by category (Leading/Coincident/Lagging)
- See statistics and recent trends

### Model Performance Page

- Compare model metrics (AUC, Precision, Recall, F1)
- View confusion matrices
- See training/test split information

### Settings Page (Admin Only)

- Trigger manual data refresh
- Configure model parameters
- Manage users
- View system information

## 🧪 Testing

Run unit tests:

```bash
python -m pytest tests/
```

Or run specific test files:

```bash
python -m pytest tests/test_data_acquisition.py
python -m pytest tests/test_ensemble_model.py
python -m pytest tests/test_scheduler.py
```

## 🔧 Troubleshooting

### "No prediction data available"

- Run the scheduler: `python scheduler/update_job.py`
- Check that `FRED_API_KEY` is set
- Verify data directory exists: `data/predictions.csv`

### "FRED_API_KEY not set"

- Set environment variable: `export FRED_API_KEY='your_key'`
- Or create `.env` file with the key

### "Authentication failed"

- Check `app/config.yaml` exists
- Verify password hash is correct
- Try resetting password in Settings (admin only)

### "Module not found" errors

- Install dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.11+)

### Scheduler not running

- Check cron job is configured: `crontab -l`
- Verify script is executable: `chmod +x scheduler/run_scheduler.sh`
- Check logs: `data/logs/scheduler_*.log`

## 📝 Maintenance

### Regular Tasks

1. **Weekly**: Scheduler automatically refreshes data
2. **Monthly**: Review model performance metrics
3. **Quarterly**: Retrain models with new data
4. **As needed**: Add/remove users, update passwords

### Data Backup

Backup the following directories:
- `data/predictions.csv`
- `data/indicators.csv`
- `data/models/` (model artifacts)
- `app/config.yaml` (user credentials)

### Logs

Check scheduler logs:
```bash
tail -f data/logs/scheduler_*.log
```

## 🔒 Security Best Practices

1. **Change default passwords** immediately
2. **Use strong passwords** for admin accounts
3. **Rotate API keys** periodically
4. **Keep dependencies updated**: `pip install --upgrade -r requirements.txt`
5. **Never commit** `.env` or `app/config.yaml` to git
6. **Use HTTPS** in production (Streamlit Cloud provides this)

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `data/logs/`
3. Check GitHub issues (if applicable)

## 📜 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Federal Reserve Economic Data (FRED) for economic indicators
- Streamlit team for the web framework
- scikit-learn, XGBoost, and other ML libraries

---

**Built with ❤️ for economic forecasting**

