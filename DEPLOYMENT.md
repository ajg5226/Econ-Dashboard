# Deployment Guide

Complete guide for deploying the Recession Prediction Web App to various platforms.

## Table of Contents

1. [Streamlit Community Cloud](#streamlit-community-cloud)
2. [Docker Deployment](#docker-deployment)
3. [Heroku Deployment](#heroku-deployment)
4. [AWS Deployment](#aws-deployment)
5. [Scheduler Setup](#scheduler-setup)
6. [Environment Variables](#environment-variables)
7. [Troubleshooting](#troubleshooting)

## Streamlit Community Cloud

### Prerequisites

- GitHub account
- Repository pushed to GitHub
- Streamlit Community Cloud account (free)

### Step-by-Step Deployment

1. **Prepare your repository**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Create Streamlit Cloud app**
   - Go to https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy!"

3. **Configure secrets**
   - Go to app settings
   - Click "Secrets"
   - Add the following:
     ```
     FRED_API_KEY=your_fred_api_key_here
     SECRET_KEY=your_secret_key_here
     ```

4. **Set up scheduler**
   - Streamlit Cloud doesn't support background processes
   - Use GitHub Actions (see Scheduler Setup section)

### GitHub Actions Scheduler

Create `.github/workflows/scheduler.yml`:

```yaml
name: Weekly Data Refresh

on:
  schedule:
    - cron: '0 3 * * 0'  # Sunday 3 AM UTC
  workflow_dispatch:  # Allows manual trigger

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run update job
        env:
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
        run: |
          python scheduler/update_job.py
      
      - name: Commit and push data
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add data/
          git commit -m "Update data $(date +'%Y-%m-%d')" || exit 0
          git push
```

Add `FRED_API_KEY` to GitHub repository secrets:
- Go to repository → Settings → Secrets → Actions
- Click "New repository secret"
- Name: `FRED_API_KEY`
- Value: Your FRED API key

## Docker Deployment

### Build and Run

1. **Build the image**
   ```bash
   docker build -t recession-web-app .
   ```

2. **Run the container**
   ```bash
   docker run -d \
     --name recession-app \
     -p 8501:8501 \
     -e FRED_API_KEY=your_key \
     -e SECRET_KEY=your_secret \
     -v $(pwd)/data:/app/data \
     recession-web-app
   ```

3. **Access the app**
   - Open http://localhost:8501

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8501:8501"
    environment:
      - FRED_API_KEY=${FRED_API_KEY}
      - SECRET_KEY=${SECRET_KEY}
    volumes:
      - ./data:/app/data
    restart: unless-stopped
  
  scheduler:
    build: .
    command: python scheduler/update_job.py
    environment:
      - FRED_API_KEY=${FRED_API_KEY}
    volumes:
      - ./data:/app/data
    restart: "no"  # Run once, then exit
```

Run with:
```bash
docker-compose up -d
```

## Heroku Deployment

### Prerequisites

- Heroku account
- Heroku CLI installed

### Deployment Steps

1. **Create Procfile**
   ```
   web: streamlit run app/main.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create app**
   ```bash
   heroku create recession-web-app
   ```

3. **Set environment variables**
   ```bash
   heroku config:set FRED_API_KEY=your_key
   heroku config:set SECRET_KEY=your_secret
   ```

4. **Deploy**
   ```bash
   git push heroku main
   ```

5. **Set up scheduler**
   - Use Heroku Scheduler addon
   - Or use external cron service

### Heroku Scheduler

1. Install addon:
   ```bash
   heroku addons:create scheduler:standard
   ```

2. Configure job:
   - Go to Heroku dashboard
   - Open Scheduler addon
   - Add job: `python scheduler/update_job.py`
   - Set frequency: Weekly

## AWS Deployment

### Option 1: Elastic Beanstalk

1. **Install EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize**
   ```bash
   eb init -p python-3.9 recession-web-app
   ```

3. **Create environment**
   ```bash
   eb create recession-web-app-env
   ```

4. **Set environment variables**
   ```bash
   eb setenv FRED_API_KEY=your_key SECRET_KEY=your_secret
   ```

5. **Deploy**
   ```bash
   eb deploy
   ```

### Option 2: EC2 with Docker

1. **Launch EC2 instance**
   - Use Ubuntu 20.04 LTS
   - Open port 8501 in security group

2. **Install Docker**
   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose
   ```

3. **Clone repository**
   ```bash
   git clone <your-repo-url>
   cd recession-web-app
   ```

4. **Set environment variables**
   ```bash
   export FRED_API_KEY=your_key
   export SECRET_KEY=your_secret
   ```

5. **Run with Docker**
   ```bash
   docker-compose up -d
   ```

6. **Set up cron for scheduler**
   ```bash
   crontab -e
   # Add: 0 3 * * 0 cd /path/to/app && docker-compose run --rm scheduler python scheduler/update_job.py
   ```

## Scheduler Setup

### Unix/Linux Cron

1. **Make script executable**
   ```bash
   chmod +x scheduler/run_scheduler.sh
   ```

2. **Edit crontab**
   ```bash
   crontab -e
   ```

3. **Add cron job**
   ```
   # Weekly on Sunday at 3 AM
   0 3 * * 0 /path/to/recession-web-app/scheduler/run_scheduler.sh
   ```

4. **Test the job**
   ```bash
   /path/to/recession-web-app/scheduler/run_scheduler.sh
   ```

### Windows Task Scheduler

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Weekly, Sunday, 3:00 AM
4. Set action: Start a program
5. Program: `python`
6. Arguments: `scheduler/update_job.py`
7. Start in: Project directory path

### External Services

- **Cron-job.org**: Free web-based cron service
- **EasyCron**: Paid service with more features
- **GitHub Actions**: Free for public repos

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `FRED_API_KEY` | FRED API key | `abc123...` |
| `SECRET_KEY` | Secret for cookie signing | `random_string` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SCHEDULER_INTERVAL` | Scheduler frequency | `weekly` |
| `PREDICTION_HORIZON` | Prediction horizon (months) | `6` |
| `TRAIN_END_DATE` | Training data end date | `2015-12-31` |

### Generating Secret Key

```python
import secrets
print(secrets.token_hex(32))
```

## Troubleshooting

### App won't start

- Check logs: `heroku logs --tail` or `docker logs <container>`
- Verify environment variables are set
- Check port is not already in use

### Scheduler not running

- Verify cron job is active: `crontab -l`
- Check scheduler logs: `data/logs/scheduler_*.log`
- Test manually: `python scheduler/update_job.py`

### Data not updating

- Check FRED API key is valid
- Verify scheduler has write permissions to `data/` directory
- Check disk space: `df -h`

### Authentication issues

- Verify `app/config.yaml` exists
- Check file permissions
- Try resetting password

### Import errors

- Install dependencies: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.9+)
- Verify all files are in correct directories

## Monitoring

### Health Checks

The app includes a health check endpoint:
- Streamlit: `/_stcore/health`
- Docker: Configured in Dockerfile

### Logging

- Application logs: Streamlit console output
- Scheduler logs: `data/logs/scheduler_*.log`
- Error logs: Check platform-specific logs

### Metrics to Monitor

- Data freshness (last update time)
- Model performance (AUC scores)
- API response times
- Error rates
- User activity

## Backup and Recovery

### Backup Strategy

1. **Data files**: `data/predictions.csv`, `data/indicators.csv`
2. **Models**: `data/models/*.pkl`
3. **User config**: `app/config.yaml`
4. **Logs**: `data/logs/`

### Automated Backups

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/backups/recession-app"
DATE=$(date +%Y%m%d)
tar -czf "$BACKUP_DIR/backup-$DATE.tar.gz" data/ app/config.yaml
```

### Recovery

1. Stop the application
2. Restore backup files
3. Restart application
4. Verify data is current

## Security Checklist

- [ ] Changed default admin password
- [ ] Set strong SECRET_KEY
- [ ] FRED_API_KEY is secure (not in code)
- [ ] HTTPS enabled (Streamlit Cloud provides)
- [ ] User credentials are hashed
- [ ] `.env` and `config.yaml` in `.gitignore`
- [ ] Dependencies are up to date
- [ ] Regular security updates

## Performance Optimization

### Caching

- Streamlit caching is already configured
- Adjust TTL in `app/utils/cache_manager.py` if needed

### Database Migration

For better performance with multiple users, consider migrating from CSV to:
- SQLite (simple)
- PostgreSQL (production)
- Redis (caching)

### Scaling

- Use load balancer for multiple instances
- Consider database for shared state
- Use CDN for static assets

---

**Need help?** Check the main README or open an issue.

