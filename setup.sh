#!/bin/bash
# Setup script for Recession Prediction Web App

echo "Setting up Recession Prediction Web App..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directories
echo "Creating data directories..."
mkdir -p data/models data/reports data/logs

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your FRED_API_KEY"
fi

# Make scheduler script executable
chmod +x scheduler/run_scheduler.sh

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your FRED_API_KEY"
echo "2. Run: python scheduler/update_job.py (to fetch initial data)"
echo "3. Run: streamlit run app/main.py (to start the web app)"
echo ""
echo "Default login:"
echo "  Username: admin"
echo "  Password: admin123"
echo "  ⚠️  Change this immediately after first login!"

