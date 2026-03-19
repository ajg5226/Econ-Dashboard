"""
Streamlit Cloud entry point
Points to the main app module
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run main app
from app.main import main

if __name__ == "__main__":
    main()

