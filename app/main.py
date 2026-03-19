"""
Main Streamlit application entry point
Handles authentication and routing to pages
"""

import streamlit as st
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

try:
    from app.auth import check_authentication, logout, get_user_role
except ImportError:
    # Fallback for direct execution
    from auth import check_authentication, logout, get_user_role

# Page configuration
st.set_page_config(
    page_title="Recession Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Check authentication
    authenticated, username, name = check_authentication()
    
    if not authenticated:
        st.stop()
    
    # User is authenticated - show main app
    st.sidebar.title("📊 Recession Prediction")
    st.sidebar.markdown(f"**Welcome, {name}!**")
    
    # Logout button
    logout()
    
    # Display user role
    role = get_user_role(username)
    if role == 'admin':
        st.sidebar.success("🔑 Admin Access")
    else:
        st.sidebar.info("👤 Viewer Access")
    
    # Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")
    
    # Main header
    st.markdown('<p class="main-header">📊 Recession Prediction Dashboard</p>', unsafe_allow_html=True)
    
    # Show dashboard by default (Streamlit will handle page routing)
    # Pages are automatically discovered from app/pages/ directory


if __name__ == "__main__":
    main()

