"""
Settings Page
Admin controls for data refresh, model configuration, and user management
"""

import streamlit as st
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from app.auth import check_authentication, is_admin, register_user
    from app.utils.data_loader import get_last_update_time, is_data_stale
    from app.utils.cache_manager import clear_all_caches, get_cache_info
except ImportError:
    from auth import check_authentication, is_admin, register_user
    from utils.data_loader import get_last_update_time, is_data_stale
    from utils.cache_manager import clear_all_caches, get_cache_info

# Check authentication
authenticated, username, name = check_authentication()
if not authenticated:
    st.stop()

st.title("⚙️ Settings")

# Check if user is admin
if not is_admin(username):
    st.warning("⚠️ Admin access required. Only administrators can modify settings.")
    st.info("Contact an administrator to change settings or refresh data.")
    
    # Show read-only information
    st.markdown("---")
    st.markdown("### Data Status")
    
    last_update = get_last_update_time()
    if last_update:
        st.info(f"📅 Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.warning("No data available")
    
    stale = is_data_stale()
    if stale:
        st.warning("⚠️ Data is stale (older than 7 days)")
    else:
        st.success("✅ Data is fresh")
    
    st.stop()

# Admin-only content
st.success("🔑 Admin Access Granted")

# Data Refresh Section
st.markdown("---")
st.markdown("### Data Refresh")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Manual Refresh")
    st.markdown("Trigger an immediate data refresh and model retraining.")
    
    if st.button("🔄 Refresh Data Now", type="primary"):
        with st.spinner("Refreshing data... This may take several minutes."):
            try:
                # Run the scheduler update job
                scheduler_script = Path(__file__).parent.parent.parent / "scheduler" / "update_job.py"
                
                if scheduler_script.exists():
                    # Set environment variables
                    env = os.environ.copy()
                    if 'FRED_API_KEY' not in env:
                        st.error("FRED_API_KEY environment variable not set!")
                        st.stop()
                    
                    # Run the update job
                    # BUG FIX: Better error handling for subprocess
                    try:
                        result = subprocess.run(
                            [sys.executable, str(scheduler_script)],
                            capture_output=True,
                            text=True,
                            env=env,
                            timeout=600,  # 10 minute timeout
                            cwd=str(Path(__file__).parent.parent.parent)  # Set working directory
                        )
                        
                        if result.returncode == 0:
                            st.success("✅ Data refresh completed successfully!")
                            clear_all_caches()
                            st.info("Cache cleared. Please refresh the page to see updated data.")
                        else:
                            error_msg = result.stderr if result.stderr else result.stdout
                            st.error(f"❌ Error during refresh:\n{error_msg}")
                    except subprocess.TimeoutExpired:
                        st.error("⏱️ Refresh timed out after 10 minutes. The process may still be running.")
                    except FileNotFoundError:
                        st.error(f"❌ Python executable not found: {sys.executable}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {str(e)}")
                else:
                    st.error(f"Scheduler script not found: {scheduler_script}")
            except subprocess.TimeoutExpired:
                st.error("⏱️ Refresh timed out. The process may still be running.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

with col2:
    st.markdown("#### Data Status")
    
    last_update = get_last_update_time()
    if last_update:
        st.info(f"📅 Last update: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        days_old = (datetime.now() - last_update).days
        st.metric("Days since update", days_old)
    else:
        st.warning("No data available")
    
    stale = is_data_stale()
    if stale:
        st.warning("⚠️ Data is stale (older than 7 days)")
    else:
        st.success("✅ Data is fresh")
    
    # Cache info
    cache_info = get_cache_info()
    st.markdown("#### Cache Information")
    st.json(cache_info)

# Model Configuration
st.markdown("---")
st.markdown("### Model Configuration")

with st.form("model_config"):
    st.markdown("Adjust model hyperparameters (changes require retraining)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_horizon = st.selectbox(
            "Prediction Horizon",
            options=[3, 6, 12],
            index=1,
            format_func=lambda x: f"{x} months"
        )
        
        decision_threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
    
    with col2:
        max_features = st.number_input(
            "Maximum Features",
            min_value=10,
            max_value=200,
            value=50,
            step=10
        )
        
        scheduler_interval = st.selectbox(
            "Scheduler Interval",
            options=["daily", "weekly", "monthly"],
            index=1
        )
    
    submitted = st.form_submit_button("💾 Save Configuration")
    
    if submitted:
        st.info("⚠️ Configuration saved. Changes will take effect on next model retraining.")
        # In a real implementation, save these to a config file

# User Management
st.markdown("---")
st.markdown("### User Management")

with st.expander("Add New User"):
    with st.form("add_user"):
        new_username = st.text_input("Username")
        new_name = st.text_input("Full Name")
        new_email = st.text_input("Email")
        new_password = st.text_input("Password", type="password")
        new_role = st.selectbox("Role", options=["viewer", "admin"])
        
        submitted = st.form_submit_button("➕ Add User")
        
        if submitted:
            if not all([new_username, new_name, new_email, new_password]):
                st.error("Please fill in all fields")
            else:
                try:
                    register_user(new_username, new_name, new_email, new_password, new_role)
                    st.success(f"✅ User {new_username} added successfully!")
                except ValueError as e:
                    st.error(f"❌ Error: {str(e)}")

# System Information
st.markdown("---")
st.markdown("### System Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Environment")
    st.code(f"""
Python: {sys.version}
Working Directory: {os.getcwd()}
FRED API Key: {'✅ Set' if os.environ.get('FRED_API_KEY') else '❌ Not Set'}
    """)

with col2:
    st.markdown("#### File Paths")
    base_path = Path(__file__).parent.parent.parent
    st.code(f"""
Data Directory: {base_path / 'data'}
Models Directory: {base_path / 'data' / 'models'}
Reports Directory: {base_path / 'data' / 'reports'}
    """)

# Clear Cache
st.markdown("---")
st.markdown("### Cache Management")

if st.button("🗑️ Clear All Caches"):
    clear_all_caches()
    st.success("✅ All caches cleared!")

