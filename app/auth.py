"""
Authentication module for Streamlit app
Uses streamlit-authenticator for secure login

Architecture for multi-page apps:
- main.py calls `render_login()` to show the login widget (once)
- Each page calls `check_authentication()` which only reads session state
- This avoids duplicate widget key errors on Streamlit Cloud
"""

import streamlit as st
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).parent / "config.yaml"


def load_config():
    """Load authentication configuration from YAML file"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as file:
            return yaml.load(file, Loader=SafeLoader)
    else:
        default_config = {
            'credentials': {
                'usernames': {
                    'admin': {
                        'email': 'admin@example.com',
                        'failed_login_attempts': 0,
                        'logged_in': False,
                        'name': 'Administrator',
                        'password': 'admin123'
                    }
                }
            },
            'cookie': {
                'expiry_days': 30,
                'key': 'recession_prediction_app',
                'name': 'recession_app_cookie'
            },
            'preauthorized': {
                'emails': []
            }
        }
        save_config(default_config)
        return default_config


def save_config(config):
    """Save authentication configuration to YAML file"""
    with open(CONFIG_FILE, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    logger.info(f"Saved authentication config to {CONFIG_FILE}")


def get_authenticator():
    """
    Initialize and return streamlit-authenticator object.
    Caches in session state to avoid creating duplicate widget instances.
    """
    if '_authenticator' not in st.session_state:
        config = load_config()
        authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days'],
        )
        st.session_state['_authenticator'] = authenticator
        st.session_state['_auth_config'] = config
    return st.session_state['_authenticator'], st.session_state['_auth_config']


def render_login():
    """
    Render the login widget. Call this ONLY from main.py.

    In Streamlit multi-page apps, main.py runs before every page.
    The login widget must be rendered exactly once per script run
    to avoid duplicate widget key errors.

    Returns:
        Tuple of (authenticated: bool, username: str, name: str)
    """
    try:
        authenticator, config = get_authenticator()

        # Render login widget (this also checks the auth cookie)
        authenticator.login(location='main', key='Login')

        authentication_status = st.session_state.get("authentication_status")
        name = st.session_state.get("name")
        username = st.session_state.get("username")

        if authentication_status is True and username and name:
            return True, username, name
        elif authentication_status is False:
            st.error('Username/password is incorrect')
            return False, None, None
        else:
            st.warning('Please enter your username and password')
            return False, None, None

    except Exception as e:
        logger.error(f"Error in authentication: {str(e)}")
        st.error(f"Authentication error: {str(e)}")
        return False, None, None


def check_authentication():
    """
    Check if the user is authenticated by reading session state.
    Call this from individual pages (dashboard.py, indicators.py, etc.).

    Does NOT render any widgets — avoids duplicate key errors.

    Returns:
        Tuple of (authenticated: bool, username: str, name: str)
    """
    authentication_status = st.session_state.get("authentication_status")
    name = st.session_state.get("name")
    username = st.session_state.get("username")

    if authentication_status is True and username and name:
        return True, username, name

    # Not authenticated — show message and let main.py handle login
    st.warning("Please log in from the main page.")
    return False, None, None


def get_user_role(username: str) -> str:
    """Get user role (admin or viewer)"""
    if username == 'admin':
        return 'admin'
    config = load_config()
    user_data = config.get('credentials', {}).get('usernames', {}).get(username, {})
    return user_data.get('role', 'viewer')


def is_admin(username: str) -> bool:
    """Check if user is an admin"""
    return get_user_role(username) == 'admin'


def register_user(username: str, name: str, email: str, password: str, role: str = 'viewer'):
    """Register a new user (admin only)"""
    if not username or not name or not email or not password:
        raise ValueError("All fields (username, name, email, password) are required")

    config = load_config()

    if 'credentials' not in config:
        config['credentials'] = {}
    if 'usernames' not in config['credentials']:
        config['credentials']['usernames'] = {}

    if username in config['credentials']['usernames']:
        raise ValueError(f"Username {username} already exists")

    config['credentials']['usernames'][username] = {
        'email': email,
        'failed_login_attempts': 0,
        'logged_in': False,
        'name': name,
        'password': password,
        'role': role
    }

    save_config(config)
    # Clear cached authenticator so it reloads with new user
    st.session_state.pop('_authenticator', None)
    st.session_state.pop('_auth_config', None)
    logger.info(f"Registered new user: {username}")


def logout():
    """Handle user logout"""
    authenticator, _ = get_authenticator()
    authenticator.logout('Logout', location='sidebar', key='Logout')
