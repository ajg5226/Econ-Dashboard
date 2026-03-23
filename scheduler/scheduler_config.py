"""
Scheduler/runtime configuration utilities.

Defines defaults and persistence helpers used by both:
- scheduler/update_job.py (runtime behavior)
- app/pages/settings.py (admin configuration UI)
"""

import json
from pathlib import Path

# Scheduler interval options
SCHEDULER_INTERVALS = {
    'daily': {
        'cron': '0 3 * * *',  # 3 AM daily
        'description': 'Daily at 3:00 AM UTC'
    },
    'weekly': {
        'cron': '0 3 * * 0',  # 3 AM every Sunday
        'description': 'Weekly on Sunday at 3:00 AM UTC'
    },
    'monthly': {
        'cron': '0 3 1 * *',  # 3 AM on 1st of month
        'description': 'Monthly on the 1st at 3:00 AM UTC'
    }
}

# Default configuration
DEFAULT_CONFIG = {
    'interval': 'weekly',
    'horizon_months': 6,
    'train_end_date': None,
    'max_features': 50,
    'threshold_override': None,
    'timeout_minutes': 30,
    'retry_attempts': 3,
    'retry_delay_seconds': 60,
}

CONFIG_PATH = Path(__file__).parent.parent / "data" / "models" / "runtime_config.json"

def get_cron_expression(interval: str = 'weekly') -> str:
    """
    Get cron expression for given interval
    
    Args:
        interval: One of 'daily', 'weekly', 'monthly'
        
    Returns:
        Cron expression string
    """
    return SCHEDULER_INTERVALS.get(interval, SCHEDULER_INTERVALS['weekly'])['cron']


def get_scheduler_description(interval: str = 'weekly') -> str:
    """
    Get human-readable description of scheduler interval
    
    Args:
        interval: One of 'daily', 'weekly', 'monthly'
        
    Returns:
        Description string
    """
    return SCHEDULER_INTERVALS.get(interval, SCHEDULER_INTERVALS['weekly'])['description']


def _validate_config(config: dict) -> dict:
    """Validate and normalize config values against supported ranges."""
    validated = DEFAULT_CONFIG.copy()
    validated.update(config or {})

    if validated.get('interval') not in SCHEDULER_INTERVALS:
        validated['interval'] = DEFAULT_CONFIG['interval']

    try:
        validated['horizon_months'] = int(validated.get('horizon_months', 6))
    except (TypeError, ValueError):
        validated['horizon_months'] = DEFAULT_CONFIG['horizon_months']
    if validated['horizon_months'] not in (3, 6, 12):
        validated['horizon_months'] = DEFAULT_CONFIG['horizon_months']

    train_end = validated.get('train_end_date')
    validated['train_end_date'] = train_end if train_end else None

    try:
        validated['max_features'] = int(validated.get('max_features', 50))
    except (TypeError, ValueError):
        validated['max_features'] = DEFAULT_CONFIG['max_features']
    validated['max_features'] = max(10, min(200, validated['max_features']))

    th = validated.get('threshold_override')
    if th in ("", None):
        validated['threshold_override'] = None
    else:
        try:
            th_val = float(th)
            validated['threshold_override'] = max(0.0, min(1.0, th_val))
        except (TypeError, ValueError):
            validated['threshold_override'] = None

    return validated


def load_runtime_config() -> dict:
    """Load persisted runtime config, merged with sane defaults."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                raw = json.load(f)
            return _validate_config(raw)
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()


def save_runtime_config(config: dict) -> dict:
    """Validate and persist runtime config to disk."""
    validated = _validate_config(config)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(validated, f, indent=2)
    return validated

