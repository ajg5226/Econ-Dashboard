"""
Scheduler configuration
Defines intervals and settings for automated data refresh
"""

from datetime import datetime

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
    'train_end_date': '2015-12-31',
    'timeout_minutes': 30,
    'retry_attempts': 3,
    'retry_delay_seconds': 60
}

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

