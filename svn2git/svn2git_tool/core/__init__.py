"""
Core module for SVN to Git migration orchestration.

Contains:
- Configuration management
- State persistence (SQLite)
- Migration orchestration
"""

from .config import Config, get_config, ConfigError
from .state_manager import StateManager, MigrationStatus
from .orchestrator import MigrationOrchestrator, TimeoutManager

__all__ = [
    "Config",
    "get_config",
    "ConfigError",
    "StateManager",
    "MigrationStatus",
    "MigrationOrchestrator",
    "TimeoutManager",
]

