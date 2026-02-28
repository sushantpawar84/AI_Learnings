"""
Configuration and credentials management for SVN to Git migration.
Loads settings from environment variables and config files.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""
    pass


class Config:
    """Configuration manager for SVN to Git migration."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration from environment variables and optional config file.

        Args:
            config_file: Path to YAML config file (optional)

        Raises:
            ConfigError: If required configuration is missing
        """
        # Load environment variables from .env file
        load_dotenv()

        # Load config file if provided
        self.config_data = {}
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                self.config_data = yaml.safe_load(f) or {}

        self._validate_required_config()

    def _get_config(self, key: str, env_var: Optional[str] = None,
                    default: Optional[str] = None) -> str:
        """
        Get configuration value from config file, environment variable, or default.

        Args:
            key: Configuration key name
            env_var: Environment variable name (if None, uses uppercase key)
            default: Default value if not found

        Returns:
            Configuration value

        Raises:
            ConfigError: If value not found and no default provided
        """
        if env_var is None:
            env_var = key.upper()

        # Priority: config file > environment > default
        if key in self.config_data:
            return str(self.config_data[key])

        env_value = os.getenv(env_var)
        if env_value:
            return env_value

        if default is not None:
            return str(default)

        raise ConfigError(f"Missing required configuration: {key} (set {env_var} env var or config file)")

    def _validate_required_config(self):
        """Validate that all required configuration is present."""
        required_keys = ['SVN_URL', 'GIT_URL']
        for key in required_keys:
            try:
                self._get_config(key, key)
            except ConfigError as e:
                raise ConfigError(f"Required config missing: {e}")

    # SVN Configuration
    @property
    def svn_url(self) -> str:
        """SVN repository URL."""
        return self._get_config('svn_url', 'SVN_URL')

    @property
    def svn_username(self) -> Optional[str]:
        """SVN username for authentication."""
        try:
            return self._get_config('svn_username', 'SVN_USERNAME')
        except ConfigError:
            return None

    @property
    def svn_password(self) -> Optional[str]:
        """SVN password for authentication."""
        try:
            return self._get_config('svn_password', 'SVN_PASSWORD')
        except ConfigError:
            return None

    # Git Configuration
    @property
    def git_url(self) -> str:
        """Git repository URL."""
        return self._get_config('git_url', 'GIT_URL')

    @property
    def git_token(self) -> Optional[str]:
        """GitHub API token for authentication."""
        try:
            return self._get_config('git_token', 'GITHUB_TOKEN')
        except ConfigError:
            return None

    @property
    def git_author_name(self) -> str:
        """Git commit author name."""
        return self._get_config('git_author_name', 'GIT_AUTHOR_NAME',
                               default='SVN Migration Bot')

    @property
    def git_author_email(self) -> str:
        """Git commit author email."""
        return self._get_config('git_author_email', 'GIT_AUTHOR_EMAIL',
                               default='svn-migration@bot.local')

    # Migration Configuration
    @property
    def batch_size(self) -> int:
        """Number of SVN revisions to process per batch."""
        return int(self._get_config('batch_size', 'BATCH_SIZE', default='50'))

    @property
    def batch_timeout_seconds(self) -> int:
        """Maximum seconds allowed per batch processing."""
        return int(self._get_config('batch_timeout_seconds', 'BATCH_TIMEOUT_SECONDS',
                                   default='300'))

    @property
    def retry_attempts(self) -> int:
        """Maximum number of retry attempts for transient failures."""
        return int(self._get_config('retry_attempts', 'RETRY_ATTEMPTS', default='3'))

    @property
    def retry_backoff_base(self) -> int:
        """Base backoff time in seconds for exponential backoff."""
        return int(self._get_config('retry_backoff_base', 'RETRY_BACKOFF_BASE',
                                   default='5'))

    # Logging Configuration
    @property
    def log_level(self) -> str:
        """Logging level (DEBUG, INFO, WARNING, ERROR)."""
        return self._get_config('log_level', 'LOG_LEVEL', default='INFO').upper()

    @property
    def log_dir(self) -> Path:
        """Directory for log files."""
        log_dir = Path(self._get_config('log_dir', 'LOG_DIR', default='./logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    # Database Configuration
    @property
    def db_path(self) -> Path:
        """Path to SQLite database file."""
        db_path = Path(self._get_config('db_path', 'DB_PATH',
                                       default='./migration_state.db'))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    # Large File Handling
    @property
    def large_file_threshold_mb(self) -> int:
        """File size threshold (MB) for special handling."""
        return int(self._get_config('large_file_threshold_mb',
                                   'LARGE_FILE_THRESHOLD_MB', default='100'))

    @property
    def extended_timeout_multiplier(self) -> float:
        """Timeout multiplier for batches with large files."""
        return float(self._get_config('extended_timeout_multiplier',
                                     'EXTENDED_TIMEOUT_MULTIPLIER', default='2.0'))


# Singleton instance
_config_instance: Optional[Config] = None


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get or create the global config instance.

    Args:
        config_file: Path to YAML config file (used only on first call)

    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_file)
    return _config_instance

