"""
SVN to Git Migration Tool - Entry Point

Migrates a complete SVN repository with full version history to Git/GitHub.
Supports incremental/resumable migration with checkpoint recovery.

Usage:
    python -m svn2git_tool.main --svn-url <url> --git-url <url> [--config config.yaml] [--resume] [--fresh]
"""

import sys
import argparse
from pathlib import Path
from svn2git_tool.utils import LoggerManager, get_logger
from svn2git_tool.core import Config, ConfigError, MigrationOrchestrator
from svn2git_tool.utils import MigrationError

logger = None


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='SVN to Git Repository Migration Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Migrate with environment variables (SVN_URL, GIT_URL, etc.)
  python main.py
  
  # Migrate with explicit URLs
  python main.py --svn-url https://svn.example.com/repo --git-url https://github.com/org/repo.git
  
  # Resume from last checkpoint
  python main.py --resume
  
  # Start fresh (clear existing state)
  python main.py --fresh
  
  # Use custom config file
  python main.py --config myconfig.yaml
        '''
    )

    parser.add_argument(
        '--svn-url',
        help='Source SVN repository URL'
    )
    parser.add_argument(
        '--git-url',
        help='Target Git repository URL'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from last checkpoint (default: True)'
    )
    parser.add_argument(
        '--fresh',
        action='store_true',
        help='Start fresh migration (clear state, overrides --resume)'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Number of revisions per batch (overrides config)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        help='Batch timeout in seconds (overrides config)'
    )

    return parser


def load_configuration(args: argparse.Namespace) -> Config:
    """
    Load configuration from arguments and files.

    Args:
        args: Parsed command-line arguments

    Returns:
        Configuration object

    Raises:
        ConfigError: If configuration is invalid
    """
    # Set environment variables from arguments if provided
    if args.svn_url:
        import os
        os.environ['SVN_URL'] = args.svn_url
    if args.git_url:
        import os
        os.environ['GIT_URL'] = args.git_url
    if args.batch_size:
        import os
        os.environ['BATCH_SIZE'] = str(args.batch_size)
    if args.timeout:
        import os
        os.environ['BATCH_TIMEOUT_SECONDS'] = str(args.timeout)

    os.environ['LOG_LEVEL'] = args.log_level

    config_file = None
    if Path(args.config).exists():
        config_file = args.config
        logger.info(f"Loading configuration from: {config_file}")

    return Config(config_file)


def main():
    """Main entry point."""
    global logger

    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_configuration(args)

        # Setup logging
        LoggerManager.setup_logging(config.log_dir, config.log_level)
        logger = get_logger(__name__)

        logger.info("SVN to Git Migration Tool Starting")
        logger.info(f"Configuration loaded successfully")

        # Create orchestrator
        orchestrator = MigrationOrchestrator(config)

        # Start migration
        resume = not args.fresh  # If --fresh, start from beginning
        orchestrator.start_migration(resume=resume)

        return 0

    except ConfigError as e:
        if logger:
            logger.error(f"Configuration error: {e}")
        else:
            print(f"ERROR: Configuration error: {e}", file=sys.stderr)
        return 1

    except MigrationError as e:
        if logger:
            logger.error(f"Migration error: {e}")
        else:
            print(f"ERROR: Migration error: {e}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        if logger:
            logger.info("Migration interrupted by user")
        else:
            print("\nMigration interrupted by user")
        return 2

    except Exception as e:
        if logger:
            logger.error(f"Unexpected error: {e}", exc_info=True)
        else:
            print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
