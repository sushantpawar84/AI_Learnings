"""
SVN to Git Migration Tool

A production-grade tool for migrating SVN repositories to Git with:
- Full version history preservation
- Incremental resumable migration
- Timeout prevention via batch processing
- Comprehensive error handling
- SQLite-based state persistence
"""

__version__ = "1.0.0"
__author__ = "SVN to Git Migration Team"
__email__ = "sushant789@gmail.com"

from svn2git_tool.main import main

__all__ = ["main"]

