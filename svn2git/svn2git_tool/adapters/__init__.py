"""
Adapter module for external repository interfaces.

Contains:
- SVN repository adapter
- Git repository adapter
"""

from .svn_adapter import SVNAdapter, SVNRevisionMetadata
from .git_adapter import GitAdapter, GitCloneProgress, GitPushProgress

__all__ = [
    "SVNAdapter",
    "SVNRevisionMetadata",
    "GitAdapter",
    "GitCloneProgress",
    "GitPushProgress",
]

