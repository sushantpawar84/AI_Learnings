"""
Git repository adapter for creating commits and pushing to GitHub.
Handles authentication and batch commit operations.
"""

import shutil
from pathlib import Path
from typing import Optional, Dict
import git
from svn2git_tool.utils import get_logger
from svn2git_tool.utils import GitAuthenticationError, GitPushError, retry_with_backoff

logger = get_logger(__name__)


class GitAdapter:
    """Adapter for interacting with Git repositories."""

    def __init__(self, git_url: str, local_repo_path: str,
                 author_name: str = 'Migration Bot',
                 author_email: str = 'bot@example.com',
                 git_token: Optional[str] = None):
        """
        Initialize Git adapter.

        Args:
            git_url: Git repository URL
            local_repo_path: Local path for cloned repository
            author_name: Author name for commits
            author_email: Author email for commits
            git_token: GitHub API token (for HTTPS auth)

        Raises:
            GitAuthenticationError: If unable to authenticate with Git
        """
        self.git_url = git_url
        self.local_repo_path = Path(local_repo_path)
        self.author_name = author_name
        self.author_email = author_email
        self.git_token = git_token

        self.repo = None
        self._initialize_repository()

    def _initialize_repository(self) -> None:
        """
        Initialize or clone Git repository.

        Raises:
            GitAuthenticationError: If authentication fails
        """
        def init():
            try:
                # Prepare Git URL with token if needed
                git_url = self.git_url
                if self.git_token and 'https://' in git_url:
                    # Insert token into URL: https://token@github.com/org/repo.git
                    git_url = git_url.replace(
                        'https://',
                        f'https://oauth2:{self.git_token}@'
                    )

                if self.local_repo_path.exists():
                    logger.info(f"Opening existing repository at {self.local_repo_path}")
                    self.repo = git.Repo(str(self.local_repo_path))
                else:
                    logger.info(f"Cloning repository: {self.git_url}")
                    self.local_repo_path.parent.mkdir(parents=True, exist_ok=True)

                    self.repo = git.Repo.clone_from(
                        git_url,
                        str(self.local_repo_path),
                        progress=GitCloneProgress()
                    )

                # Configure user
                with self.repo.config_writer() as git_config:
                    git_config.set_value('user', 'name', self.author_name)
                    git_config.set_value('user', 'email', self.author_email)

                logger.info(f"Repository initialized: {self.local_repo_path}")

            except git.GitCommandError as e:
                error_msg = str(e)
                if 'authentication' in error_msg.lower() or 'permission' in error_msg.lower():
                    raise GitAuthenticationError(f"Git authentication failed: {e}")
                raise GitPushError(f"Git operation failed: {e}")
            except Exception as e:
                raise GitAuthenticationError(f"Failed to initialize Git repository: {e}")

        retry_with_backoff(init, max_attempts=3, base_delay=5)

    def create_commit(self, files_content: Dict[str, str],
                     author_name: str,
                     author_email: str,
                     commit_message: str,
                     commit_date: Optional[str] = None) -> str:
        """
        Create a commit in the repository.

        Args:
            files_content: Dictionary of {file_path: content}
            author_name: Commit author name
            author_email: Commit author email
            commit_message: Commit message
            commit_date: Commit date (ISO format), defaults to now

        Returns:
            Git commit SHA

        Raises:
            GitPushError: If commit creation fails
        """
        try:
            if not self.repo:
                raise GitPushError("Repository not initialized")

            # Write files
            for file_path, content in files_content.items():
                full_path = self.local_repo_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)

                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            # Stage all changes
            self.repo.index.add(list(files_content.keys()))

            # Create commit with proper author info
            commit_kwargs = {
                'author': git.Actor(author_name, author_email),
                'committer': git.Actor(author_name, author_email),
                'message': commit_message
            }

            if commit_date:
                commit_kwargs['commit_date'] = commit_date

            commit = self.repo.index.commit(**commit_kwargs)

            logger.debug(f"Created commit: {commit.hexsha[:8]} - {commit_message[:50]}")
            return commit.hexsha

        except Exception as e:
            raise GitPushError(f"Failed to create commit: {e}")

    def create_empty_commit(self, author_name: str,
                           author_email: str,
                           commit_message: str,
                           commit_date: Optional[str] = None) -> str:
        """
        Create an empty commit (no file changes).

        Args:
            author_name: Commit author name
            author_email: Commit author email
            commit_message: Commit message
            commit_date: Commit date (ISO format)

        Returns:
            Git commit SHA

        Raises:
            GitPushError: If commit creation fails
        """
        try:
            if not self.repo:
                raise GitPushError("Repository not initialized")

            commit_kwargs = {
                'author': git.Actor(author_name, author_email),
                'committer': git.Actor(author_name, author_email),
                'message': commit_message,
                'allow_empty': True
            }

            if commit_date:
                commit_kwargs['commit_date'] = commit_date

            commit = self.repo.index.commit(**commit_kwargs)

            logger.debug(f"Created empty commit: {commit.hexsha[:8]}")
            return commit.hexsha

        except Exception as e:
            raise GitPushError(f"Failed to create empty commit: {e}")

    def push_to_remote(self, branch: str = 'master',
                      force: bool = False) -> None:
        """
        Push commits to remote repository.

        Args:
            branch: Branch name to push
            force: Force push if needed

        Raises:
            GitPushError: If push fails
        """
        def push():
            try:
                if not self.repo:
                    raise GitPushError("Repository not initialized")

                origin = self.repo.remote('origin')

                push_kwargs = {'progress': GitPushProgress()}
                if force:
                    push_kwargs['force'] = True

                logger.info(f"Pushing branch '{branch}' to remote...")
                origin.push(branch, **push_kwargs)
                logger.info(f"Successfully pushed '{branch}' to remote")

            except git.GitCommandError as e:
                error_msg = str(e)
                if 'permission' in error_msg.lower() or 'authentication' in error_msg.lower():
                    raise GitAuthenticationError(f"Git push authentication failed: {e}")
                raise GitPushError(f"Git push failed: {e}")
            except Exception as e:
                raise GitPushError(f"Failed to push to remote: {e}")

        retry_with_backoff(push, max_attempts=3, base_delay=10)

    def get_last_commit_sha(self) -> Optional[str]:
        """
        Get the SHA of the last commit on current branch.

        Returns:
            Commit SHA or None if no commits
        """
        try:
            if not self.repo or not self.repo.head.is_valid():
                return None
            return self.repo.head.commit.hexsha
        except Exception:
            return None

    def get_commit_count(self) -> int:
        """
        Get total number of commits in repository.

        Returns:
            Commit count
        """
        try:
            if not self.repo:
                return 0
            return len(list(self.repo.iter_commits()))
        except Exception:
            return 0

    def reset_to_commit(self, commit_sha: str) -> None:
        """
        Reset repository to a specific commit.

        Args:
            commit_sha: Commit SHA to reset to

        Raises:
            GitPushError: If reset fails
        """
        try:
            if not self.repo:
                raise GitPushError("Repository not initialized")

            self.repo.head.reset(commit_sha, index=True, working_tree=True)
            logger.info(f"Reset to commit {commit_sha[:8]}")

        except Exception as e:
            raise GitPushError(f"Failed to reset to commit: {e}")

    def cleanup(self) -> None:
        """Clean up local repository."""
        try:
            if self.local_repo_path.exists():
                shutil.rmtree(str(self.local_repo_path))
                logger.info(f"Cleaned up repository at {self.local_repo_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup repository: {e}")


class GitCloneProgress(git.RemoteProgress):
    """Progress handler for git clone operations."""

    def update(self, op_code, cur_count, max_count=None, message=''):
        """Update progress."""
        if message:
            logger.debug(f"Clone: {message}")


class GitPushProgress(git.RemoteProgress):
    """Progress handler for git push operations."""

    def update(self, op_code, cur_count, max_count=None, message=''):
        """Update progress."""
        if message:
            logger.debug(f"Push: {message}")

