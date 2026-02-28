"""
SVN repository adapter for fetching revisions and metadata.
Handles authentication and incremental revision fetching.
"""

import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from svn2git_tool.utils import get_logger
from svn2git_tool.utils import SVNConnectionError, SVNFetchError, retry_with_backoff

logger = get_logger(__name__)


class SVNRevisionMetadata:
    """Represents metadata for an SVN revision."""

    def __init__(self, revision: int, author: str, timestamp: str,
                 message: str, files_changed: int):
        """
        Initialize SVN revision metadata.

        Args:
            revision: Revision number
            author: Commit author
            timestamp: Commit timestamp
            message: Commit message
            files_changed: Number of files changed
        """
        self.revision = revision
        self.author = author
        self.timestamp = timestamp
        self.message = message
        self.files_changed = files_changed

    def __repr__(self) -> str:
        return f"SVNRev({self.revision}, {self.author}, {self.files_changed} files)"


class SVNAdapter:
    """Adapter for interacting with SVN repositories."""

    def __init__(self, svn_url: str, username: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Initialize SVN adapter.

        Args:
            svn_url: SVN repository URL
            username: SVN username
            password: SVN password

        Raises:
            SVNConnectionError: If unable to connect to SVN
        """
        self.svn_url = svn_url
        self.username = username
        self.password = password

        # Verify connection
        self._verify_connection()

    def _get_svn_command(self, *args) -> List[str]:
        """
        Build SVN command with authentication.

        Args:
            *args: SVN command arguments

        Returns:
            List of command arguments ready for subprocess
        """
        cmd = ['svn']

        if self.username:
            cmd.extend(['--username', self.username])
        if self.password:
            cmd.extend(['--password', self.password])

        # Non-interactive mode, trust server certificates
        cmd.extend(['--non-interactive', '--trust-server-cert'])

        cmd.extend(args)
        return cmd

    def _verify_connection(self) -> None:
        """
        Verify that SVN repository is accessible.

        Raises:
            SVNConnectionError: If unable to connect
        """
        def verify():
            try:
                cmd = self._get_svn_command('info', self.svn_url)
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    raise SVNConnectionError(
                        f"SVN connection failed: {result.stderr}"
                    )
            except subprocess.TimeoutExpired:
                raise SVNConnectionError("SVN connection timeout")
            except FileNotFoundError:
                raise SVNConnectionError(
                    "SVN command not found. Please install Subversion."
                )

        retry_with_backoff(verify, max_attempts=3, base_delay=5)
        logger.info(f"SVN connection verified: {self.svn_url}")

    def get_repository_info(self) -> Dict[str, Any]:
        """
        Get repository information.

        Returns:
            Dictionary with repository info

        Raises:
            SVNFetchError: If unable to fetch info
        """
        def fetch():
            try:
                cmd = self._get_svn_command('info', '--xml', self.svn_url)
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode != 0:
                    raise SVNFetchError(f"Failed to get repository info: {result.stderr}")

                root = ET.fromstring(result.stdout)
                entry = root.find('.//entry')

                if entry is None:
                    raise SVNFetchError("Invalid repository info response")

                revision = int(entry.get('revision', 0))
                url = entry.findtext('.//url', '')

                return {
                    'url': url,
                    'revision': revision,
                    'last_changed_rev': revision
                }

            except ET.ParseError as e:
                raise SVNFetchError(f"Failed to parse repository info: {e}")

        return retry_with_backoff(fetch, max_attempts=3, base_delay=5)

    def get_revision_count(self) -> int:
        """
        Get total number of revisions in repository.

        Returns:
            Total revision count

        Raises:
            SVNFetchError: If unable to fetch revision count
        """
        info = self.get_repository_info()
        return info.get('revision', 0)

    def fetch_revisions(self, start_revision: int = 1,
                       end_revision: Optional[int] = None,
                       limit: Optional[int] = None) -> List[SVNRevisionMetadata]:
        """
        Fetch SVN revisions within a range.

        Args:
            start_revision: Starting revision (inclusive)
            end_revision: Ending revision (inclusive), None for HEAD
            limit: Maximum revisions to return

        Returns:
            List of SVN revision metadata

        Raises:
            SVNFetchError: If unable to fetch revisions
        """
        if end_revision is None:
            repo_info = self.get_repository_info()
            end_revision = repo_info['revision']

        # Limit the range if needed
        if limit:
            end_revision = min(start_revision + limit - 1, end_revision)

        def fetch():
            try:
                revision_range = f'{start_revision}:{end_revision}'
                cmd = self._get_svn_command(
                    'log', self.svn_url,
                    f'-r', revision_range,
                    '--xml',
                    '--verbose'
                )

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode != 0:
                    raise SVNFetchError(
                        f"Failed to fetch revisions {revision_range}: {result.stderr}"
                    )

                return self._parse_log_xml(result.stdout)

            except subprocess.TimeoutExpired:
                raise SVNFetchError(
                    f"Timeout fetching revisions {revision_range}"
                )
            except ET.ParseError as e:
                raise SVNFetchError(f"Failed to parse revision log: {e}")

        revisions = retry_with_backoff(fetch, max_attempts=3, base_delay=5)
        logger.info(f"Fetched {len(revisions)} revisions (range: {start_revision}:{end_revision})")
        return revisions

    def _parse_log_xml(self, xml_content: str) -> List[SVNRevisionMetadata]:
        """
        Parse SVN log XML response.

        Args:
            xml_content: XML content from svn log --xml

        Returns:
            List of SVN revision metadata

        Raises:
            SVNFetchError: If parsing fails
        """
        try:
            root = ET.fromstring(xml_content)
            revisions = []

            for logentry in root.findall('.//logentry'):
                revision = int(logentry.get('revision', 0))
                author = logentry.findtext('author', 'unknown')
                timestamp = logentry.findtext('date', '')
                message = logentry.findtext('msg', '')

                # Count changed paths
                paths = logentry.find('paths')
                files_changed = len(paths) if paths is not None else 0

                metadata = SVNRevisionMetadata(
                    revision=revision,
                    author=author,
                    timestamp=timestamp,
                    message=message,
                    files_changed=files_changed
                )
                revisions.append(metadata)

            # Reverse to get chronological order (SVN returns newest first)
            revisions.reverse()
            return revisions

        except ET.ParseError as e:
            raise SVNFetchError(f"Failed to parse SVN log XML: {e}")

    def export_revision(self, revision: int, target_path: str) -> None:
        """
        Export a specific revision to a directory.

        Args:
            revision: Revision to export
            target_path: Target directory path

        Raises:
            SVNFetchError: If export fails
        """
        def export():
            try:
                cmd = self._get_svn_command(
                    'export',
                    f'{self.svn_url}@{revision}',
                    target_path,
                    '--force'
                )

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode != 0:
                    raise SVNFetchError(
                        f"Failed to export revision {revision}: {result.stderr}"
                    )

            except subprocess.TimeoutExpired:
                raise SVNFetchError(f"Timeout exporting revision {revision}")

        retry_with_backoff(export, max_attempts=3, base_delay=5)

    def get_revision_diff(self, revision: int) -> str:
        """
        Get unified diff for a specific revision.

        Args:
            revision: Revision to get diff for

        Returns:
            Unified diff content

        Raises:
            SVNFetchError: If unable to fetch diff
        """
        def fetch():
            try:
                cmd = self._get_svn_command(
                    'diff',
                    f'{self.svn_url}@{revision-1}',
                    f'{self.svn_url}@{revision}'
                )

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode != 0 and 'W' not in result.returncode:
                    raise SVNFetchError(
                        f"Failed to get diff for revision {revision}: {result.stderr}"
                    )

                return result.stdout

            except subprocess.TimeoutExpired:
                raise SVNFetchError(f"Timeout fetching diff for revision {revision}")

        return retry_with_backoff(fetch, max_attempts=3, base_delay=5)

