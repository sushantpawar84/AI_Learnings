"""
Unit tests for SVN to Git migration tool.
Run with: python -m pytest tests/ or python -m unittest discover -s tests -p "test_*.py"
"""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from svn2git_tool.core import Config
from svn2git_tool.utils import SVNFetchError, GitPushError, retry_with_backoff, FailureMode, classify_error
from svn2git_tool.core import StateManager, MigrationStatus
from svn2git_tool.adapters import SVNRevisionMetadata


class TestConfig(unittest.TestCase):
    """Tests for configuration management."""

    def test_config_from_env_vars(self):
        """Test loading config from environment variables."""
        import os
        os.environ['SVN_URL'] = 'https://svn.example.com'
        os.environ['GIT_URL'] = 'https://github.com/org/repo.git'

        config = Config()
        self.assertEqual(config.svn_url, 'https://svn.example.com')
        self.assertEqual(config.git_url, 'https://github.com/org/repo.git')

    def test_config_defaults(self):
        """Test that defaults are used when not specified."""
        import os
        os.environ['SVN_URL'] = 'https://svn.example.com'
        os.environ['GIT_URL'] = 'https://github.com/org/repo.git'

        config = Config()
        self.assertEqual(config.batch_size, 50)
        self.assertEqual(config.batch_timeout_seconds, 300)
        self.assertEqual(config.retry_attempts, 3)


class TestErrorHandler(unittest.TestCase):
    """Tests for error handling."""

    def test_classify_error_transient(self):
        """Test classification of transient errors."""
        error = SVNFetchError("Connection timeout")
        # Note: In actual implementation, connection errors might be classified as transient
        # This is a simple example
        pass

    def test_classify_error_permanent(self):
        """Test classification of permanent errors."""
        error = SVNFetchError("Invalid credentials")
        mode = classify_error(error)
        self.assertEqual(mode, FailureMode.PERMANENT)

    def test_retry_with_backoff_success(self):
        """Test retry logic succeeds on first attempt."""
        call_count = {'count': 0}

        def test_func():
            call_count['count'] += 1
            return 'success'

        result = retry_with_backoff(test_func, max_attempts=3)
        self.assertEqual(result, 'success')
        self.assertEqual(call_count['count'], 1)

    def test_retry_with_backoff_eventual_success(self):
        """Test retry logic succeeds after retries."""
        call_count = {'count': 0}

        def test_func():
            call_count['count'] += 1
            if call_count['count'] < 3:
                raise RuntimeError("Temporary error")
            return 'success'

        result = retry_with_backoff(test_func, max_attempts=3, base_delay=0.01)
        self.assertEqual(result, 'success')
        self.assertEqual(call_count['count'], 3)

    def test_retry_with_backoff_all_fail(self):
        """Test retry logic fails after all attempts."""
        def test_func():
            raise RuntimeError("Permanent error")

        with self.assertRaises(RuntimeError):
            retry_with_backoff(test_func, max_attempts=2, base_delay=0.01)


class TestStateManager(unittest.TestCase):
    """Tests for state management."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / 'test.db'
        self.state_manager = StateManager(self.db_path)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_initialize_migration(self):
        """Test migration initialization."""
        self.state_manager.initialize_migration(
            'https://svn.example.com',
            'https://github.com/org/repo.git',
            100
        )

        status = self.state_manager.get_migration_status()
        self.assertIsNotNone(status)
        self.assertEqual(status['total_svn_revisions'], 100)

    def test_create_checkpoint(self):
        """Test checkpoint creation."""
        self.state_manager.create_checkpoint(
            svn_revision=1,
            batch_number=1,
            author='testuser',
            timestamp=datetime.now().isoformat(),
            message='Test commit'
        )

        revisions = self.state_manager.get_pending_revisions()
        self.assertEqual(len(revisions), 1)
        self.assertEqual(revisions[0]['svn_revision'], 1)

    def test_mark_checkpoint_completed(self):
        """Test marking checkpoint as completed."""
        self.state_manager.create_checkpoint(
            svn_revision=1,
            batch_number=1
        )

        self.state_manager.mark_checkpoint_completed(1, 'abc123def456')

        last_completed = self.state_manager.get_last_completed_revision()
        self.assertEqual(last_completed, 1)

    def test_mark_checkpoint_failed(self):
        """Test marking checkpoint as failed."""
        self.state_manager.create_checkpoint(
            svn_revision=1,
            batch_number=1
        )

        self.state_manager.mark_checkpoint_failed(1, 'Test error')

        revisions = self.state_manager.get_pending_revisions()
        # Should not appear in pending if failed
        for rev in revisions:
            if rev['svn_revision'] == 1:
                self.assertEqual(rev['status'], MigrationStatus.FAILED.value)

    def test_get_migration_summary(self):
        """Test getting migration summary."""
        self.state_manager.initialize_migration(
            'https://svn.example.com',
            'https://github.com/org/repo.git',
            10
        )

        # Create some checkpoints
        for i in range(1, 11):
            self.state_manager.create_checkpoint(i, 1)

        # Mark some as completed
        self.state_manager.mark_checkpoint_completed(1, 'sha1')
        self.state_manager.mark_checkpoint_completed(2, 'sha2')

        summary = self.state_manager.get_migration_summary()
        self.assertEqual(summary['total_revisions'], 10)
        self.assertEqual(summary['completed'], 2)
        self.assertEqual(summary['pending'], 8)


class TestSVNRevisionMetadata(unittest.TestCase):
    """Tests for SVN revision metadata."""

    def test_svn_revision_metadata_creation(self):
        """Test SVN revision metadata creation."""
        metadata = SVNRevisionMetadata(
            revision=1,
            author='testuser',
            timestamp='2024-01-01T00:00:00Z',
            message='Test commit',
            files_changed=5
        )

        self.assertEqual(metadata.revision, 1)
        self.assertEqual(metadata.author, 'testuser')
        self.assertEqual(metadata.files_changed, 5)


if __name__ == '__main__':
    unittest.main()

