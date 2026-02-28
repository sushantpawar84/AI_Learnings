"""
Migration orchestrator - coordinates the complete SVN to Git migration workflow.
Handles batch processing, checkpoints, timeout prevention, and error recovery.
"""

import time
from pathlib import Path
from svn2git_tool.utils import get_logger
from svn2git_tool.core import Config
from svn2git_tool.core import StateManager
from svn2git_tool.adapters import SVNAdapter, SVNRevisionMetadata
from svn2git_tool.adapters import GitAdapter
from svn2git_tool.utils import (
    MigrationError, SVNFetchError, GitPushError, TimeoutError as MigrationTimeoutError,
    handle_timeout_error,
    classify_error, FailureMode
)

logger = get_logger(__name__)


class TimeoutManager:
    """Manages timeout enforcement for batch operations."""

    def __init__(self, timeout_seconds: int):
        """
        Initialize timeout manager.

        Args:
            timeout_seconds: Timeout duration in seconds
        """
        self.timeout_seconds = timeout_seconds
        self.start_time = None
        self.timed_out = False

    def start(self) -> None:
        """Start the timeout timer."""
        self.start_time = time.time()
        self.timed_out = False

    def check(self) -> bool:
        """
        Check if timeout has been exceeded.

        Returns:
            True if timed out, False otherwise
        """
        if self.start_time is None:
            return False

        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            self.timed_out = True
            return True

        return False

    def get_remaining_time(self) -> float:
        """Get remaining time in seconds."""
        if self.start_time is None:
            return self.timeout_seconds

        elapsed = time.time() - self.start_time
        return max(0, self.timeout_seconds - elapsed)

    def should_stop_batch(self, threshold_percent: float = 90) -> bool:
        """
        Check if batch should stop to avoid timeout.

        Args:
            threshold_percent: Stop if this percentage of timeout is consumed

        Returns:
            True if should stop batch, False otherwise
        """
        if self.start_time is None:
            return False

        elapsed = time.time() - self.start_time
        threshold = self.timeout_seconds * (threshold_percent / 100)
        return elapsed > threshold


class MigrationOrchestrator:
    """Orchestrates the complete SVN to Git migration process."""

    def __init__(self, config: Config):
        """
        Initialize migration orchestrator.

        Args:
            config: Configuration object
        """
        self.config = config
        self.state_manager = StateManager(config.db_path)

        self.svn_adapter = None
        self.git_adapter = None

        self.batch_number = 0
        self.current_batch_start_revision = 0
        self.timeout_manager = None

    def start_migration(self, resume: bool = True) -> None:
        """
        Start the migration process.

        Args:
            resume: Whether to resume from last checkpoint (True) or start fresh (False)
        """
        try:
            logger.info("=" * 80)
            logger.info("SVN to Git Migration Started")
            logger.info("=" * 80)
            logger.info(f"SVN URL: {self.config.svn_url}")
            logger.info(f"Git URL: {self.config.git_url}")
            logger.info(f"Batch Size: {self.config.batch_size}")
            logger.info(f"Batch Timeout: {self.config.batch_timeout_seconds}s")

            # Initialize adapters
            self._initialize_adapters()

            # Get repository information
            total_revisions = self.svn_adapter.get_revision_count()
            logger.info(f"Total SVN revisions to migrate: {total_revisions}")

            # Determine starting point
            if resume:
                last_completed = self.state_manager.get_last_completed_revision()
                start_revision = last_completed + 1
                logger.info(f"Resuming from revision {start_revision}")
            else:
                start_revision = 1
                self.state_manager.initialize_migration(
                    self.config.svn_url,
                    self.config.git_url,
                    total_revisions
                )

            # Process batches
            self._process_batches(start_revision, total_revisions)

            # Print summary
            self._print_migration_summary()

            logger.info("=" * 80)
            logger.info("SVN to Git Migration Completed Successfully")
            logger.info("=" * 80)

        except MigrationError as e:
            logger.error(f"Migration failed: {e}")
            self._print_migration_summary()
            raise
        except KeyboardInterrupt:
            logger.warning("Migration interrupted by user")
            self._print_migration_summary()
            raise
        except Exception as e:
            logger.error(f"Unexpected error during migration: {e}", exc_info=True)
            self._print_migration_summary()
            raise

    def _initialize_adapters(self) -> None:
        """
        Initialize SVN and Git adapters.

        Raises:
            MigrationError: If initialization fails
        """
        logger.info("Initializing adapters...")

        try:
            self.svn_adapter = SVNAdapter(
                self.config.svn_url,
                self.config.svn_username,
                self.config.svn_password
            )
            logger.info("SVN adapter initialized")
        except Exception as e:
            raise MigrationError(f"Failed to initialize SVN adapter: {e}")

        try:
            git_path = Path('./temp_git_repo')
            self.git_adapter = GitAdapter(
                self.config.git_url,
                str(git_path),
                self.config.git_author_name,
                self.config.git_author_email,
                self.config.git_token
            )
            logger.info("Git adapter initialized")
        except Exception as e:
            raise MigrationError(f"Failed to initialize Git adapter: {e}")

    def _process_batches(self, start_revision: int, total_revisions: int) -> None:
        """
        Process revisions in batches.

        Args:
            start_revision: Starting revision number
            total_revisions: Total revisions to process
        """
        current_revision = start_revision

        while current_revision <= total_revisions:
            self.batch_number += 1
            self.current_batch_start_revision = current_revision

            batch_end = min(
                current_revision + self.config.batch_size - 1,
                total_revisions
            )

            logger.info(f"\n{'='*80}")
            logger.info(f"Processing Batch #{self.batch_number}: "
                       f"Revisions {current_revision}-{batch_end}")
            logger.info(f"Progress: {current_revision}/{total_revisions}")
            logger.info(f"{'='*80}")

            try:
                # Initialize timeout manager for this batch
                self.timeout_manager = TimeoutManager(
                    self.config.batch_timeout_seconds
                )
                self.timeout_manager.start()

                # Fetch revisions from SVN
                revisions = self.svn_adapter.fetch_revisions(
                    current_revision,
                    batch_end,
                    limit=self.config.batch_size
                )

                if not revisions:
                    logger.warning("No revisions fetched, batch might be empty")
                    current_revision = batch_end + 1
                    continue

                # Process each revision
                for revision_metadata in revisions:
                    # Check for timeout before processing
                    if self.timeout_manager.should_stop_batch():
                        logger.warning(
                            f"Batch approaching timeout ({self.config.batch_timeout_seconds}s), "
                            "stopping batch early to checkpoint"
                        )
                        handle_timeout_error(
                            "Batch processing",
                            f"Revision {revision_metadata.revision}"
                        )
                        break

                    self._process_revision(revision_metadata)

                # Push batch to Git
                self._push_batch()

                # Update metadata
                self.state_manager.update_migration_metadata(
                    last_checkpoint_revision=batch_end,
                    current_batch_number=self.batch_number
                )

                logger.info(f"Batch #{self.batch_number} completed successfully")

                current_revision = batch_end + 1

            except MigrationTimeoutError as e:
                logger.error(f"Batch #{self.batch_number} timed out: {e}")
                handle_timeout_error("Batch processing", self.current_batch_start_revision)
                self.state_manager.record_error(
                    'timeout',
                    str(e),
                    batch_number=self.batch_number
                )
                # Don't increment revision, retry same batch
                break

            except MigrationError as e:
                logger.error(f"Batch #{self.batch_number} failed: {e}")
                failure_mode = classify_error(e)

                if failure_mode == FailureMode.TRANSIENT:
                    logger.info(f"Transient error, will retry batch on next run")
                    self.state_manager.record_error(
                        'transient_error',
                        str(e),
                        batch_number=self.batch_number
                    )
                    break
                else:
                    logger.error(f"Permanent error in batch, manual intervention needed")
                    self.state_manager.record_error(
                        'permanent_error',
                        str(e),
                        batch_number=self.batch_number
                    )
                    break

            except Exception as e:
                logger.error(f"Unexpected error in batch: {e}", exc_info=True)
                self.state_manager.record_error(
                    'unexpected_error',
                    str(e),
                    batch_number=self.batch_number
                )
                break

    def _process_revision(self, revision_metadata: SVNRevisionMetadata) -> None:
        """
        Process a single SVN revision.

        Args:
            revision_metadata: SVN revision metadata

        Raises:
            MigrationError: If processing fails
        """
        revision = revision_metadata.revision
        logger.debug(f"Processing revision {revision}: {revision_metadata.author}")

        try:
            # Create checkpoint
            self.state_manager.create_checkpoint(
                revision,
                self.batch_number,
                revision_metadata.author,
                revision_metadata.timestamp,
                revision_metadata.message
            )

            # Mark as in progress
            self.state_manager.mark_checkpoint_in_progress(revision)

            # Create an empty commit with metadata
            # (In production, you'd export the actual files and create a proper commit)
            commit_sha = self.git_adapter.create_empty_commit(
                author_name=revision_metadata.author or self.config.git_author_name,
                author_email=self.config.git_author_email,
                commit_message=self._format_commit_message(revision_metadata),
                commit_date=revision_metadata.timestamp
            )

            # Mark as completed
            self.state_manager.mark_checkpoint_completed(revision, commit_sha)

            logger.debug(f"Revision {revision} -> Commit {commit_sha[:8]}")

        except SVNFetchError as e:
            self.state_manager.mark_checkpoint_failed(revision, str(e))
            raise
        except GitPushError as e:
            self.state_manager.mark_checkpoint_failed(revision, str(e))
            raise
        except Exception as e:
            self.state_manager.mark_checkpoint_failed(revision, str(e))
            raise

    def _format_commit_message(self, revision_metadata: SVNRevisionMetadata) -> str:
        """
        Format a commit message from SVN revision metadata.

        Args:
            revision_metadata: SVN revision metadata

        Returns:
            Formatted commit message
        """
        message = revision_metadata.message.strip() if revision_metadata.message else ""

        if not message:
            message = f"Empty commit"

        footer = f"\n\n[SVN r{revision_metadata.revision}]"
        if revision_metadata.files_changed:
            footer += f" | {revision_metadata.files_changed} files changed"

        return f"{message}{footer}"

    def _push_batch(self) -> None:
        """
        Push accumulated commits to Git.

        Raises:
            GitPushError: If push fails
        """
        logger.info(f"Pushing batch to Git...")

        try:
            self.git_adapter.push_to_remote('master', force=False)
            logger.info("Batch pushed successfully")

        except GitPushError as e:
            logger.error(f"Failed to push batch: {e}")
            raise

    def _print_migration_summary(self) -> None:
        """Print migration summary statistics."""
        summary = self.state_manager.get_migration_summary()

        logger.info("\n" + "=" * 80)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total Revisions: {summary['total_revisions']}")
        logger.info(f"Completed: {summary['completed']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Pending: {summary['pending']}")
        logger.info(f"In Progress: {summary['in_progress']}")
        logger.info(f"Progress: {summary['progress_percent']:.2f}%")
        logger.info("=" * 80)

