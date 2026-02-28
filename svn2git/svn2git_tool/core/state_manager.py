"""
Persistent state management for SVN to Git migration.
Tracks migration progress, checkpoints, and commit mappings using SQLite.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from svn2git_tool.utils import get_logger

logger = get_logger(__name__)


class MigrationStatus(Enum):
    """Status of a migration checkpoint."""
    PENDING = 'pending'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'
    SKIPPED = 'skipped'


class StateManager:
    """Manages persistent migration state in SQLite database."""

    def __init__(self, db_path: Path):
        """
        Initialize state manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Migration metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS migration_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    svn_url TEXT NOT NULL,
                    git_url TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    last_update_time TEXT NOT NULL,
                    last_checkpoint_revision INTEGER DEFAULT 0,
                    total_svn_revisions INTEGER DEFAULT 0,
                    current_batch_number INTEGER DEFAULT 0,
                    migration_status TEXT DEFAULT 'in_progress'
                )
            ''')

            # Migration checkpoints table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS migration_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    svn_revision INTEGER NOT NULL UNIQUE,
                    git_commit_sha TEXT,
                    svn_author TEXT,
                    svn_timestamp TEXT,
                    svn_message TEXT,
                    batch_number INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0
                )
            ''')

            # Migration errors table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS migration_errors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    svn_revision INTEGER,
                    batch_number INTEGER,
                    error_type TEXT NOT NULL,
                    error_message TEXT NOT NULL,
                    error_timestamp TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolution_notes TEXT
                )
            ''')

            # Create indexes for performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_checkpoint_status 
                ON migration_checkpoints(status)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_checkpoint_batch 
                ON migration_checkpoints(batch_number)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_error_revision 
                ON migration_errors(svn_revision)
            ''')

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

        finally:
            conn.close()

    def initialize_migration(self, svn_url: str, git_url: str,
                            total_revisions: int) -> None:
        """
        Initialize a new migration session.

        Args:
            svn_url: Source SVN URL
            git_url: Target Git URL
            total_revisions: Total SVN revisions to migrate
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO migration_metadata 
                (svn_url, git_url, start_time, last_update_time, 
                 total_svn_revisions, migration_status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (svn_url, git_url, now, now, total_revisions, 'in_progress'))

            conn.commit()
            logger.info(f"Initialized migration: SVN={svn_url}, Git={git_url}")

        finally:
            conn.close()

    def get_migration_status(self) -> Optional[Dict[str, Any]]:
        """
        Get current migration status.

        Returns:
            Dictionary with migration metadata or None if not initialized
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT * FROM migration_metadata ORDER BY id DESC LIMIT 1')
            row = cursor.fetchone()

            if row:
                return dict(row)
            return None

        finally:
            conn.close()

    def get_last_completed_revision(self) -> int:
        """
        Get the last successfully completed SVN revision.

        Returns:
            Last completed revision number, or 0 if none
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT MAX(svn_revision) as max_revision
                FROM migration_checkpoints
                WHERE status = ?
            ''', (MigrationStatus.COMPLETED.value,))

            row = cursor.fetchone()
            return row['max_revision'] or 0

        finally:
            conn.close()

    def create_checkpoint(self, svn_revision: int, batch_number: int,
                         author: str = '', timestamp: str = '',
                         message: str = '') -> None:
        """
        Create a new migration checkpoint for an SVN revision.

        Args:
            svn_revision: SVN revision number
            batch_number: Batch number this revision belongs to
            author: SVN commit author
            timestamp: SVN commit timestamp
            message: SVN commit message
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT OR REPLACE INTO migration_checkpoints
                (svn_revision, batch_number, svn_author, svn_timestamp,
                 svn_message, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (svn_revision, batch_number, author, timestamp, message,
                  now, now, MigrationStatus.PENDING.value))

            conn.commit()

        finally:
            conn.close()

    def mark_checkpoint_in_progress(self, svn_revision: int) -> None:
        """
        Mark a checkpoint as currently being processed.

        Args:
            svn_revision: SVN revision number
        """
        self._update_checkpoint_status(svn_revision, MigrationStatus.IN_PROGRESS)

    def mark_checkpoint_completed(self, svn_revision: int, git_commit_sha: str) -> None:
        """
        Mark a checkpoint as successfully completed.

        Args:
            svn_revision: SVN revision number
            git_commit_sha: Resulting Git commit SHA
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            now = datetime.now().isoformat()
            cursor.execute('''
                UPDATE migration_checkpoints
                SET status = ?, git_commit_sha = ?, updated_at = ?, retry_count = 0
                WHERE svn_revision = ?
            ''', (MigrationStatus.COMPLETED.value, git_commit_sha, now, svn_revision))

            conn.commit()
            logger.debug(f"Checkpoint completed: revision {svn_revision} -> {git_commit_sha[:8]}")

        finally:
            conn.close()

    def mark_checkpoint_failed(self, svn_revision: int, error_message: str) -> None:
        """
        Mark a checkpoint as failed.

        Args:
            svn_revision: SVN revision number
            error_message: Description of the failure
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            now = datetime.now().isoformat()
            cursor.execute('''
                UPDATE migration_checkpoints
                SET status = ?, error_message = ?, updated_at = ?
                WHERE svn_revision = ?
            ''', (MigrationStatus.FAILED.value, error_message, now, svn_revision))

            conn.commit()
            logger.warning(f"Checkpoint failed: revision {svn_revision} - {error_message}")

        finally:
            conn.close()

    def increment_retry_count(self, svn_revision: int) -> None:
        """
        Increment retry count for a checkpoint.

        Args:
            svn_revision: SVN revision number
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                UPDATE migration_checkpoints
                SET retry_count = retry_count + 1, updated_at = ?
                WHERE svn_revision = ?
            ''', (datetime.now().isoformat(), svn_revision))

            conn.commit()

        finally:
            conn.close()

    def _update_checkpoint_status(self, svn_revision: int,
                                 status: MigrationStatus) -> None:
        """
        Update the status of a checkpoint.

        Args:
            svn_revision: SVN revision number
            status: New status
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            now = datetime.now().isoformat()
            cursor.execute('''
                UPDATE migration_checkpoints
                SET status = ?, updated_at = ?
                WHERE svn_revision = ?
            ''', (status.value, now, svn_revision))

            conn.commit()

        finally:
            conn.close()

    def get_pending_revisions(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all pending (not yet processed) revisions.

        Args:
            limit: Maximum number of revisions to return

        Returns:
            List of pending revision dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            query = '''
                SELECT * FROM migration_checkpoints
                WHERE status IN (?, ?)
                ORDER BY svn_revision ASC
            '''
            params = [MigrationStatus.PENDING.value, MigrationStatus.IN_PROGRESS.value]

            if limit:
                query += ' LIMIT ?'
                params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

        finally:
            conn.close()

    def get_batch_revisions(self, batch_number: int) -> List[Dict[str, Any]]:
        """
        Get all revisions in a specific batch.

        Args:
            batch_number: Batch number

        Returns:
            List of revisions in the batch
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute('''
                SELECT * FROM migration_checkpoints
                WHERE batch_number = ?
                ORDER BY svn_revision ASC
            ''', (batch_number,))

            return [dict(row) for row in cursor.fetchall()]

        finally:
            conn.close()

    def record_error(self, error_type: str, error_message: str,
                    svn_revision: Optional[int] = None,
                    batch_number: Optional[int] = None) -> None:
        """
        Record an error in the error log.

        Args:
            error_type: Type/category of the error
            error_message: Detailed error message
            svn_revision: Associated SVN revision (if applicable)
            batch_number: Associated batch number (if applicable)
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            now = datetime.now().isoformat()
            cursor.execute('''
                INSERT INTO migration_errors
                (svn_revision, batch_number, error_type, error_message, error_timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (svn_revision, batch_number, error_type, error_message, now))

            conn.commit()

        finally:
            conn.close()

    def update_migration_metadata(self, **kwargs) -> None:
        """
        Update migration metadata fields.

        Args:
            **kwargs: Metadata fields to update
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Build dynamic update query
            allowed_fields = {
                'last_checkpoint_revision', 'current_batch_number',
                'migration_status', 'total_svn_revisions'
            }

            updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
            if not updates:
                return

            updates['last_update_time'] = datetime.now().isoformat()

            set_clause = ', '.join(f'{k} = ?' for k in updates.keys())
            query = f'UPDATE migration_metadata SET {set_clause}'

            cursor.execute(query, list(updates.values()))
            conn.commit()

        finally:
            conn.close()

    def get_migration_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current migration progress.

        Returns:
            Dictionary with migration summary statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Get metadata
            cursor.execute('SELECT * FROM migration_metadata ORDER BY id DESC LIMIT 1')
            metadata = dict(cursor.fetchone() or {})

            # Get checkpoint stats
            cursor.execute('''
                SELECT status, COUNT(*) as count
                FROM migration_checkpoints
                GROUP BY status
            ''')
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

            # Get total and completed
            total = sum(status_counts.values())
            completed = status_counts.get(MigrationStatus.COMPLETED.value, 0)
            failed = status_counts.get(MigrationStatus.FAILED.value, 0)
            pending = status_counts.get(MigrationStatus.PENDING.value, 0)
            in_progress = status_counts.get(MigrationStatus.IN_PROGRESS.value, 0)

            return {
                'metadata': metadata,
                'total_revisions': total,
                'completed': completed,
                'failed': failed,
                'pending': pending,
                'in_progress': in_progress,
                'progress_percent': (completed / total * 100) if total > 0 else 0,
                'status_counts': status_counts
            }

        finally:
            conn.close()

