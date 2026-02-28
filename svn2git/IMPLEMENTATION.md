# Implementation Guide - SVN to Git Migration Tool

## Overview

This document explains the architecture, design decisions, and implementation details of the SVN to Git migration tool.

## Architecture Layers

### 1. Configuration Layer (`config.py`)

**Responsibility:** Load and manage configuration from environment variables, config files, and CLI arguments.

**Key Features:**
- Hierarchical configuration (CLI > Environment > Config File > Defaults)
- Environment variable interpolation in YAML
- Validation of required settings
- Type-safe property accessors

**Usage:**
```python
from config import Config
config = Config('config.yaml')
print(config.batch_size)  # 50
print(config.svn_url)     # https://svn.example.com
```

### 2. Logging Layer (`logger.py`)

**Responsibility:** Provide structured logging to file and console with color support.

**Key Features:**
- Colored console output
- Detailed file logging with timestamps and function names
- Configurable log levels
- Separate handlers for console and file

**Usage:**
```python
from logger import LoggerManager, get_logger
LoggerManager.setup_logging(Path('./logs'), 'INFO')
logger = get_logger(__name__)
logger.info("Starting migration...")
```

### 3. Error Handling Layer (`error_handler.py`)

**Responsibility:** Provide structured error handling, classification, and recovery strategies.

**Key Classes:**
- **Custom Exceptions**: `SVNConnectionError`, `GitPushError`, `TimeoutError`, etc.
- **Retry Logic**: `retry_with_backoff()` with exponential backoff
- **Error Classification**: Classify errors as transient or permanent

**Design Pattern:**
- All operations that might fail use retry logic
- Transient errors (network) get retried
- Permanent errors (auth) fail fast
- Timeout errors checkpoint and allow resume

**Usage:**
```python
from error_handler import retry_with_backoff, SVNConnectionError

def fetch_data():
    # ... operation that might fail
    pass

try:
    result = retry_with_backoff(fetch_data, max_attempts=3, base_delay=5)
except Exception as e:
    logger.error(f"Operation failed: {e}")
```

### 4. State Management Layer (`state_manager.py`)

**Responsibility:** Persist migration state to SQLite and enable checkpoint/resume capability.

**Database Schema:**

```sql
-- Migration metadata
migration_metadata (
  id INTEGER PRIMARY KEY,
  svn_url TEXT,
  git_url TEXT,
  start_time TEXT,
  last_checkpoint_revision INTEGER,
  total_svn_revisions INTEGER,
  migration_status TEXT
)

-- Per-revision tracking
migration_checkpoints (
  id INTEGER PRIMARY KEY,
  svn_revision INTEGER UNIQUE,
  git_commit_sha TEXT,
  batch_number INTEGER,
  status TEXT,  -- 'pending', 'in_progress', 'completed', 'failed'
  retry_count INTEGER,
  error_message TEXT
)

-- Error log
migration_errors (
  id INTEGER PRIMARY KEY,
  svn_revision INTEGER,
  error_type TEXT,
  error_message TEXT,
  error_timestamp TEXT,
  resolved INTEGER
)
```

**Key Responsibilities:**
- Track which revisions have been processed
- Store commit mappings (SVN rev → Git SHA)
- Record errors for debugging
- Enable resume from last checkpoint

**Usage:**
```python
from state_manager import StateManager

state = StateManager(Path('migration_state.db'))
state.initialize_migration(svn_url, git_url, total_revisions)

# During migration
state.create_checkpoint(revision=123, batch_number=1)
state.mark_checkpoint_in_progress(revision=123)
state.mark_checkpoint_completed(revision=123, git_commit_sha='abc123')

# Query progress
last_completed = state.get_last_completed_revision()  # 122
summary = state.get_migration_summary()
```

### 5. SVN Adapter Layer (`svn_adapter.py`)

**Responsibility:** Interface with SVN repository to fetch revisions and metadata.

**Key Features:**
- Authenticate to SVN with credentials
- Fetch revisions in batches
- Parse SVN log XML format
- Error handling and retry logic

**SVN Operations:**
```python
from svn_adapter import SVNAdapter

svn = SVNAdapter(
    'https://svn.example.com/repo',
    username='user',
    password='pass'
)

# Get repository info
info = svn.get_repository_info()
total_revs = svn.get_revision_count()

# Fetch batch of revisions
revisions = svn.fetch_revisions(start=100, end=150)
# Returns list of SVNRevisionMetadata objects:
# - revision: int
# - author: str
# - timestamp: str (ISO format)
# - message: str
# - files_changed: int
```

**Design Notes:**
- Wraps SVN command-line tool (not library) for maximum compatibility
- Parses XML output for structured data
- All operations use retry_with_backoff for resilience
- Timeout handling per operation

### 6. Git Adapter Layer (`git_adapter.py`)

**Responsibility:** Interface with Git repository to create commits and push.

**Key Features:**
- Clone or open existing Git repository
- Create commits with proper metadata
- Push to remote with authentication
- Progress tracking for long operations

**Git Operations:**
```python
from git_adapter import GitAdapter

git = GitAdapter(
    'https://github.com/org/repo.git',
    './temp_git_repo',
    author_name='Bot',
    author_email='bot@example.com',
    git_token='token'
)

# Create commit
sha = git.create_commit(
    files_content={'file.txt': 'content'},
    author_name='Original Author',
    author_email='author@example.com',
    commit_message='Commit message',
    commit_date='2024-01-01T00:00:00Z'
)

# Create empty commit (for empty changesets)
sha = git.create_empty_commit(
    author_name='Original Author',
    author_email='author@example.com',
    commit_message='Empty commit'
)

# Push to remote
git.push_to_remote('master', force=False)
```

**Design Notes:**
- Uses GitPython library for clean Git operations
- Preserves original author and date information
- Supports HTTPS + token authentication
- Clones entire repository into temp directory

### 7. Orchestration Layer (`orchestrator.py`)

**Responsibility:** Coordinate the complete migration workflow.

**Key Components:**

**TimeoutManager:**
```python
timeout = TimeoutManager(300)  # 5 minutes
timeout.start()

if timeout.should_stop_batch(threshold_percent=90):
    # Stop processing this batch
    checkpoint_and_retry_next_run()
```

**MigrationOrchestrator:**
Orchestrates the entire pipeline:

1. **Initialize**: Setup adapters, config, state
2. **Fetch**: Get batch of SVN revisions
3. **Process**: For each revision:
   - Create checkpoint in database
   - Create Git commit
   - Store Git SHA in database
4. **Push**: Push batch to GitHub
5. **Checkpoint**: Save progress
6. **Resume**: On restart, skip already-processed revisions

**Batch Processing Flow:**

```
Start Batch
    ↓
Initialize Timeout Manager (5 min limit)
    ↓
Fetch N revisions from SVN
    ↓
For each revision:
    ├─ Check timeout (stop early if needed)
    ├─ Create database checkpoint
    ├─ Create Git commit
    ├─ Update checkpoint with Git SHA
    └─ Continue if not timed out
    ↓
Push all commits to Git
    ↓
Mark batch completed in database
    ↓
Checkpoint progress
    ↓
Loop to next batch
```

**Error Handling in Orchestrator:**

```python
try:
    process_batch()
except MigrationTimeoutError:
    # Batch timed out - checkpoint partial progress and exit
    # Next run will resume from here
    break

except MigrationError as e:
    if is_transient(e):
        # Retry on next run
        log_error(e)
        break
    else:
        # Permanent error - log and continue with next batch
        log_error(e)
```

### 8. CLI Entry Point (`main.py`)

**Responsibility:** Command-line interface and application bootstrap.

**Features:**
- Argument parsing
- Configuration loading
- Logging setup
- Error handling
- Exit codes

**Usage Examples:**
```bash
# Basic
python main.py

# With options
python main.py --svn-url https://svn.example.com --git-url https://github.com/org/repo.git --batch-size 100

# Resume
python main.py --resume

# Fresh start
python main.py --fresh

# Debug
python main.py --log-level DEBUG
```

## Key Design Patterns

### 1. Retry with Exponential Backoff

**Pattern:**
```python
retry_with_backoff(
    func=operation,
    max_attempts=3,
    base_delay=5,           # 5 seconds
    backoff_multiplier=3.0  # 5, 15, 45 seconds
)
```

**Benefits:**
- Handles transient network failures
- Avoids overwhelming failing services
- Configurable for different scenarios

### 2. Checkpoint and Resume

**Pattern:**
```
1. Create checkpoint (INSERT INTO migration_checkpoints)
2. Mark as IN_PROGRESS
3. Perform operation
4. Mark as COMPLETED with result
```

**Benefits:**
- No duplicate processing
- Resume from exact stopping point
- Atomic operations prevent partial failures

### 3. Batch Processing with Timeout

**Pattern:**
```
FOR each batch:
  timeout.start()
  FOR each item in batch:
    IF timeout.should_stop_batch():
      break  # Exit early, save progress
    process(item)
  push_batch()
```

**Benefits:**
- Prevents long-running operations
- Allows graceful degradation
- Predictable checkpoint intervals

### 4. Layered Architecture

Each layer has single responsibility:
- Config: Load settings
- Logger: Output events
- Error Handler: Handle failures
- State: Persist progress
- Adapters: Interface with external systems
- Orchestrator: Coordinate workflow

**Benefits:**
- Testable in isolation
- Easy to extend
- Clear dependencies

## Incremental Migration Strategy

### First Run
```
Initialize database with total revisions
Fetch revisions 1-50
Create checkpoints for 1-50
Create Git commits
Push to GitHub
Mark checkpoints completed
```

### Resume After Failure
```
Query database: SELECT MAX(svn_revision) WHERE status='completed'
Get last_completed = 50
Fetch revisions 51-100
Create checkpoints for 51-100 (skips 1-50 entirely)
Create Git commits
Push to GitHub
```

**Key Advantage:** Zero duplicate processing, resume from exact point

## Timeout Handling

### Problem
SVN can hang on large commits, Git push can be slow, network can be unstable.

### Solution
Each batch has configurable timeout (default 300s):

1. Start timer at beginning of batch
2. Check timer before processing each revision
3. If approaching 90% of timeout, stop batch
4. Save progress to database
5. Next run resumes from saved checkpoint

### Code
```python
timeout_manager = TimeoutManager(300)  # 5 minutes
timeout_manager.start()

for revision in batch:
    if timeout_manager.should_stop_batch(threshold_percent=90):
        break  # Stop at 90% to avoid going over
    process(revision)
```

## Extensibility Points

### 1. Add Branch/Tag Support
Extend `svn_adapter.py` with:
- `fetch_branches()` to get branch list
- `fetch_tags()` to get tag list
Extend `git_adapter.py` with:
- `create_branch()` to create branches
- `create_tag()` to create tags

### 2. Add Custom Mapping
Create post-processor to:
- Strip SVN-specific metadata from commit messages
- Transform directory structure
- Merge multiple SVN repos into one Git repo

### 3. Add Validation
After migration completes:
```python
def validate_migration():
    svn_count = svn.get_revision_count()
    git_count = git.get_commit_count()
    assert svn_count == git_count, "Commit count mismatch!"
```

### 4. Add Reporting
Generate HTML report showing:
- Migration timeline
- Commits per day
- Authors breakdown
- Error summary

## Performance Considerations

### 1. Batch Size
- **Too small (10):** Many roundtrips, slower overall
- **Too large (500):** Risk of timeout, hard to debug
- **Optimal (50-100):** Balance between throughput and recovery

### 2. Timeout
- **Too short (60s):** Frequent timeouts, slow progress
- **Too long (1800s):** Hard to handle failures quickly
- **Optimal (300s):** Reasonable for most operations

### 3. Database Queries
- Indexed on `svn_revision` and `status` for fast lookups
- Batch operations use transactions for efficiency
- Old log entries can be archived after migration

## Security Considerations

1. **Credentials:** Never hardcode, use environment variables only
2. **Logs:** Don't log passwords, only log up to first 8 chars of tokens
3. **Database:** SQLite file contains no credentials, only Git SHAs
4. **Network:** Use HTTPS for SVN and Git URLs
5. **Tokens:** Use personal access tokens with minimal required permissions

## Debugging

### Enable Debug Logging
```bash
python main.py --log-level DEBUG
```

### Check Database State
```bash
sqlite3 migration_state.db
SELECT svn_revision, status, git_commit_sha FROM migration_checkpoints 
WHERE status != 'completed'
ORDER BY svn_revision;
```

### Monitor Network Operations
```bash
# On Windows, use Wireshark or similar
# Enable verbose SVN logging
SVN_DEBUG=1 python main.py
```

## Testing

See `tests.py` for unit tests covering:
- Configuration loading
- Error classification and retry logic
- State management operations
- SVN metadata parsing
- Migration summary calculations

Run tests:
```bash
python -m unittest tests.py -v
```

---

**For questions or extensions, refer to the README.md and code comments.**

