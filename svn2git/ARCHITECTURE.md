# System Architecture & Data Flow

## High-Level System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   SVN to Git Migration System                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │             Command-Line Interface (CLI)                │   │
│  │  Arguments: --svn-url, --git-url, --batch-size, etc.   │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │           Configuration Manager                         │   │
│  │  - Load from .env, config.yaml, CLI args              │   │
│  │  - Validate required settings                         │   │
│  │  - Provide typed property accessors                   │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │         Logging Framework Setup                        │   │
│  │  - Console output (colored)                           │   │
│  │  - File output (detailed)                             │   │
│  │  - Multiple log levels                                │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                      │
│  ┌────────────────────────▼─────────────────────────────────┐   │
│  │         Migration Orchestrator                         │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │  Batch Processing Pipeline                       │  │   │
│  │  │  1. Initialize state & progress tracking        │  │   │
│  │  │  2. Loop: Process batches until complete        │  │   │
│  │  │  3. Per Batch:                                  │  │   │
│  │  │     a) Fetch N revisions from SVN              │  │   │
│  │  │     b) Create Git commits                       │  │   │
│  │  │     c) Push to GitHub                           │  │   │
│  │  │     d) Checkpoint progress                      │  │   │
│  │  │  4. Generate final summary                      │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  │                    │                                      │   │
│  │    ┌───────────────┼───────────────┐                     │   │
│  │    │               │               │                     │   │
│  └────┼───────────────┼───────────────┼─────────────────────┘   │
│       │               │               │                         │
│  ┌────▼──┐     ┌──────▼──────┐  ┌───▼──────┐                   │
│  │ SVN   │     │ Git         │  │ State    │                   │
│  │Adapter│     │ Adapter     │  │ Manager  │                   │
│  ├───────┤     ├─────────────┤  ├──────────┤                   │
│  │ Fetch │     │ Create      │  │ SQLite   │                   │
│  │ revs  │     │ commits     │  │ Database │                   │
│  │ Auth  │     │ Push remote │  │ Progress │                   │
│  │ Parse │     │ Auth        │  │ tracking │                   │
│  └───────┘     └─────────────┘  └──────────┘                   │
│       │               │               │                         │
│       ▼               ▼               ▼                         │
│  ┌─────────────────────────────────────────────┐              │
│  │         Error Handler                       │              │
│  │  - Classify errors (transient/permanent)    │              │
│  │  - Retry with exponential backoff           │              │
│  │  - Log errors with context                  │              │
│  │  - Recovery strategies                      │              │
│  └─────────────────────────────────────────────┘              │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
    SVN Server           GitHub Server          SQLite Database
    (Remote)             (Remote)                (Local)
```

## Detailed Data Flow

### Flow 1: Initialization Phase

```
main.py
  │
  ├─ Parse CLI arguments
  │
  ├─ Load Configuration
  │   ├─ Read environment variables
  │   ├─ Read config.yaml
  │   └─ Validate required settings
  │
  ├─ Setup Logging
  │   ├─ Create log directory
  │   ├─ Initialize console handler
  │   └─ Initialize file handler
  │
  └─ Initialize Orchestrator
      │
      ├─ Create SVN Adapter
      │   └─ Verify SVN connection
      │
      ├─ Create Git Adapter
      │   ├─ Clone or open Git repo
      │   └─ Configure user info
      │
      ├─ Create State Manager
      │   └─ Initialize SQLite database
      │
      └─ Query SVN for total revisions
          └─ Initialize migration metadata in DB
```

### Flow 2: Migration Batch Processing

```
Batch #N (Revisions 101-150)
│
├─ Initialize TimeoutManager (300 second limit)
│
├─ Fetch SVN Revisions 101-150
│   │
│   ├─ Execute: svn log --xml -r 101:150 <url>
│   │
│   ├─ Parse XML response
│   │   └─ Extract: revision, author, timestamp, message
│   │
│   └─ Return List[SVNRevisionMetadata]
│
├─ For Each Revision (101, 102, 103, ...):
│   │
│   ├─ Check timeout: should_stop_batch()?
│   │   └─ If approaching 90% timeout, break early
│   │
│   ├─ Create Database Checkpoint
│   │   │
│   │   └─ INSERT INTO migration_checkpoints
│   │       (svn_revision, batch_number, status='pending')
│   │
│   ├─ Mark as IN_PROGRESS
│   │   │
│   │   └─ UPDATE status = 'in_progress'
│   │
│   ├─ Create Git Commit
│   │   │
│   │   ├─ Preserve original author info
│   │   ├─ Preserve original timestamp
│   │   ├─ Use original commit message
│   │   │
│   │   └─ Return Git commit SHA
│   │
│   ├─ Mark as COMPLETED with Git SHA
│   │   │
│   │   └─ UPDATE git_commit_sha, status = 'completed'
│   │
│   └─ Log progress
│
├─ Push Batch to GitHub
│   │
│   ├─ Execute: git push origin master
│   │
│   └─ Handle auth if needed
│
├─ Update Migration Metadata
│   │
│   └─ UPDATE last_checkpoint_revision = 150
│
└─ Next Batch or Complete?
    ├─ If more revisions: goto Batch #N+1
    └─ If done: goto Completion
```

### Flow 3: Error Handling

```
Operation Fails (e.g., Git push error)
│
├─ Classify Error
│   │
│   ├─ If Transient (network):
│   │   │
│   │   ├─ Log warning with retry info
│   │   │
│   │   ├─ Retry with Exponential Backoff
│   │   │   ├─ Attempt 1: wait 5s, retry
│   │   │   ├─ Attempt 2: wait 15s, retry
│   │   │   └─ Attempt 3: wait 45s, fail
│   │   │
│   │   ├─ Record error in migration_errors table
│   │   │
│   │   └─ Return to previous state, next run will retry
│   │
│   └─ If Permanent (auth, validation):
│       │
│       ├─ Log error with details
│       │
│       ├─ Mark checkpoint as FAILED
│       │
│       ├─ Record error in migration_errors table
│       │
│       └─ Continue with next item (don't retry)
│
└─ Migration continues or stops based on error type
```

### Flow 4: Resume from Checkpoint

```
Restart after interruption
│
├─ Load configuration (same as before)
│
├─ Query database: SELECT MAX(svn_revision) WHERE status='completed'
│   └─ Get last_completed = 150
│
├─ Next start revision = 151
│
├─ Continue processing from revision 151
│   ├─ Fetch revisions 151-200
│   ├─ Create commits
│   └─ Push to GitHub
│
└─ No duplicate processing!
    (Revisions 1-150 already completed and present in Git)
```

## Database Schema Relationships

```
migration_metadata (1)
  ├─ svn_url
  ├─ git_url
  ├─ start_time
  ├─ last_checkpoint_revision ──┐
  │   total_svn_revisions       │
  │   migration_status          │
  │                             │
  └─ Last-Known Good: ──────────┤
      On resume: SELECT MAX(svn_revision) WHERE status='completed'
                                │
migration_checkpoints (N)       │
  ├─ svn_revision (UNIQUE) ◄────┘
  ├─ git_commit_sha
  ├─ batch_number
  ├─ status (pending/in_progress/completed/failed)
  ├─ author
  ├─ timestamp
  ├─ message
  ├─ retry_count
  └─ error_message

migration_errors (N)
  ├─ svn_revision (FK to checkpoints)
  ├─ batch_number
  ├─ error_type
  ├─ error_message
  ├─ error_timestamp
  └─ resolved
```

## Batch Processing Timeline

```
Time (seconds)
     │
   0 │ ┌─ Batch Start
     │ │  Initialize timeout manager
     │ │
  30 │ ├─ Fetch SVN revisions (50 revs)
     │ │  Parse metadata
     │ │
  60 │ ├─ Create Git commits (50 commits)
     │ │  1 commit per second
     │ │
 240 │ ├─ Push to GitHub
     │ │
 270 │ ├─ Update database
     │ │
 290 │ ├─ Approaching 90% of timeout (270/300)
     │ │  Stop batch, save checkpoint
     │ │
 300 │ └─ End Batch / Timeout Limit
     │
 320 │ ┌─ Next Batch (if resumed)
     │ │  Fetch revisions 51-100
     │ │  ...
```

## State Transitions During Migration

```
START
  │
  ├─► PENDING (Create checkpoint)
  │      │
  │      ├─► IN_PROGRESS (Mark as processing)
  │      │      │
  │      │      ├─► COMPLETED (Success, save Git SHA)
  │      │      │      └─ [Can query by last completed]
  │      │      │
  │      │      └─► FAILED (Error occurred)
  │      │             └─ [Logged in migration_errors]
  │      │
  │      └─ [If timeout before processing]
  │         └─ PENDING (Stays pending for next run)
  │
  └─ SKIPPED (If already completed on resume)
       └─ [Will not be reprocessed]
```

## Timeout and Batch Boundary Strategy

```
Batch Processing Window: 300 seconds
├─────────────────────────────────────────┤
0                                      300

Threshold at 90%: 270 seconds
├─────────────────────────┤ ┌─────────────┤
0                        270             300
(Process normally)     (Check flag)  (Hard stop)

When timeout.should_stop_batch():
  └─ Returns True at 270 seconds
  └─ Batch stops gracefully
  └─ Remaining revisions in batch stay PENDING
  └─ Next run processes them
  └─ No timeout errors, clean checkpoints
```

## Concurrency & Safety

### Single-Instance Design (Recommended)

```
migration_state.db
│
└─ Only ONE migration process at a time
   │
   ├─ SQLite database locked during writes
   ├─ Prevents duplicate processing
   ├─ No conflicts between processes
   └─ Clean checkpoint guarantees
```

### Multiple Instances (Not Recommended, but possible)

```
migration_state_1.db ──► GitHub /branch-a
migration_state_2.db ──► GitHub /branch-b

├─ Use different DB_PATH for each instance
├─ Use different log files
├─ Use different temp_git_repo paths
└─ Manually coordinate Git branches
```

## Performance Metrics

```
Typical Batch Processing:
├─ Batch Size: 50 revisions
├─ Fetch Time: 10-30 seconds
├─ Git Commit Time: 30-60 seconds (1-2 sec/commit)
├─ Push Time: 20-40 seconds
├─ Database Update: < 1 second
└─ Total Per Batch: ~80-150 seconds

Throughput:
├─ Small repos (< 1k commits): 5-10 minutes
├─ Medium repos (10k commits): 1-2 hours
├─ Large repos (100k+ commits): 8-24 hours
└─ Very large repos: May need multiple batches

Bottlenecks:
├─ SVN fetch (depends on server performance)
├─ Git push (depends on network)
├─ Database writes (SQLite is fast)
└─ Git commit creation (sequential)
```

## Security Data Flow

```
Configuration
├─ .env (local, never committed) ✓
├─ config.yaml (local, can be shared)
└─ Environment variables (secure)
     │
     ├─ NOT stored in database ✓
     ├─ NOT logged (except first chars) ✓
     └─ NOT visible in process list ✓

Database (migration_state.db)
├─ Contains: SVN revisions, Git SHAs, timestamps ✓
├─ Contains: Error messages (no credentials) ✓
├─ Does NOT contain: Passwords, tokens ✓
└─ Does NOT contain: SVN/Git authentication ✓

Logs
├─ Console: Progress only ✓
├─ File: Detailed but no credentials ✓
└─ Passwords/tokens: Never logged ✓
```

## Extensibility Hooks

```
Add branch/tag support:
├─ Extend SVN Adapter: fetch_branches(), fetch_tags()
├─ Extend Git Adapter: create_branch(), create_tag()
└─ Extend Orchestrator: process_branches()

Add post-processing:
├─ Add transform layer after Git commit
├─ Modify files before committing
└─ Validate commits after push

Add metrics/monitoring:
├─ Hook into progress updates
├─ Send metrics to monitoring system
└─ Generate HTML reports

Add custom error handling:
├─ Extend error_handler.py
├─ Add custom retry strategies
└─ Add notifications/alerts
```

---

**This architecture ensures:**
- ✅ Fault tolerance (resume from checkpoints)
- ✅ Data consistency (atomic operations)
- ✅ Timeout prevention (batch boundaries)
- ✅ Error resilience (retry + classification)
- ✅ Observable (comprehensive logging)
- ✅ Extensible (clean interfaces)

