# SVN to Git Migration Tool

A production-grade Python tool for migrating complete SVN repositories to Git/GitHub with full version history, comprehensive logging, and incremental resumable checkpoints.

## Features

✅ **Full Version History Migration** - Migrate all SVN commits with original authors and timestamps  
✅ **Incremental/Resumable** - If migration fails, resume from last checkpoint (not from beginning)  
✅ **Timeout Prevention** - Batch processing with configurable timeouts prevents long hangs  
✅ **Comprehensive Logging** - Detailed logs to file and console with multiple log levels  
✅ **State Persistence** - SQLite database tracks progress, allows recovery after crashes  
✅ **Authentication Support** - SVN and Git authentication with credentials/tokens  
✅ **Error Recovery** - Retry logic with exponential backoff for transient failures  
✅ **Modular Architecture** - Clean separation of concerns (SVN, Git, State, Orchestrator)  

## Architecture

```
┌─────────────────────────────────┐
│      main.py (CLI Entry)        │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│   Configuration Manager         │
│   (config.py)                   │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│   Migration Orchestrator        │
│   (orchestrator.py)             │
├────────────┬────────────────────┤
│            │                    │
│   ┌────────▼────────┐  ┌────────▼─────────┐
│   │  SVN Adapter    │  │  Git Adapter    │
│   │(svn_adapter.py) │  │(git_adapter.py) │
│   └─────────────────┘  └─────────────────┘
│
│   ┌─────────────────────────────┐
│   │   State Manager             │
│   │  (state_manager.py)         │
│   │  SQLite: Progress Tracking  │
│   └─────────────────────────────┘
│
│   ┌─────────────────────────────┐
│   │  Error Handler              │
│   │ (error_handler.py)          │
│   │ Retry, Recovery, Logging    │
│   └─────────────────────────────┘
└─────────────────────────────────┘
```

## Installation

### Prerequisites

- **Python 3.8+**
- **Git** (https://git-scm.com)
- **Subversion (SVN)** (https://subversion.apache.org)
- GitHub account with repository access

### Setup

1. **Clone or download this repository**

   ```bash
   git clone <your-repo-url> svn2git
   cd svn2git
   ```

2. **Create Python virtual environment**

   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Configure credentials**

   Copy the example files and fill in your credentials:

   ```bash
   cp .env.example .env
   cp config.yaml.example config.yaml
   ```

   Edit `.env` with your actual credentials:
   ```
   SVN_URL=https://svn.example.com/project
   SVN_USERNAME=your_username
   SVN_PASSWORD=your_password
   GIT_URL=https://github.com/yourorg/yourrepo.git
   GITHUB_TOKEN=your_github_token
   ```

   Or edit `config.yaml` with your settings.

## Usage

### Basic Migration

Start migration with default settings:

```bash
python main.py
```

### With Command-Line Arguments

```bash
# Specify URLs directly
python main.py \
  --svn-url https://svn.example.com/project \
  --git-url https://github.com/org/repo.git

# Set batch size and timeout
python main.py --batch-size 100 --timeout 600

# Use custom config file
python main.py --config myconfig.yaml

# Start fresh (clear state and restart)
python main.py --fresh

# Resume from last checkpoint (default behavior)
python main.py --resume

# Set log level
python main.py --log-level DEBUG
```

### Command-Line Options

```
--svn-url URL              Source SVN repository URL
--git-url URL              Target Git repository URL
--config FILE              Path to YAML config file (default: config.yaml)
--resume                   Resume from last checkpoint (default: True)
--fresh                    Start fresh (clears state, overrides --resume)
--log-level LEVEL          Logging level: DEBUG, INFO, WARNING, ERROR
--batch-size NUM           Number of revisions per batch
--timeout SECONDS          Batch timeout in seconds
```

## Configuration

### Configuration File (config.yaml)

```yaml
# SVN Configuration
svn_url: https://svn.example.com/project/trunk
svn_username: username
svn_password: password

# Git Configuration
git_url: https://github.com/org/repo.git
git_token: github_token
git_author_name: Migration Bot
git_author_email: bot@example.com

# Migration Settings
batch_size: 50              # Revisions per batch
batch_timeout_seconds: 300  # 5 minutes per batch
retry_attempts: 3           # Retries for transient errors

# Logging
log_level: INFO
log_dir: ./logs

# Database
db_path: ./migration_state.db
```

### Environment Variables

All configuration values can be overridden with environment variables:

```bash
export SVN_URL=https://svn.example.com/project
export SVN_USERNAME=user
export SVN_PASSWORD=pass
export GIT_URL=https://github.com/org/repo.git
export GITHUB_TOKEN=token
export BATCH_SIZE=100
export BATCH_TIMEOUT_SECONDS=600
export LOG_LEVEL=DEBUG
```

## How It Works

### Batch Processing Pipeline

1. **Fetch SVN Revisions** → Retrieve batch of N revisions from SVN
2. **Create Checkpoints** → Record pending revisions in SQLite
3. **Create Git Commits** → Convert each SVN revision to Git commit
4. **Push to GitHub** → Push batch to remote repository
5. **Update State** → Mark batch as completed in database
6. **Checkpoint** → Save progress, ready for resume

### Timeout Prevention

- Each batch has a configurable timeout (default: 5 minutes)
- If processing approaches timeout threshold (90%), batch stops early and checkpoints
- Resuming continues from where it left off, not from the beginning
- Large files (> 100MB) get extended timeout multiplier

### Error Recovery

**Transient Errors** (Network, temporary failures):
- Automatic retry with exponential backoff
- Configurable retry attempts (default: 3)
- Recorded in error log

**Permanent Errors** (Auth, data validation):
- Logged and flagged for manual review
- Migration continues with next batch
- Can be investigated later in error logs

**Timeout Errors**:
- Current batch checkpoint is saved
- On restart, migration resumes from saved checkpoint
- No commits are lost or duplicated

### State Tracking

The SQLite database tracks:

- **migration_metadata** - Overall migration progress
- **migration_checkpoints** - Per-revision status and Git commit SHA
- **migration_errors** - Detailed error log with recovery hints

Query migration status anytime:

```bash
sqlite3 migration_state.db "SELECT * FROM migration_metadata;"
sqlite3 migration_state.db "SELECT svn_revision, status, git_commit_sha FROM migration_checkpoints LIMIT 10;"
```

## Monitoring & Logging

### Log Files

Logs are written to `./logs/migration_TIMESTAMP.log`

View logs in real-time:

```bash
# Windows (PowerShell)
Get-Content -Path .\logs\migration_*.log -Wait

# macOS/Linux
tail -f logs/migration_*.log
```

### Log Levels

- **DEBUG** - Detailed diagnostic information
- **INFO** - General progress information (default)
- **WARNING** - Warning messages (non-fatal issues)
- **ERROR** - Error messages (problems that need attention)

### Migration Summary

At completion, you'll see:

```
================================================================================
MIGRATION SUMMARY
================================================================================
Total Revisions: 1000
Completed: 1000
Failed: 0
Pending: 0
In Progress: 0
Progress: 100.00%
================================================================================
```

## Handling Failures

### Migration Interrupted (Ctrl+C)

Simply restart migration:

```bash
python main.py  # Automatically resumes from checkpoint
```

### Connection Lost

Restart when connection is restored:

```bash
python main.py  # Resumes from checkpoint
```

### Authentication Failed

Update credentials in `.env` or `config.yaml` and restart:

```bash
# Edit .env with correct credentials
python main.py --resume  # Resumes with new credentials
```

### Verify Progress

Check database to see current progress:

```bash
sqlite3 migration_state.db \
  "SELECT COUNT(*) as total, \
          SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed, \
          SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed \
   FROM migration_checkpoints;"
```

## Performance Tuning

### For Large Repositories (100k+ commits)

Decrease batch size and timeout:

```bash
python main.py --batch-size 25 --timeout 180  # 25 commits per batch, 3 min timeout
```

### For Small Repositories

Increase batch size for faster processing:

```bash
python main.py --batch-size 200 --timeout 600  # 200 commits per batch, 10 min timeout
```

### For Slow Network

Increase timeout and retry attempts:

```bash
# In config.yaml
batch_timeout_seconds: 600
retry_attempts: 5
```

## Troubleshooting

### "SVN command not found"

Install Subversion:

- **Windows**: Use Chocolatey: `choco install svn`
- **macOS**: Use Homebrew: `brew install subversion`
- **Linux**: `sudo apt-get install subversion`

### "Git authentication failed"

Verify credentials:

```bash
# Test SVN connection
svn info https://your-svn-url --username username --password password

# Test Git authentication
git clone https://github.com/org/repo.git test-dir
```

### Database locked error

Database might be corrupted. Remove and restart:

```bash
rm migration_state.db
python main.py --fresh  # Start fresh migration
```

### Migration hangs

Increase timeout or decrease batch size:

```bash
python main.py --timeout 900 --batch-size 25
```

## Advanced Usage

### Dry Run (Check Configuration)

```bash
python main.py --log-level DEBUG | head -50
```

### Continue After Error

If migration stops due to error, fix the issue and resume:

```bash
python main.py --resume  # Continues from last checkpoint
```

### Check Migration Status Without Resuming

```bash
sqlite3 migration_state.db \
  "SELECT status, COUNT(*) as count FROM migration_checkpoints GROUP BY status;"
```

### Manual Recovery

If database gets corrupted:

```bash
# Backup and remove database
mv migration_state.db migration_state.db.backup

# Start fresh (but your Git repo will have commits from last run)
python main.py --fresh
```

## Project Structure

```
svn2git/
├── main.py                 # CLI entry point
├── config.py               # Configuration management
├── logger.py               # Logging framework
├── error_handler.py        # Error handling & recovery
├── state_manager.py        # SQLite state tracking
├── svn_adapter.py          # SVN repository interface
├── git_adapter.py          # Git repository interface
├── orchestrator.py         # Migration orchestrator
├── config.yaml.example     # Example config file
├── .env.example            # Example environment variables
├── requirements.txt        # Python dependencies
├── migration_state.db      # SQLite database (created on first run)
└── logs/                   # Log files directory (created on first run)
```

## Contributing

To contribute improvements:

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

[Your License Here]

## Support

For issues, questions, or feature requests, please open an issue in the repository.

## FAQ

**Q: Can I migrate specific revisions only?**  
A: Currently, the tool migrates all revisions. You can modify `orchestrator.py` to accept revision range arguments.

**Q: What if my SVN repo has branches/tags?**  
A: Current implementation treats SVN as a single trunk. See the architecture section for how to extend branch/tag handling.

**Q: Is it safe to run multiple migrations simultaneously?**  
A: No, use separate database paths (`DB_PATH` env var) if you need parallel migrations.

**Q: Can I migrate to GitHub automatically?**  
A: Yes, use a GitHub Personal Access Token in `GITHUB_TOKEN`. Repository must already exist.

**Q: How do I verify the migration was successful?**  
A: Compare commit counts: `git log --oneline | wc -l` should match SVN revision count.

---

**Happy Migrating! 🚀**

