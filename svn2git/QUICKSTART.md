# Quick Start Guide - SVN to Git Migration

## 5-Minute Setup

### 1. Install Dependencies

```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Install SVN and Git (if not already installed)

**Windows:**
```bash
# Using Chocolatey
choco install git svn
```

**macOS:**
```bash
brew install git subversion
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install git subversion
```

### 3. Configure Credentials

```bash
# Copy example files
cp .env.example .env
cp config.yaml.example config.yaml

# Edit .env with your credentials
# On Windows:
notepad .env

# On macOS/Linux:
nano .env
```

Fill in your credentials:
```
SVN_URL=https://your-svn-server.com/repo
SVN_USERNAME=your_username
SVN_PASSWORD=your_password
GIT_URL=https://github.com/yourorg/yourrepo.git
GITHUB_TOKEN=your_personal_access_token
```

### 4. Start Migration

```bash
python main.py
```

That's it! The tool will:
- Connect to your SVN server
- Fetch all commits with their history
- Create equivalent Git commits
- Push to your GitHub repository
- Save progress in a local database

## Monitor Progress

Watch the logs in real-time:

**Windows (PowerShell):**
```bash
Get-Content -Path .\logs\migration_*.log -Wait
```

**macOS/Linux:**
```bash
tail -f logs/migration_*.log
```

## Handling Interruptions

If the migration stops (network issue, timeout, etc.):

```bash
# Simply restart - it will resume from where it left off
python main.py
```

The tool remembers your progress in `migration_state.db`.

## Common Issues

### "SVN command not found"
Install Subversion (see "Install SVN and Git" section above)

### "Git authentication failed"
Make sure your GITHUB_TOKEN is valid and the repository exists

### "Timeout" errors
Decrease batch size:
```bash
python main.py --batch-size 25 --timeout 600
```

## Advanced Options

```bash
# Start fresh (ignore previous progress)
python main.py --fresh

# Use custom config file
python main.py --config myconfig.yaml

# Debug mode with detailed logging
python main.py --log-level DEBUG

# Custom batch settings
python main.py --batch-size 100 --timeout 600
```

## Verify Success

After migration completes:

```bash
# Check local Git commits
git log --oneline | wc -l

# Should match your SVN revision count
# Compare in database:
sqlite3 migration_state.db "SELECT COUNT(*) FROM migration_checkpoints WHERE status='completed';"
```

## Need Help?

See the full README.md for:
- Detailed configuration
- Troubleshooting guide
- Performance tuning
- Architecture overview

---

**Happy Migrating! 🚀**

