# Deployment & Setup Checklist

## Pre-Deployment

- [ ] **Python Environment**
  - [ ] Python 3.8+ installed
  - [ ] Virtual environment created: `python -m venv venv`
  - [ ] Virtual environment activated
  - [ ] Dependencies installed: `pip install -r requirements.txt`

- [ ] **External Tools**
  - [ ] Git installed (`git --version`)
  - [ ] Subversion (SVN) installed (`svn --version`)
  - [ ] SVN server is accessible
  - [ ] GitHub account and repository created

- [ ] **Credentials**
  - [ ] SVN username and password ready
  - [ ] GitHub Personal Access Token generated
  - [ ] Credentials stored in `.env` (copy from `.env.example`)
  - [ ] `.env` file added to `.gitignore` ✓

- [ ] **Configuration**
  - [ ] `config.yaml` created (copy from `config.yaml.example`)
  - [ ] SVN URL verified
  - [ ] Git URL verified
  - [ ] Batch size set appropriately
  - [ ] Timeout set appropriately
  - [ ] Log directory configured

## Pre-Migration Testing

- [ ] **Connection Tests**
  ```bash
  # Test SVN connection
  svn info https://your-svn-url --username user --password pass
  
  # Test Git authentication
  git clone https://github.com/yourorg/yourrepo.git test-dir
  rm -r test-dir
  ```

- [ ] **Database Setup**
  - [ ] `migration_state.db` will be created automatically
  - [ ] Logs directory will be created automatically
  - [ ] Database path is writable

- [ ] **Dry Run**
  ```bash
  # Run with debug logging to verify setup
  python main.py --log-level DEBUG
  ```
  Check that:
  - [ ] Configuration loads successfully
  - [ ] SVN connection established
  - [ ] Git repository opened/cloned
  - [ ] Database initialized
  - [ ] Logging working correctly

## During Migration

- [ ] **Monitoring**
  - [ ] Monitor logs in real-time: `tail -f logs/migration_*.log`
  - [ ] Check database periodically: `sqlite3 migration_state.db "SELECT * FROM migration_metadata;"`
  - [ ] System has sufficient disk space

- [ ] **Network**
  - [ ] Stable internet connection
  - [ ] No proxies or firewalls blocking SVN/Git
  - [ ] VPN connected if required

- [ ] **System**
  - [ ] Sufficient RAM (minimum 2GB recommended)
  - [ ] Sufficient disk space for temporary repo
  - [ ] No antivirus blocking database access

## Post-Migration

- [ ] **Verification**
  ```bash
  # Check database
  sqlite3 migration_state.db \
    "SELECT COUNT(*) as total, \
            SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as completed \
     FROM migration_checkpoints;"
  
  # Check Git commits
  cd temp_git_repo
  git log --oneline | wc -l
  
  # Should match total SVN revisions
  ```

- [ ] **Data Quality**
  - [ ] All revisions migrated (count matches)
  - [ ] Commit authors preserved
  - [ ] Commit timestamps preserved
  - [ ] Commit messages intact
  - [ ] File structure correct
  - [ ] No merge conflicts

- [ ] **GitHub Repository**
  - [ ] Commits visible in GitHub
  - [ ] History complete
  - [ ] Branches correct (if migrated)
  - [ ] Tags correct (if migrated)

- [ ] **Documentation**
  - [ ] Update team about migration
  - [ ] Update any CI/CD pipelines
  - [ ] Update repository links in documentation
  - [ ] Archive SVN access info if no longer needed

- [ ] **Cleanup**
  - [ ] Backup original SVN repository
  - [ ] Archive migration logs
  - [ ] Remove temporary files
  - [ ] Document any issues encountered

## Troubleshooting During Migration

### If migration hangs:
1. Check log file for errors
2. Verify network connection
3. Check firewall/proxy settings
4. Kill process (Ctrl+C)
5. Increase timeout and restart: `python main.py --timeout 600`

### If authentication fails:
1. Verify credentials in `.env`
2. Test manually: `svn info <url> --username user --password pass`
3. For Git: verify token has repo access
4. Update credentials and restart

### If database is locked:
1. No other instance should be running
2. Check for hanging processes
3. If corrupted: `rm migration_state.db` and restart with `--fresh`

### If commit count doesn't match:
1. Check database: `SELECT COUNT(*) FROM migration_checkpoints WHERE status='completed';`
2. Check for failed commits: `SELECT * FROM migration_checkpoints WHERE status='failed';`
3. Review error log: `SELECT * FROM migration_errors;`
4. Check Git: `git log --oneline | wc -l`

## Performance Tuning Checklist

- [ ] **For Large Repositories (100k+ commits)**
  - [ ] Decrease batch size: `--batch-size 25`
  - [ ] Increase timeout: `--timeout 600`
  - [ ] Monitor memory usage

- [ ] **For Small Repositories (< 1k commits)**
  - [ ] Increase batch size: `--batch-size 200`
  - [ ] Normal timeout: `--timeout 300`

- [ ] **For Slow Networks**
  - [ ] Increase timeout: `--timeout 600`
  - [ ] Increase retry attempts in config
  - [ ] Decrease batch size slightly

- [ ] **For Fast Networks**
  - [ ] Increase batch size: `--batch-size 150`
  - [ ] Standard timeout: `--timeout 300`

## Security Checklist

- [ ] **Credentials**
  - [ ] `.env` file NOT committed to Git
  - [ ] `.gitignore` includes `.env` ✓
  - [ ] No credentials in `config.yaml`
  - [ ] No credentials in logs
  - [ ] SVN password not shared

- [ ] **Git Token**
  - [ ] Token has minimum required permissions (repo only)
  - [ ] Token expiration set (if possible)
  - [ ] Token is not hardcoded anywhere

- [ ] **Database Security**
  - [ ] `migration_state.db` not shared publicly
  - [ ] Database permissions set correctly
  - [ ] Backup of database secured

- [ ] **After Migration**
  - [ ] Revoke SVN access if no longer needed
  - [ ] Revoke old Git token if creating new one
  - [ ] Update team about new Git repository location

## Automation Checklist (Optional)

If running automated migrations:

- [ ] **Scheduling**
  - [ ] Cron job or scheduled task configured
  - [ ] Error notifications set up
  - [ ] Success notifications set up

- [ ] **Environment**
  - [ ] All environment variables set in automation context
  - [ ] Working directory correct
  - [ ] Python venv path correct
  - [ ] File permissions allow read/write

- [ ] **Logging**
  - [ ] Automated logs aggregated
  - [ ] Log rotation configured
  - [ ] Error logs monitored

## Rollback Plan

If migration has issues:

1. **Keep SVN repository accessible** - Don't delete until verification complete
2. **Backup GitHub repository** - If possible, create backup branch
3. **Document issues** - Save error logs and state for analysis
4. **Reset GitHub** - If needed, reset to known good state
5. **Retry** - With adjusted settings (batch size, timeout)

## Final Sign-Off

- [ ] **Technical Review**
  - [ ] Code reviewed by team member
  - [ ] All tests pass
  - [ ] Documentation complete

- [ ] **Migration Execution**
  - [ ] SVN backup complete
  - [ ] Dry run successful
  - [ ] Go-ahead from team

- [ ] **Completion**
  - [ ] All data migrated successfully
  - [ ] Data quality verified
  - [ ] Team notified
  - [ ] Old VCS decommissioned (if decided)

---

## Quick Reference

### Start Fresh Migration
```bash
python main.py --fresh
```

### Resume Interrupted Migration
```bash
python main.py --resume
```

### Monitor Progress
```bash
tail -f logs/migration_*.log
```

### Check Database Status
```bash
sqlite3 migration_state.db "SELECT status, COUNT(*) FROM migration_checkpoints GROUP BY status;"
```

### Debug Information
```bash
python main.py --log-level DEBUG 2>&1 | tee debug.log
```

---

**Ready to migrate? Start with the QUICKSTART.md!**

