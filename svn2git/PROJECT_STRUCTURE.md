# Project Structure

## Organized Python Package Layout

The project now follows standard Python package structure:

```
svn2git/
├── svn2git_tool/                 # Main package
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # CLI entry point
│   │
│   ├── core/                    # Core business logic
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   ├── state_manager.py    # SQLite state persistence
│   │   └── orchestrator.py     # Migration orchestration
│   │
│   ├── adapters/                # External system adapters
│   │   ├── __init__.py
│   │   ├── svn_adapter.py      # SVN repository interface
│   │   └── git_adapter.py      # Git repository interface
│   │
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── logger.py           # Logging framework
│       └── error_handler.py    # Error handling & recovery
│
├── tests/                       # Test suite
│   ├── __init__.py
│   └── test_all.py            # Unit tests
│
├── setup.py                    # Package installation script
├── pyproject.toml              # Modern Python project config
├── requirements.txt            # Runtime dependencies
├── README.md                   # Project documentation
└── [other documentation files]
```

## Running the Tool

### Option 1: Direct execution (development)
```bash
python -m svn2git_tool.main --svn-url <url> --git-url <url>
```

### Option 2: Installed as package
```bash
pip install -e .  # Install in editable mode
svn2git --svn-url <url> --git-url <url>
```

### Option 3: Using module directly
```bash
python -m svn2git_tool --svn-url <url> --git-url <url>
```

## Running Tests

```bash
# Using unittest
python -m unittest discover -s tests -p "test_*.py"

# Or using pytest (if installed)
pytest tests/

# With coverage
pytest --cov=svn2git_tool tests/
```

## Module Organization

### svn2git_tool.core
Contains core business logic and orchestration:
- **Config**: Configuration management from files and environment
- **StateManager**: SQLite database for state persistence
- **MigrationOrchestrator**: Main migration workflow controller
- **TimeoutManager**: Timeout enforcement for batch processing

### svn2git_tool.adapters
Contains interfaces to external systems:
- **SVNAdapter**: Interface to SVN repositories
- **GitAdapter**: Interface to Git repositories

### svn2git_tool.utils
Contains utility functions and frameworks:
- **Logger**: Structured logging with color support
- **ErrorHandler**: Error classification, retry logic, recovery strategies

## Import Examples

### Before (flat structure)
```python
from config import Config
from logger import get_logger
from error_handler import MigrationError
```

### After (package structure)
```python
from svn2git_tool.core import Config
from svn2git_tool.utils import get_logger, MigrationError
from svn2git_tool.adapters import SVNAdapter, GitAdapter
```

## Benefits of This Structure

✅ **Standard Python packaging** - Follows PEP standards  
✅ **Easy installation** - Can be installed as a package  
✅ **Clear organization** - Related code grouped together  
✅ **Scalability** - Easy to add new modules/packages  
✅ **Namespace isolation** - Avoid name conflicts  
✅ **Professional** - Industry-standard structure  
✅ **Testing** - Standard test discovery patterns  

## Installation for Development

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Now you can import from anywhere
python -c "from svn2git_tool.core import Config; print(Config)"
```

## Next Steps

1. **Review structure**: The files are now properly organized
2. **Run tests**: `python -m unittest discover -s tests`
3. **Start tool**: `python -m svn2git_tool.main --help`
4. **Install package**: `pip install -e .`

