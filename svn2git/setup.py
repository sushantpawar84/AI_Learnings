#!/usr/bin/env python
"""
Setup script for SVN to Git Migration Tool
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

setup(
    name="svn2git-tool",
    version="1.0.0",
    description="A production-grade tool for migrating SVN repositories to Git with full version history",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SVN to Git Migration Team",
    author_email="migration@example.com",
    url="https://github.com/yourusername/svn2git",
    packages=find_packages(exclude=["tests", "tests.*", "docs", "*.egg-info"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv==1.0.0",
        "gitpython==3.1.43",
        "pysvn==1.9.21",
        "requests==2.31.0",
        "pyyaml==6.0.1",
        "colorlog==6.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "svn2git=svn2git_tool.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Version Control :: RCS",
        "Topic :: System :: Distributed Computing",
        "Topic :: System :: Systems Administration",
    ],
    keywords="svn git migration version-control repository",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/svn2git/issues",
        "Source": "https://github.com/yourusername/svn2git",
        "Documentation": "https://github.com/yourusername/svn2git/blob/main/README.md",
    },
)

