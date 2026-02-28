"""
Multi-Project Management System
Manages multiple projects with different configurations and datasets.
Supports file type handlers for extensibility and scaling to other project types.
"""

import os
import json
from typing import Dict, List, Any, Optional, Type
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod

from project_analyser.ai.dataset.codeIndexer import CodeIndexer, CodeChunk
from project_analyser.ai.dataset.datasetManager import IncrementalDatasetBuilder


@dataclass
class ProjectConfig:
    """Configuration for a project"""
    project_name: str
    project_root: str
    project_type: str  # "microservices", "documentation", "monolith", etc.
    file_types: List[str]  # Extensions to include: ['.java', '.py', '.md']
    description: str = ""
    model_name: str = "gpt2"
    training_config: Dict[str, Any] = None
    created_at: str = None
    last_updated: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()


class FileTypeHandler(ABC):
    """Abstract base class for file type handlers"""

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """Check if this handler can process the file"""
        pass

    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from file"""
        pass

    @abstractmethod
    def generate_training_description(self, content: str) -> str:
        """Generate training description for file content"""
        pass


class JavaFileHandler(FileTypeHandler):
    """Handler for Java files"""

    def can_handle(self, file_path: str) -> bool:
        return file_path.endswith('.java')

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return {}

        metadata = {
            "file_type": "java",
            "package": self._extract_package(content),
            "classes": self._extract_classes(content),
            "imports": self._extract_imports(content),
            "lines_of_code": len(content.split('\n'))
        }
        return metadata

    def generate_training_description(self, content: str) -> str:
        package = self._extract_package(content)
        classes = self._extract_classes(content)
        desc = f"Java file"
        if package:
            desc += f" in package {package}"
        if classes:
            desc += f" containing classes: {', '.join(classes)}"
        return desc

    @staticmethod
    def _extract_package(content: str) -> Optional[str]:
        import re
        match = re.search(r'package\s+([\w.]+)', content)
        return match.group(1) if match else None

    @staticmethod
    def _extract_classes(content: str) -> List[str]:
        import re
        matches = re.findall(r'(public\s+)?(class|interface)\s+(\w+)', content)
        return [match[2] for match in matches]

    @staticmethod
    def _extract_imports(content: str) -> List[str]:
        import re
        matches = re.findall(r'import\s+([\w.*]+)', content)
        return matches


class MarkdownFileHandler(FileTypeHandler):
    """Handler for Markdown documentation files"""

    def can_handle(self, file_path: str) -> bool:
        return file_path.endswith('.md')

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return {}

        metadata = {
            "file_type": "markdown",
            "headings": self._extract_headings(content),
            "lines": len(content.split('\n')),
            "has_code_blocks": "```" in content
        }
        return metadata

    def generate_training_description(self, content: str) -> str:
        headings = self._extract_headings(content)
        desc = "Documentation file"
        if headings:
            desc += f" with sections: {', '.join(headings[:3])}"
        return desc

    @staticmethod
    def _extract_headings(content: str) -> List[str]:
        import re
        matches = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        return matches


class YamlFileHandler(FileTypeHandler):
    """Handler for YAML configuration files"""

    def can_handle(self, file_path: str) -> bool:
        return file_path.endswith(('.yaml', '.yml'))

    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return {}

        metadata = {
            "file_type": "yaml",
            "top_level_keys": self._extract_keys(content),
            "lines": len(content.split('\n'))
        }
        return metadata

    def generate_training_description(self, content: str) -> str:
        keys = self._extract_keys(content)
        desc = "YAML configuration file"
        if keys:
            desc += f" with properties: {', '.join(keys[:5])}"
        return desc

    @staticmethod
    def _extract_keys(content: str) -> List[str]:
        import re
        matches = re.findall(r'^(\w+):\s*', content, re.MULTILINE)
        return list(set(matches))


class ProjectManager:
    """Manages multiple projects and their datasets"""

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.projects_dir = os.path.join(workspace_root, "projects")
        self.projects_config_file = os.path.join(self.projects_dir, "projects.json")

        os.makedirs(self.projects_dir, exist_ok=True)
        self.projects = self._load_projects()

        # Initialize file type handlers
        self.file_handlers = {
            '.java': JavaFileHandler(),
            '.md': MarkdownFileHandler(),
            '.yaml': YamlFileHandler(),
            '.yml': YamlFileHandler()
        }

    def create_project(self, config: ProjectConfig) -> bool:
        """Create a new project"""

        if config.project_name in self.projects:
            print(f"Project '{config.project_name}' already exists")
            return False

        print(f"Creating project: {config.project_name}")

        # Create project directory structure
        project_root = os.path.join(self.projects_dir, config.project_name)
        os.makedirs(project_root, exist_ok=True)

        # Store configuration
        self.projects[config.project_name] = asdict(config)
        self._save_projects()

        print(f"Project '{config.project_name}' created successfully")
        return True

    def index_project(self, project_name: str, force_reindex: bool = False) -> Dict[str, Any]:
        """Index a project's code"""

        if project_name not in self.projects:
            print(f"Project '{project_name}' not found")
            return {}

        config_dict = self.projects[project_name]
        project_root = config_dict['project_root']

        print(f"\nIndexing project: {project_name}")
        print(f"Project root: {project_root}")

        # Create indexer with custom file types
        indexer = CodeIndexer(project_root)

        # Index project
        chunks, stats = indexer.index_project(force_reindex=force_reindex)

        # Update project metadata
        config_dict['last_updated'] = datetime.now().isoformat()
        config_dict['total_files_indexed'] = stats['total_files']
        config_dict['total_chunks'] = stats['total_chunks']
        self._save_projects()

        return {
            "project": project_name,
            "chunks": chunks,
            "stats": stats
        }

    def build_dataset(self, project_name: str, force_new_version: bool = False) -> Dict[str, Any]:
        """Build training dataset for a project"""

        if project_name not in self.projects:
            print(f"Project '{project_name}' not found")
            return {}

        # First, index the project
        index_result = self.index_project(project_name, force_reindex=force_new_version)
        if not index_result:
            return {}

        chunks = index_result['chunks']

        # Create dataset builder
        dataset_dir = os.path.join(self.projects_dir, project_name, "datasets")
        builder = IncrementalDatasetBuilder(dataset_dir)

        # Build dataset
        version_key, stats = builder.build_dataset(chunks, project_name, force_new_version)

        return {
            "project": project_name,
            "version_key": version_key,
            "stats": stats
        }

    def get_project_info(self, project_name: str) -> Dict[str, Any]:
        """Get detailed information about a project"""

        if project_name not in self.projects:
            return {}

        config = self.projects[project_name]
        project_dir = os.path.join(self.projects_dir, project_name)

        # Get dataset info
        dataset_dir = os.path.join(project_dir, "datasets")
        datasets = []
        if os.path.exists(dataset_dir):
            versions_dir = os.path.join(dataset_dir, "versions")
            if os.path.exists(versions_dir):
                datasets = [f for f in os.listdir(versions_dir) if f.endswith('.jsonl')]

        return {
            "config": config,
            "datasets": datasets,
            "project_dir": project_dir
        }

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects"""
        return list(self.projects.values())

    def get_handler_for_file(self, file_path: str) -> Optional[FileTypeHandler]:
        """Get appropriate handler for a file"""
        for handler in self.file_handlers.values():
            if handler.can_handle(file_path):
                return handler
        return None

    def analyze_project_structure(self, project_name: str) -> Dict[str, Any]:
        """Analyze the structure of a project"""

        if project_name not in self.projects:
            return {}

        project_root = self.projects[project_name]['project_root']
        analysis = {
            "project": project_name,
            "file_types": {},
            "handlers_available": list(self.file_handlers.keys()),
            "total_files_by_type": {}
        }

        # Scan project
        for root, dirs, files in os.walk(project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'dist', 'build', 'target']]

            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file_path)

                # Count files by type
                analysis['total_files_by_type'][ext] = analysis['total_files_by_type'].get(ext, 0) + 1

                # Check if we have a handler
                handler = self.get_handler_for_file(file_path)
                if handler:
                    handler_name = handler.__class__.__name__
                    if handler_name not in analysis['file_types']:
                        analysis['file_types'][handler_name] = {
                            "extension": ext,
                            "count": 0,
                            "examples": []
                        }
                    analysis['file_types'][handler_name]['count'] += 1
                    if len(analysis['file_types'][handler_name]['examples']) < 3:
                        analysis['file_types'][handler_name]['examples'].append(
                            os.path.relpath(file_path, project_root)
                        )

        return analysis

    def _load_projects(self) -> Dict[str, Dict[str, Any]]:
        """Load projects configuration"""
        if os.path.exists(self.projects_config_file):
            try:
                with open(self.projects_config_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_projects(self):
        """Save projects configuration"""
        try:
            with open(self.projects_config_file, 'w') as f:
                json.dump(self.projects, f, indent=2)
        except Exception as e:
            print(f"Error saving projects: {e}")


class ChangeDetector:
    """Detects changes in project files for incremental updates"""

    def __init__(self, project_manager: ProjectManager):
        self.project_manager = project_manager

    def detect_changes(self, project_name: str) -> Dict[str, List[str]]:
        """Detect file changes since last indexing"""

        if project_name not in self.project_manager.projects:
            return {}

        config = self.project_manager.projects[project_name]
        project_root = config['project_root']

        changes = {
            "added": [],
            "modified": [],
            "deleted": []
        }

        # This is a simplified implementation
        # In production, use git diff or file hashing

        return changes

    def trigger_incremental_update(self, project_name: str) -> bool:
        """Trigger incremental update for a project"""

        changes = self.detect_changes(project_name)

        if not changes or not any(changes.values()):
            print(f"No changes detected in project '{project_name}'")
            return False

        print(f"Changes detected in project '{project_name}':")
        print(f"  Added: {len(changes['added'])}")
        print(f"  Modified: {len(changes['modified'])}")
        print(f"  Deleted: {len(changes['deleted'])}")

        # Build new dataset version with changes
        return self.project_manager.build_dataset(project_name, force_new_version=True)


if __name__ == "__main__":
    # Example usage
    manager = ProjectManager("C:/Sushant/AI Learning/Projects")

    # Create a project
    config = ProjectConfig(
        project_name="my_microservices",
        project_root="C:/Sushant/testWorkspace/microservices/testProj",
        project_type="microservices",
        file_types=['.java'],
        description="Microservices project for code understanding"
    )

    manager.create_project(config)


