"""
Code Indexer Module
Intelligently extracts code chunks from Java and other file types with metadata.
Supports method-level, class-level, and file-level extraction.
"""

import os
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class CodeChunk:
    """Represents a code chunk with metadata"""
    chunk_id: str  # unique identifier
    file_path: str
    file_type: str
    chunk_type: str  # "file", "class", "method", "code_block"
    chunk_name: str  # class/method name or "file_content"
    start_line: int
    end_line: int
    code_content: str
    description: str  # inferred or extracted from comments/docstring
    context: str  # surrounding context for better understanding
    checksum: str  # for change detection
    extracted_at: str
    parent_class: str = None  # for methods, which class they belong to
    parent_method: str = None  # for nested blocks


class JavaCodeExtractor:
    """Extracts code chunks from Java files"""

    def __init__(self):
        self.method_pattern = re.compile(
            r'(public|private|protected)?\s+(static)?\s+(\w+[\[\]]*)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\{',
            re.MULTILINE
        )
        self.class_pattern = re.compile(
            r'(public)?\s+(class|interface)\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*\{',
            re.MULTILINE
        )
        self.javadoc_pattern = re.compile(r'/\*\*.*?\*/', re.DOTALL)
        self.single_comment_pattern = re.compile(r'//.*')

    def extract_chunks(self, file_path: str) -> List[CodeChunk]:
        """Extract all code chunks from a Java file"""
        chunks = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return chunks

        # Extract file-level chunk
        file_chunk = CodeChunk(
            chunk_id=self._generate_chunk_id(file_path, "file", 0),
            file_path=file_path,
            file_type="java",
            chunk_type="file",
            chunk_name="file_content",
            start_line=1,
            end_line=len(lines),
            code_content=content,
            description=self._infer_file_description(content),
            context="",
            checksum=self._compute_checksum(content),
            extracted_at=datetime.now().isoformat()
        )
        chunks.append(file_chunk)

        # Extract classes
        class_matches = list(self.class_pattern.finditer(content))
        for i, match in enumerate(class_matches):
            start_pos = match.start()
            line_num = content[:start_pos].count('\n') + 1

            # Find closing brace for class
            class_body_start = match.end()
            brace_count = 1
            pos = class_body_start
            while pos < len(content) and brace_count > 0:
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                pos += 1

            class_content = content[match.start():pos]
            end_line = content[:pos].count('\n') + 1
            class_name = match.group(3)

            class_chunk = CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, "class", line_num),
                file_path=file_path,
                file_type="java",
                chunk_type="class",
                chunk_name=class_name,
                start_line=line_num,
                end_line=end_line,
                code_content=class_content,
                description=self._extract_javadoc(class_content) or self._infer_class_description(class_name, class_content),
                context="",
                checksum=self._compute_checksum(class_content),
                extracted_at=datetime.now().isoformat(),
                parent_class=class_name
            )
            chunks.append(class_chunk)

            # Extract methods within this class
            method_chunks = self._extract_methods_from_class(
                class_content, file_path, class_name, line_num
            )
            chunks.extend(method_chunks)

        return chunks

    def _extract_methods_from_class(self, class_content: str, file_path: str,
                                    class_name: str, class_start_line: int) -> List[CodeChunk]:
        """Extract methods from class content"""
        methods = []
        matches = list(self.method_pattern.finditer(class_content))

        for match in matches:
            start_pos = match.start()
            line_num = class_content[:start_pos].count('\n') + class_start_line

            # Find closing brace for method
            method_body_start = match.end()
            brace_count = 1
            pos = method_body_start
            while pos < len(class_content) and brace_count > 0:
                if class_content[pos] == '{':
                    brace_count += 1
                elif class_content[pos] == '}':
                    brace_count -= 1
                pos += 1

            method_content = class_content[match.start():pos]
            end_line = class_content[:pos].count('\n') + class_start_line
            method_name = match.group(4)
            return_type = match.group(3)

            method_chunk = CodeChunk(
                chunk_id=self._generate_chunk_id(file_path, "method", line_num),
                file_path=file_path,
                file_type="java",
                chunk_type="method",
                chunk_name=method_name,
                start_line=line_num,
                end_line=end_line,
                code_content=method_content,
                description=self._extract_javadoc(method_content) or
                            self._infer_method_description(method_name, return_type, method_content),
                context=class_name,
                checksum=self._compute_checksum(method_content),
                extracted_at=datetime.now().isoformat(),
                parent_class=class_name
            )
            methods.append(method_chunk)

        return methods

    def _extract_javadoc(self, content: str) -> str:
        """Extract Javadoc comment from code"""
        match = self.javadoc_pattern.search(content)
        if match:
            javadoc = match.group(0)
            # Clean up javadoc markers
            javadoc = re.sub(r'/?\*+/?', '', javadoc).strip()
            return javadoc
        return ""

    def _infer_file_description(self, content: str) -> str:
        """Infer file purpose from content"""
        if 'Controller' in content:
            return "REST API Controller handling HTTP requests"
        elif 'Service' in content:
            return "Business logic service layer"
        elif 'Repository' in content:
            return "Data access layer with database operations"
        elif 'Config' in content:
            return "Configuration class"
        else:
            return "Java source file"

    def _infer_class_description(self, class_name: str, content: str) -> str:
        """Infer class purpose from name and content"""
        if 'Controller' in class_name:
            return f"REST API Controller: {class_name} handles HTTP requests and routing"
        elif 'Service' in class_name:
            return f"Business Service: {class_name} implements core business logic"
        elif 'Repository' in class_name or 'Dao' in class_name:
            return f"Data Access: {class_name} handles database operations"
        elif 'Config' in class_name:
            return f"Configuration: {class_name} defines application settings"
        elif 'Model' in class_name or 'Entity' in class_name or 'DTO' in class_name:
            return f"Data Model: {class_name} represents domain object"
        else:
            return f"Class: {class_name}"

    def _infer_method_description(self, method_name: str, return_type: str, content: str) -> str:
        """Infer method purpose from name and signature"""
        if method_name.startswith('get'):
            return f"Getter method for retrieving data"
        elif method_name.startswith('set'):
            return f"Setter method for updating data"
        elif method_name.startswith('create') or method_name.startswith('add'):
            return f"Creates or adds new data"
        elif method_name.startswith('update'):
            return f"Updates existing data"
        elif method_name.startswith('delete') or method_name.startswith('remove'):
            return f"Deletes data"
        elif method_name.startswith('find') or method_name.startswith('search'):
            return f"Queries and retrieves data"
        else:
            return f"Method: {method_name}() returns {return_type}"

    @staticmethod
    def _generate_chunk_id(file_path: str, chunk_type: str, line_num: int) -> str:
        """Generate unique chunk ID"""
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:6]
        return f"{file_hash}_{chunk_type}_{line_num}"

    @staticmethod
    def _compute_checksum(content: str) -> str:
        """Compute checksum for change detection"""
        return hashlib.sha256(content.encode()).hexdigest()


class MultiFormatExtractor:
    """Handles extraction from multiple file types"""

    EXTRACTORS = {
        '.java': JavaCodeExtractor,
        # Extensions for future file types
        # '.py': PythonCodeExtractor,
        # '.js': JavaScriptExtractor,
    }

    def __init__(self):
        self.extractors = {ext: cls() for ext, cls in self.EXTRACTORS.items()}

    def extract_from_file(self, file_path: str) -> List[CodeChunk]:
        """Extract chunks from file based on extension"""
        _, ext = os.path.splitext(file_path)

        if ext in self.extractors:
            return self.extractors[ext].extract_chunks(file_path)
        else:
            # Return file as-is for unsupported types
            return self._extract_generic_file(file_path)

    def _extract_generic_file(self, file_path: str) -> List[CodeChunk]:
        """Extract generic file as single chunk"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []

        chunk = CodeChunk(
            chunk_id=hashlib.md5(file_path.encode()).hexdigest()[:12],
            file_path=file_path,
            file_type=os.path.splitext(file_path)[1].lstrip('.'),
            chunk_type="file",
            chunk_name=os.path.basename(file_path),
            start_line=1,
            end_line=len(content.split('\n')),
            code_content=content,
            description=f"File: {os.path.basename(file_path)}",
            context="",
            checksum=JavaCodeExtractor._compute_checksum(content),
            extracted_at=datetime.now().isoformat()
        )
        return [chunk]


class CodeIndexer:
    """Main indexer for scanning projects and building code index"""

    def __init__(self, project_root: str, output_dir: str = None):
        self.project_root = project_root
        self.output_dir = output_dir or os.path.join(project_root, ".code_index")
        self.extractor = MultiFormatExtractor()
        self.file_types = ['.java', '.py', '.xml', '.yaml', '.properties', '.sql', '.md']
        self.metadata_file = os.path.join(self.output_dir, "metadata.json")
        self.chunks_file = os.path.join(self.output_dir, "chunks.jsonl")

        os.makedirs(self.output_dir, exist_ok=True)
        self.metadata = self._load_metadata()

    def index_project(self, force_reindex: bool = False) -> Tuple[List[CodeChunk], Dict[str, Any]]:
        """Index entire project, optionally forcing full reindex"""
        chunks = []
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "new_files": 0,
            "modified_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "chunks_by_type": {}
        }

        # Scan all files
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden and irrelevant directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'dist', 'build', 'target']]

            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)

                if ext not in self.file_types:
                    continue

                stats["total_files"] += 1
                relative_path = os.path.relpath(file_path, self.project_root)

                # Check if file changed
                file_hash = self._compute_file_hash(file_path)

                if force_reindex or relative_path not in self.metadata:
                    stats["new_files"] += 1
                elif self.metadata[relative_path]['checksum'] != file_hash:
                    stats["modified_files"] += 1
                else:
                    # File unchanged, skip
                    continue

                stats["processed_files"] += 1

                # Extract chunks
                file_chunks = self.extractor.extract_from_file(file_path)
                for chunk in file_chunks:
                    chunk.file_path = os.path.relpath(file_path, self.project_root)
                    chunks.append(chunk)
                    stats["chunks_by_type"][chunk.chunk_type] = \
                        stats["chunks_by_type"].get(chunk.chunk_type, 0) + 1

                # Update metadata
                self.metadata[relative_path] = {
                    "checksum": file_hash,
                    "last_indexed": datetime.now().isoformat(),
                    "chunks_count": len(file_chunks)
                }

        # Detect deleted files
        indexed_files = set(self.metadata.keys())
        existing_files = set()
        for root, dirs, files in os.walk(self.project_root):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'dist', 'build', 'target']]
            for file in files:
                file_path = os.path.join(root, file)
                _, ext = os.path.splitext(file)
                if ext in self.file_types:
                    existing_files.add(os.path.relpath(file_path, self.project_root))

        deleted = indexed_files - existing_files
        stats["deleted_files"] = len(deleted)
        for file in deleted:
            del self.metadata[file]

        stats["total_chunks"] = len(chunks)

        # Save results
        self._save_chunks(chunks)
        self._save_metadata()

        return chunks, stats

    def incremental_index(self) -> Tuple[List[CodeChunk], Dict[str, Any]]:
        """Perform incremental indexing (default behavior)"""
        return self.index_project(force_reindex=False)

    def get_chunks_for_project(self) -> List[CodeChunk]:
        """Load all indexed chunks"""
        chunks = []
        try:
            if os.path.exists(self.chunks_file):
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        chunk_dict = json.loads(line)
                        chunks.append(CodeChunk(**chunk_dict))
        except Exception as e:
            print(f"Error loading chunks: {e}")
        return chunks

    @staticmethod
    def _compute_file_hash(file_path: str) -> str:
        """Compute hash of file for change detection"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except:
            return ""

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from disk"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_metadata(self):
        """Save metadata to disk"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")

    def _save_chunks(self, chunks: List[CodeChunk]):
        """Save chunks to JSONL file"""
        try:
            with open(self.chunks_file, 'w', encoding='utf-8') as f:
                for chunk in chunks:
                    f.write(json.dumps(asdict(chunk)) + '\n')
        except Exception as e:
            print(f"Error saving chunks: {e}")


if __name__ == "__main__":
    # Test indexer
    indexer = CodeIndexer("C:/Sushant/testWorkspace/microservices/testProj")
    chunks, stats = indexer.incremental_index()

    print(f"\nIndexing Statistics:")
    print(f"Total files: {stats['total_files']}")
    print(f"Processed files: {stats['processed_files']}")
    print(f"New files: {stats['new_files']}")
    print(f"Modified files: {stats['modified_files']}")
    print(f"Deleted files: {stats['deleted_files']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Chunks by type: {stats['chunks_by_type']}")

    print(f"\nFirst 3 chunks:")
    for chunk in chunks[:3]:
        print(f"  - {chunk.chunk_type}: {chunk.chunk_name} ({chunk.file_path}:{chunk.start_line}-{chunk.end_line})")

s fi