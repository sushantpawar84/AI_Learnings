"""
Incremental Dataset Manager
Handles versioned dataset creation with support for continuous updates.
Creates training pairs from code chunks and manages dataset versions.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from project_analyser.ai.dataset.codeIndexer import CodeChunk


@dataclass
class TrainingPair:
    """Represents a training pair for the model"""
    pair_id: str
    input_text: str  # code snippet or query
    output_text: str  # explanation or expected answer
    chunk_id: str  # reference to source chunk
    pair_type: str  # "explain_method", "explain_class", "explain_file", "explain_logic", "qa"
    tags: List[str]  # for filtering/querying
    created_at: str
    version: int  # dataset version


class TrainingPairGenerator:
    """Generates training pairs from code chunks"""

    def __init__(self):
        self.pair_counter = 0

    def generate_pairs(self, chunks: List[CodeChunk]) -> List[TrainingPair]:
        """Generate training pairs from code chunks"""
        pairs = []

        for chunk in chunks:
            if chunk.chunk_type == "method":
                pairs.extend(self._generate_method_pairs(chunk))
            elif chunk.chunk_type == "class":
                pairs.extend(self._generate_class_pairs(chunk))
            elif chunk.chunk_type == "file":
                pairs.extend(self._generate_file_pairs(chunk))

        return pairs

    def _generate_method_pairs(self, chunk: CodeChunk) -> List[TrainingPair]:
        """Generate training pairs for a method"""
        pairs = []

        # Pair 1: What does this method do?
        pair1 = TrainingPair(
            pair_id=self._generate_pair_id(),
            input_text=f"Explain this Java method:\n\n{chunk.code_content}",
            output_text=chunk.description or self._infer_description(chunk),
            chunk_id=chunk.chunk_id,
            pair_type="explain_method",
            tags=[chunk.chunk_type, "method_explanation", chunk.parent_class or ""],
            created_at=datetime.now().isoformat(),
            version=1
        )
        pairs.append(pair1)

        # Pair 2: Direct question about method
        pair2 = TrainingPair(
            pair_id=self._generate_pair_id(),
            input_text=f"What does the method '{chunk.chunk_name}' do in class '{chunk.parent_class}'?",
            output_text=chunk.description or self._infer_description(chunk),
            chunk_id=chunk.chunk_id,
            pair_type="qa",
            tags=[chunk.chunk_type, "method_qa", chunk.parent_class or ""],
            created_at=datetime.now().isoformat(),
            version=1
        )
        pairs.append(pair2)

        # Pair 3: How to use this method
        pair3 = TrainingPair(
            pair_id=self._generate_pair_id(),
            input_text=f"How to use the {chunk.chunk_name}() method?\n\nMethod code:\n{chunk.code_content}",
            output_text=self._generate_usage_explanation(chunk),
            chunk_id=chunk.chunk_id,
            pair_type="explain_logic",
            tags=[chunk.chunk_type, "method_usage", chunk.parent_class or ""],
            created_at=datetime.now().isoformat(),
            version=1
        )
        pairs.append(pair3)

        return pairs

    def _generate_class_pairs(self, chunk: CodeChunk) -> List[TrainingPair]:
        """Generate training pairs for a class"""
        pairs = []

        # Pair 1: What does this class do?
        pair1 = TrainingPair(
            pair_id=self._generate_pair_id(),
            input_text=f"Explain this Java class:\n\n{chunk.code_content[:1000]}...",
            output_text=chunk.description or self._infer_description(chunk),
            chunk_id=chunk.chunk_id,
            pair_type="explain_class",
            tags=[chunk.chunk_type, "class_explanation"],
            created_at=datetime.now().isoformat(),
            version=1
        )
        pairs.append(pair1)

        # Pair 2: Direct question about class
        pair2 = TrainingPair(
            pair_id=self._generate_pair_id(),
            input_text=f"What is the purpose of the '{chunk.chunk_name}' class?",
            output_text=chunk.description or self._infer_description(chunk),
            chunk_id=chunk.chunk_id,
            pair_type="qa",
            tags=[chunk.chunk_type, "class_qa"],
            created_at=datetime.now().isoformat(),
            version=1
        )
        pairs.append(pair2)

        return pairs

    def _generate_file_pairs(self, chunk: CodeChunk) -> List[TrainingPair]:
        """Generate training pairs for a file"""
        pairs = []

        # Pair 1: File overview
        pair1 = TrainingPair(
            pair_id=self._generate_pair_id(),
            input_text=f"Summarize the purpose of this Java file:\n\n{chunk.code_content[:800]}...",
            output_text=chunk.description or self._infer_description(chunk),
            chunk_id=chunk.chunk_id,
            pair_type="explain_file",
            tags=[chunk.chunk_type, "file_explanation"],
            created_at=datetime.now().isoformat(),
            version=1
        )
        pairs.append(pair1)

        return pairs

    def _infer_description(self, chunk: CodeChunk) -> str:
        """Infer description from code if not provided"""
        if chunk.description:
            return chunk.description

        # Fallback descriptions
        if chunk.chunk_type == "method":
            return f"Method '{chunk.chunk_name}' in class {chunk.parent_class}"
        elif chunk.chunk_type == "class":
            return f"Java class '{chunk.chunk_name}'"
        else:
            return f"Code content from {chunk.file_path}"

    def _generate_usage_explanation(self, chunk: CodeChunk) -> str:
        """Generate explanation of how to use a method"""
        if "void" in chunk.code_content and "System.out" in chunk.code_content:
            return f"This is a utility method. Call {chunk.chunk_name}() to execute its operations."
        elif "return" in chunk.code_content:
            return f"This method returns a value. Use the result from {chunk.chunk_name}() for further processing."
        else:
            return f"Call this method as: {chunk.chunk_name}()"

    @staticmethod
    def _generate_pair_id() -> str:
        """Generate unique pair ID"""
        return hashlib.md5(
            f"{datetime.now().isoformat()}{os.urandom(8).hex()}".encode()
        ).hexdigest()[:12]


class DatasetVersionManager:
    """Manages dataset versions and incremental updates"""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.versions_file = os.path.join(dataset_dir, "versions.json")
        self.pairs_dir = os.path.join(dataset_dir, "versions")
        self.metadata_file = os.path.join(dataset_dir, "dataset_metadata.json")

        os.makedirs(self.pairs_dir, exist_ok=True)
        self.versions = self._load_versions()
        self.metadata = self._load_metadata()

    def create_or_update_dataset(self, pairs: List[TrainingPair],
                                  project_name: str,
                                  force_new_version: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Create or update dataset version"""

        current_version = self.metadata.get(project_name, {}).get("current_version", 0)
        new_version = current_version + 1 if force_new_version or not self.versions.get(project_name) else current_version

        version_key = f"{project_name}_v{new_version}"
        version_file = os.path.join(self.pairs_dir, f"{version_key}.jsonl")

        # Save pairs
        try:
            with open(version_file, 'w', encoding='utf-8') as f:
                for pair in pairs:
                    pair.version = new_version
                    f.write(json.dumps(asdict(pair)) + '\n')
        except Exception as e:
            print(f"Error saving dataset version: {e}")
            return "", {}

        # Update metadata
        stats = {
            "version": new_version,
            "total_pairs": len(pairs),
            "pairs_by_type": {},
            "created_at": datetime.now().isoformat(),
            "file_path": version_file
        }

        for pair in pairs:
            pair_type = pair.pair_type
            stats["pairs_by_type"][pair_type] = stats["pairs_by_type"].get(pair_type, 0) + 1

        self.versions[version_key] = stats

        # Update project metadata
        if project_name not in self.metadata:
            self.metadata[project_name] = {}

        self.metadata[project_name].update({
            "current_version": new_version,
            "total_pairs": len(pairs),
            "last_updated": datetime.now().isoformat(),
            "pairs_by_type": stats["pairs_by_type"]
        })

        self._save_versions()
        self._save_metadata()

        return version_key, stats

    def get_dataset_for_training(self, project_name: str, version: int = None) -> Tuple[List[TrainingPair], str]:
        """Get dataset for training"""
        if version is None:
            version = self.metadata.get(project_name, {}).get("current_version", 1)

        version_key = f"{project_name}_v{version}"
        version_file = os.path.join(self.pairs_dir, f"{version_key}.jsonl")

        pairs = []
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        pair_dict = json.loads(line)
                        pairs.append(TrainingPair(**pair_dict))
            except Exception as e:
                print(f"Error loading dataset: {e}")

        return pairs, version_key

    def get_incremental_updates(self, project_name: str,
                               since_version: int) -> Tuple[List[TrainingPair], Dict[str, Any]]:
        """Get training pairs added since a specific version"""
        current_version = self.metadata.get(project_name, {}).get("current_version", 1)
        new_pairs = []
        stats = {
            "versions_included": list(range(since_version + 1, current_version + 1)),
            "total_new_pairs": 0,
            "pairs_by_type": {}
        }

        for version in range(since_version + 1, current_version + 1):
            version_key = f"{project_name}_v{version}"
            version_file = os.path.join(self.pairs_dir, f"{version_key}.jsonl")

            if os.path.exists(version_file):
                try:
                    with open(version_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            pair_dict = json.loads(line)
                            pair = TrainingPair(**pair_dict)
                            new_pairs.append(pair)
                            pair_type = pair.pair_type
                            stats["pairs_by_type"][pair_type] = stats["pairs_by_type"].get(pair_type, 0) + 1
                except Exception as e:
                    print(f"Error loading version {version_key}: {e}")

        stats["total_new_pairs"] = len(new_pairs)
        return new_pairs, stats

    def get_version_history(self, project_name: str) -> List[Dict[str, Any]]:
        """Get version history for a project"""
        history = []
        project_versions = [k for k in self.versions.keys() if k.startswith(f"{project_name}_v")]
        project_versions.sort(key=lambda x: int(x.split('_v')[1]))

        for version_key in project_versions:
            history.append(self.versions[version_key])

        return history

    def _load_versions(self) -> Dict[str, Any]:
        """Load versions metadata"""
        if os.path.exists(self.versions_file):
            try:
                with open(self.versions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _load_metadata(self) -> Dict[str, Any]:
        """Load dataset metadata"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_versions(self):
        """Save versions metadata"""
        try:
            with open(self.versions_file, 'w', encoding='utf-8') as f:
                json.dump(self.versions, f, indent=2)
        except Exception as e:
            print(f"Error saving versions: {e}")

    def _save_metadata(self):
        """Save dataset metadata"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")


class IncrementalDatasetBuilder:
    """Orchestrates incremental dataset creation and updates"""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.pair_generator = TrainingPairGenerator()
        self.version_manager = DatasetVersionManager(dataset_dir)
        os.makedirs(dataset_dir, exist_ok=True)

    def build_dataset(self, chunks: List[CodeChunk],
                     project_name: str,
                     force_new_version: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Build training dataset from code chunks"""

        print(f"Generating training pairs from {len(chunks)} chunks...")
        pairs = self.pair_generator.generate_pairs(chunks)
        print(f"Generated {len(pairs)} training pairs")

        print(f"Creating dataset version...")
        version_key, stats = self.version_manager.create_or_update_dataset(
            pairs, project_name, force_new_version
        )

        print(f"Dataset created: {version_key}")
        print(f"  - Total pairs: {stats['total_pairs']}")
        print(f"  - Pairs by type: {stats['pairs_by_type']}")

        return version_key, stats

    def get_training_dataset(self, project_name: str) -> List[TrainingPair]:
        """Get current training dataset"""
        pairs, _ = self.version_manager.get_dataset_for_training(project_name)
        return pairs

    def get_incremental_updates(self, project_name: str, since_version: int) -> Tuple[List[TrainingPair], Dict[str, Any]]:
        """Get new training pairs since a version"""
        return self.version_manager.get_incremental_updates(project_name, since_version)


if __name__ == "__main__":
    # Test dataset builder
    from project_analyser.ai.dataset.codeIndexer import CodeIndexer

    # Index project
    print("Indexing project...")
    indexer = CodeIndexer("C:/Sushant/testWorkspace/microservices/testProj")
    chunks, index_stats = indexer.incremental_index()
    print(f"Indexed {len(chunks)} chunks")

    # Build dataset
    print("\nBuilding dataset...")
    dataset_dir = "C:/Sushant/AI Learning/Datasets/microservices"
    builder = IncrementalDatasetBuilder(dataset_dir)
    version_key, dataset_stats = builder.build_dataset(chunks, "microservices_project")

    print(f"\nDataset statistics:")
    print(f"  Version: {dataset_stats['version']}")
    print(f"  Total pairs: {dataset_stats['total_pairs']}")
    print(f"  Pairs by type: {dataset_stats['pairs_by_type']}")

    # Show version history
    print(f"\nVersion history:")
    history = builder.version_manager.get_version_history("microservices_project")
    for v in history:
        print(f"  - v{v['version']}: {v['total_pairs']} pairs ({v['created_at']})")

