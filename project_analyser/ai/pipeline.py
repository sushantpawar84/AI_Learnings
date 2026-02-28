"""
End-to-End AI Code Understanding Pipeline
Orchestrates the complete workflow from code ingestion to model inference.
Handles project setup, training, and continuous learning.
"""

import os
import json
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

# Import all components (updated package path)
from project_analyser.ai.dataset.codeIndexer import CodeIndexer, CodeChunk
from project_analyser.ai.dataset.datasetManager import IncrementalDatasetBuilder, TrainingPair
from project_analyser.ai.dataset.codeTokenizer import CodeDatasetTokenizer, TokenizationPipeline
from project_analyser.ai.llm.incrementalTraining import CodeModelTrainer, TrainingConfig, TrainingOrchestrator
from project_analyser.ai.llm.codeInference import CodeExplainerAPI, QueryResponse
from project_analyser.ai.projectManagement import ProjectManager, ProjectConfig, ChangeDetector


class AICodeUnderstandingPipeline:
    """Main pipeline orchestrating the entire workflow"""

    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.project_manager = ProjectManager(workspace_root)
        self.change_detector = ChangeDetector(self.project_manager)
        self.tokenizer = CodeDatasetTokenizer()
        self.training_config = TrainingConfig()
        self.orchestrator = TrainingOrchestrator(self.training_config)
        self.inference_api = None

        # Create necessary directories
        self.artifacts_dir = os.path.join(workspace_root, "artifacts")
        self.logs_dir = os.path.join(self.artifacts_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

    # ===== PHASE 1: PROJECT SETUP =====

    def setup_project(self, project_name: str, project_root: str,
                     project_type: str = "microservices",
                     file_types: List[str] = None) -> bool:
        """Setup a new project for training"""

        if file_types is None:
            file_types = ['.java'] if project_type == "microservices" else ['.md', '.yaml', '.json']

        config = ProjectConfig(
            project_name=project_name,
            project_root=project_root,
            project_type=project_type,
            file_types=file_types,
            description=f"{project_type.capitalize()} project for code understanding"
        )

        return self.project_manager.create_project(config)

    # ===== PHASE 2: CODE INDEXING =====

    def index_project(self, project_name: str, force_reindex: bool = False) -> Tuple[List[CodeChunk], Dict[str, Any]]:
        """Index project code"""

        print("\n" + "="*80)
        print(f"PHASE 1: CODE INDEXING - Project: {project_name}")
        print("="*80)

        result = self.project_manager.index_project(project_name, force_reindex)

        if result:
            chunks = result['chunks']
            stats = result['stats']

            self._log_event({
                "phase": "indexing",
                "project": project_name,
                "status": "success",
                "total_chunks": len(chunks),
                "stats": stats
            })

            print(f"\nIndexing complete!")
            print(f"  Total files: {stats['total_files']}")
            print(f"  Processed files: {stats['processed_files']}")
            print(f"  Total chunks: {stats['total_chunks']}")
            print(f"  Chunks by type: {stats['chunks_by_type']}")

            return chunks, stats

        return [], {}

    # ===== PHASE 3: DATASET CREATION =====

    def create_dataset(self, project_name: str, chunks: List[CodeChunk] = None) -> Tuple[str, Dict[str, Any]]:
        """Create training dataset from indexed code"""

        print("\n" + "="*80)
        print(f"PHASE 2: DATASET CREATION - Project: {project_name}")
        print("="*80)

        # Index if chunks not provided
        if chunks is None:
            chunks, _ = self.index_project(project_name)

        # Create dataset
        project_info = self.project_manager.get_project_info(project_name)
        dataset_dir = os.path.join(project_info['project_dir'], "datasets")

        builder = IncrementalDatasetBuilder(dataset_dir)
        version_key, stats = builder.build_dataset(chunks, project_name)

        self._log_event({
            "phase": "dataset_creation",
            "project": project_name,
            "status": "success",
            "version_key": version_key,
            "total_pairs": stats['total_pairs'],
            "pairs_by_type": stats['pairs_by_type']
        })

        print(f"\nDataset creation complete!")
        print(f"  Version: {version_key}")
        print(f"  Total training pairs: {stats['total_pairs']}")
        print(f"  Pairs by type: {stats['pairs_by_type']}")

        return version_key, stats

    # ===== PHASE 4: TOKENIZATION =====

    def tokenize_dataset(self, project_name: str, version_key: str) -> Dict[str, Any]:
        """Tokenize training dataset"""

        print("\n" + "="*80)
        print(f"PHASE 3: TOKENIZATION - Project: {project_name}")
        print("="*80)

        # Get dataset file path
        project_info = self.project_manager.get_project_info(project_name)
        dataset_dir = os.path.join(project_info['project_dir'], "datasets")
        input_file = os.path.join(dataset_dir, "versions", f"{version_key}.jsonl")
        output_file = os.path.join(dataset_dir, f"{version_key}_tokenized.jsonl")

        # Tokenize
        pipeline = TokenizationPipeline()
        stats = pipeline.process_dataset(input_file, output_file)

        self._log_event({
            "phase": "tokenization",
            "project": project_name,
            "status": "success",
            "tokenized_file": output_file,
            "stats": stats
        })

        return {**stats, "tokenized_file": output_file}

    # ===== PHASE 5: MODEL TRAINING =====

    def train_model(self, project_name: str, tokenized_file: str,
                   eval_file: str = None) -> Dict[str, Any]:
        """Train or fine-tune model on code dataset"""

        print("\n" + "="*80)
        print(f"PHASE 4: MODEL TRAINING - Project: {project_name}")
        print("="*80)

        # Train
        metrics = self.orchestrator.train_on_new_project(tokenized_file, eval_file)

        self._log_event({
            "phase": "training",
            "project": project_name,
            "status": "success",
            "metrics": metrics
        })

        print(f"\nTraining complete!")
        print(f"  Model version: {metrics.get('version', 'N/A')}")
        print(f"  Train loss: {metrics.get('train_loss', 'N/A')}")
        print(f"  Checkpoint: {metrics.get('checkpoint', 'N/A')}")

        return metrics

    # ===== PHASE 6: MODEL INFERENCE =====

    def setup_inference(self, checkpoint_path: str, chunks_file: str):
        """Setup inference engine"""

        print("\n" + "="*80)
        print("PHASE 5: INFERENCE SETUP")
        print("="*80)

        self.inference_api = CodeExplainerAPI(
            model_path=checkpoint_path,
            tokenizer_path=checkpoint_path,
            chunks_file=chunks_file
        )

        print(f"Inference engine ready!")
        return self.inference_api

    def query_code(self, question: str) -> QueryResponse:
        """Query the model about code"""

        if self.inference_api is None:
            print("Error: Inference engine not initialized. Call setup_inference first.")
            return None

        response = self.inference_api.explain(question)
        return response

    # ===== PHASE 7: CONTINUOUS LEARNING =====

    def detect_and_update(self, project_name: str) -> bool:
        """Detect code changes and trigger incremental training"""

        print("\n" + "="*80)
        print(f"PHASE 6: CONTINUOUS LEARNING - Project: {project_name}")
        print("="*80)

        # Detect changes
        changes = self.change_detector.detect_changes(project_name)

        has_changes = any(changes.values()) if changes else False

        if not has_changes:
            print(f"No changes detected in {project_name}")
            self._log_event({
                "phase": "continuous_learning",
                "project": project_name,
                "status": "no_changes"
            })
            return False

        print(f"Changes detected!")
        print(f"  Added: {len(changes.get('added', []))}")
        print(f"  Modified: {len(changes.get('modified', []))}")
        print(f"  Deleted: {len(changes.get('deleted', []))}")

        # Perform incremental update
        print(f"\nPerforming incremental update...")

        # 1. Re-index
        chunks, _ = self.index_project(project_name, force_reindex=False)

        # 2. Create new dataset version
        version_key, dataset_stats = self.create_dataset(project_name, chunks)

        # 3. Tokenize
        tokenization_stats = self.tokenize_dataset(project_name, version_key)

        # 4. Incremental training
        tokenized_file = tokenization_stats['tokenized_file']
        training_metrics = self.orchestrator.incremental_retrain(tokenized_file)

        self._log_event({
            "phase": "continuous_learning",
            "project": project_name,
            "status": "success",
            "changes": changes,
            "new_version": version_key,
            "training_metrics": training_metrics
        })

        print(f"Incremental update complete!")
        return True

    # ===== UTILITY METHODS =====

    def analyze_project(self, project_name: str) -> Dict[str, Any]:
        """Analyze project structure and readiness"""

        print("\n" + "="*80)
        print(f"PROJECT ANALYSIS - {project_name}")
        print("="*80)

        analysis = self.project_manager.analyze_project_structure(project_name)

        print(f"\nProject Structure Analysis:")
        print(f"  File types available: {analysis.get('handlers_available', [])}")
        print(f"  Total files by type: {analysis.get('total_files_by_type', {})}")

        if analysis.get('file_types'):
            print(f"\n  Recognized file types:")
            for handler_name, info in analysis['file_types'].items():
                print(f"    - {handler_name}: {info['count']} files")

        return analysis

    def run_full_pipeline(self, project_name: str, project_root: str) -> Dict[str, Any]:
        """Execute complete pipeline from code to trained model"""

        print("\n" + "#"*80)
        print("# AI CODE UNDERSTANDING PIPELINE - END TO END")
        print("#"*80)

        start_time = datetime.now()

        # Step 1: Setup project
        print(f"\n[1/7] Setting up project...")
        if not self.setup_project(project_name, project_root):
            print(f"Project setup failed!")
            return {}

        # Step 2: Index code
        print(f"\n[2/7] Indexing code...")
        chunks, index_stats = self.index_project(project_name)
        if not chunks:
            print(f"Code indexing failed!")
            return {}

        # Step 3: Create dataset
        print(f"\n[3/7] Creating dataset...")
        version_key, dataset_stats = self.create_dataset(project_name, chunks)

        # Step 4: Tokenize
        print(f"\n[4/7] Tokenizing dataset...")
        tokenization_stats = self.tokenize_dataset(project_name, version_key)

        # Step 5: Train model
        print(f"\n[5/7] Training model...")
        training_metrics = self.train_model(
            project_name,
            tokenization_stats['tokenized_file']
        )

        # Step 6: Setup inference
        print(f"\n[6/7] Setting up inference...")
        checkpoint_path = training_metrics.get('checkpoint', '')
        chunks_file = os.path.join(
            self.project_manager.projects_dir,
            project_name,
            ".code_index",
            "chunks.jsonl"
        )

        if os.path.exists(checkpoint_path):
            self.setup_inference(checkpoint_path, chunks_file)

        # Step 7: Ready for queries
        print(f"\n[7/7] Pipeline ready for inference!")

        execution_time = (datetime.now() - start_time).total_seconds()

        summary = {
            "status": "success",
            "execution_time_seconds": execution_time,
            "index_stats": index_stats,
            "dataset_stats": dataset_stats,
            "tokenization_stats": tokenization_stats,
            "training_metrics": training_metrics
        }

        self._log_event({
            "phase": "pipeline_complete",
            "project": project_name,
            "status": "success",
            "execution_time_seconds": execution_time,
            "summary": summary
        })

        print("\n" + "#"*80)
        print(f"# PIPELINE COMPLETE in {execution_time:.2f} seconds")
        print("#"*80)

        return summary

    def _log_event(self, event: Dict[str, Any]):
        """Log pipeline event"""
        log_file = os.path.join(self.logs_dir, "pipeline.jsonl")
        try:
            event['timestamp'] = datetime.now().isoformat()
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            print(f"Error logging event: {e}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "projects": self.project_manager.list_projects(),
            "inference_ready": self.inference_api is not None,
            "workspace_root": self.workspace_root,
            "artifacts_dir": self.artifacts_dir
        }


# ===== EXAMPLE USAGE =====

if __name__ == "__main__":

    # Initialize pipeline
    pipeline = AICodeUnderstandingPipeline(
        "C:/Sushant/AI Learning/Projects"
    )

    # Run complete pipeline
    # summary = pipeline.run_full_pipeline(


