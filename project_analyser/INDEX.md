# 📚 Complete System Index & Navigation Guide

## 🗺️ Where to Start?

### For Quick Setup (5 min)
→ **[QUICKSTART.md](QUICKSTART.md)**
- Install dependencies
- Setup first project
- Run pipeline
- Ask questions

### For Complete Technical Details
→ **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)**
- Detailed module documentation
- Configuration options
- Performance metrics
- Troubleshooting

### For Working Examples
→ **[EXAMPLES.py](EXAMPLES.py)**
- 10 complete runnable examples
- Copy-paste code snippets
- Expected outputs

### For Understanding Architecture
→ **[ARCHITECTURE.md](ARCHITECTURE.md)**
- System design diagrams
- Data flow visualization
- Directory structure
- Component interactions

### For Implementation Overview
→ **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)**
- What was built
- Key innovations
- System capabilities
- Next steps

### For Project Overview
→ **[README.md](README.md)**
- Feature list
- Quick start
- Use cases
- Configuration

---

## 📁 Core Modules Reference

### Phase 1: Code Indexing
**File**: `project_analyser/ai/dataset/codeIndexer.py` (600+ lines)

**Key Classes**:
- `CodeChunk` - Data structure for code snippets with metadata
- `JavaCodeExtractor` - Extracts Java methods, classes, files
- `MultiFormatExtractor` - Extensible framework for multiple file types
- `CodeIndexer` - Main indexer that scans entire projects

**Key Methods**:
```python
indexer = CodeIndexer(project_root)
chunks, stats = indexer.incremental_index()  # Index project
chunks, stats = indexer.index_project(force_reindex=True)  # Force full reindex
chunks = indexer.get_chunks_for_project()  # Load indexed chunks
```

**Use When**: You need to extract code structure from source files

---

### Phase 2: Dataset Creation
**File**: `project_analyser/ai/dataset/datasetManager.py` (400+ lines)

**Key Classes**:
- `TrainingPair` - Represents input-output training example
- `TrainingPairGenerator` - Creates training pairs from code chunks
- `DatasetVersionManager` - Manages dataset versions with history
- `IncrementalDatasetBuilder` - Orchestrates dataset creation

**Key Methods**:
```python
builder = IncrementalDatasetBuilder(dataset_dir)
version_key, stats = builder.build_dataset(chunks, project_name)  # Create dataset
pairs, _ = builder.version_manager.get_dataset_for_training(project_name)  # Load pairs
pairs, stats = builder.get_incremental_updates(project_name, since_version)  # Get new pairs
```

**Use When**: You want to generate training data from indexed code

---

### Phase 3: Tokenization
**File**: `project_analyser/ai/dataset/codeTokenizer.py` (450+ lines)

**Key Classes**:
- `CodeTokenizer` - Tokenizes text with code-specific handling
- `DynamicBatcher` - Creates efficient batches based on token count
- `CodeDatasetTokenizer` - End-to-end tokenization pipeline
- `TokenizationPipeline` - High-level orchestration

**Key Methods**:
```python
pipeline = TokenizationPipeline()
stats = pipeline.process_dataset(input_file, output_file)  # Tokenize dataset
tokenized, stats = pipeline.dataset_tokenizer.tokenize_dataset_file(input_file, output_file)
batches = pipeline.batcher.create_batches(tokenized_pairs)  # Create batches
```

**Use When**: You need to convert training data to token sequences

---

### Phase 4: Model Training
**File**: `project_analyser/ai/llm/incrementalTraining.py` (450+ lines)

**Key Classes**:
- `TrainingConfig` - Configuration dataclass for training
- `ModelCheckpointManager` - Manages model checkpoints and versions
- `CodeModelTrainer` - Fine-tunes LLM with LoRA
- `TrainingOrchestrator` - High-level training interface

**Key Methods**:
```python
trainer = CodeModelTrainer(config)
trainer.prepare_model()  # Load model and apply LoRA
metrics = trainer.train_on_dataset(train_file, eval_file)  # Train
metrics = trainer.incremental_train(new_data_file, checkpoint_version)  # Incremental
response = trainer.generate_from_prompt(prompt)  # Generate text
```

**Use When**: You want to fine-tune the language model

---

### Phase 5: Inference & Querying
**File**: `project_analyser/ai/llm/codeInference.py` (500+ lines)

**Key Classes**:
- `QueryType` - Enum for different question types
- `QueryResponse` - Response with confidence and citations
- `CodeRetriever` - Retrieves relevant code chunks (RAG)
- `CodeModelInference` - Model inference engine
- `QueryCache` - Caching layer for queries
- `CodeExplainerAPI` - High-level user API

**Key Methods**:
```python
api = CodeExplainerAPI(model_path, tokenizer_path, chunks_file)
response = api.explain(question)  # Answer any question
response = api.inference.explain_method(method_name, class_name)
response = api.inference.explain_class(class_name)
response = api.inference.explain_file(file_path)
response = api.inference.explain_code(code_snippet)
api.print_response(response)  # Pretty print response
```

**Use When**: You want to ask questions about code

---

### Phase 6: Project Management
**File**: `project_analyser/ai/projectManagement.py` (600+ lines)

**Key Classes**:
- `ProjectConfig` - Configuration for a project
- `FileTypeHandler` - Abstract base for file type handlers
- `JavaFileHandler`, `MarkdownFileHandler`, `YamlFileHandler` - Specific handlers
- `ProjectManager` - Manages multiple projects
- `ChangeDetector` - Detects code changes

**Key Methods**:
```python
manager = ProjectManager(workspace_root)
manager.create_project(config)  # Create new project
manager.index_project(project_name)  # Index project code
manager.build_dataset(project_name)  # Build training dataset
manager.list_projects()  # List all projects
manager.analyze_project_structure(project_name)  # Analyze structure
detector.detect_changes(project_name)  # Detect code changes
```

**Use When**: You manage multiple projects or need file handlers

---

### Phase 7: Pipeline Orchestration
**File**: `project_analyser/ai/pipeline.py` (500+ lines)

**Key Class**:
- `AICodeUnderstandingPipeline` - Main orchestrator

**Key Methods**:
```python
pipeline = AICodeUnderstandingPipeline(workspace_root)

# Setup
pipeline.setup_project(name, root, type)

# Full pipeline
summary = pipeline.run_full_pipeline(name, root)

# Individual phases
chunks, stats = pipeline.index_project(name)
version_key, stats = pipeline.create_dataset(name, chunks)
tok_stats = pipeline.tokenize_dataset(name, version_key)
train_metrics = pipeline.train_model(name, tok_stats['tokenized_file'])
pipeline.setup_inference(checkpoint_path, chunks_file)
response = pipeline.query_code(question)

# Continuous learning
has_changes = pipeline.detect_and_update(name)

# Status
status = pipeline.get_pipeline_status()
```

**Use When**: You want end-to-end orchestration or single API

---

### CLI Interface
**File**: `project_analyser/ai/cli.py` (450+ lines)

**Key Class**:
- `CodeUnderstandingCLI` - Command-line interface

**Commands**:
```powershell
# Project management
python -m project_analyser.ai.cli setup-project --name NAME --root ROOT
python -m project_analyser.ai.cli list-projects
python -m project_analyser.ai.cli analyze-project --name NAME

# Pipeline
python -m project_analyser.ai.cli run-pipeline --name NAME --root ROOT
python -m project_analyser.ai.cli index --name NAME
python -m project_analyser.ai.cli create-dataset --name NAME
python -m project_analyser.ai.cli tokenize --name NAME --version VERSION
python -m project_analyser.ai.cli train --name NAME --version VERSION

# Inference
python -m project_analyser.ai.cli query --question "Your question"

# System
python -m project_analyser.ai.cli status
python -m project_analyser.ai.cli help
```

**Use When**: You prefer command-line interaction

---

## 🎯 Common Tasks & Solutions

### Task: Index My Java Microservices
```python
from project_analyser.ai.dataset.codeIndexer import CodeIndexer

indexer = CodeIndexer("C:/path/to/microservices")
chunks, stats = indexer.incremental_index()
print(f"Found {len(chunks)} code chunks")
```

### Task: Generate Training Data
```python
from project_analyser.ai.dataset.datasetManager import IncrementalDatasetBuilder

builder = IncrementalDatasetBuilder("./datasets")
version_key, stats = builder.build_dataset(chunks, "my_project")
print(f"Created {stats['total_pairs']} training pairs")
```

### Task: Fine-tune LLM
```python
from project_analyser.ai.llm.incrementalTraining import CodeModelTrainer, TrainingConfig

config = TrainingConfig(num_train_epochs=3)
trainer = CodeModelTrainer(config)
trainer.prepare_model()
metrics = trainer.train_on_dataset("tokenized_data.jsonl")
```

### Task: Ask About Code
```python
from project_analyser.ai.llm.codeInference import CodeExplainerAPI

api = CodeExplainerAPI("./checkpoint", "./checkpoint", "chunks.jsonl")
response = api.explain("What does UserController do?")
print(response.response)
print(f"Sources: {response.source_citations}")
```

### Task: Manage Multiple Projects
```python
from project_analyser.ai.projectManagement import ProjectManager, ProjectConfig

manager = ProjectManager("./workspace")
for project in manager.list_projects():
    print(f"{project['project_name']}: {project['project_type']}")
```

### Task: Auto-Update on Code Changes
```python
from project_analyser.ai.pipeline import AICodeUnderstandingPipeline
import time

pipeline = AICodeUnderstandingPipeline("./workspace")
while True:
    has_changes = pipeline.detect_and_update("my_project")
    if has_changes:
        print("Model updated!")
    time.sleep(3600)  # Check hourly
```

### Task: Full Pipeline in One Call
```python
from project_analyser.ai.pipeline import AICodeUnderstandingPipeline

pipeline = AICodeUnderstandingPipeline("./workspace")
summary = pipeline.run_full_pipeline("my_project", "C:/code")
print(f"Complete in {summary['execution_time_seconds']:.2f}s")
```

---

## 🔍 Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                        CLI Interface                         │
│                  (test/ai/cli.py)                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              End-to-End Pipeline                            │
│         (test/ai/pipeline.py)                              │
│         Coordinates all 7 phases                            │
└──────────┬──────────────┬──────────────┬────────────────────┘
           │              │              │
    ┌──────▼────┐  ┌──────▼──────┐  ┌───▼─────────┐
    │  Indexing  │  │  Dataset    │  │ Training    │
    │  Phase     │  │  Phase      │  │ Phase       │
    │            │  │             │  │             │
    └────┬───────┘  └──────┬──────┘  └───┬─────────┘
         │                 │             │
    ┌────▼─────────────────▼──────┐ ┌───▼─────────┐
    │   codeIndexer.py            │ │ Inference   │
    │   datasetManager.py         │ │ Phase       │
    │   codeTokenizer.py          │ │             │
    └────┬─────────────────────────┘ └───┬─────────┘
         │                               │
         ├───────────────────────────────┤
         │                               │
    ┌────▼──────────────────────────┐   │
    │  projectManagement.py         │   │
    │  (Multi-project support)      │   │
    │                               │   │
    └────┬──────────────────────────┘   │
         │                              │
    ┌────▼──────────────────────────┐   │
    │  incrementalTraining.py       │   │
    │  (LoRA fine-tuning)           │   │
    │                               │   │
    └────┬──────────────────────────┘   │
         │                              │
         └──────────────┬───────────────┘
                        │
                    ┌───▼──────────┐
                    │ codeInference│
                    │ .py          │
                    │ (Query API)  │
                    └──────────────┘
```

---

## 📊 File Statistics

| File | Lines | Classes | Functions |
|------|-------|---------|-----------|
| codeIndexer.py | 600+ | 5 | 25+ |
| datasetManager.py | 400+ | 4 | 20+ |
| codeTokenizer.py | 450+ | 5 | 20+ |
| incrementalTraining.py | 450+ | 4 | 15+ |
| codeInference.py | 500+ | 8 | 30+ |
| projectManagement.py | 600+ | 8 | 25+ |
| pipeline.py | 500+ | 1 | 25+ |
| cli.py | 450+ | 1 | 15+ |
| **Total** | **3950+** | **36** | **175+** |

**Documentation**: 4000+ lines (guides, examples, architecture)

---

## ✨ Key Features by Module

| Feature | Module | Example |
|---------|--------|---------|
| Extract Java code | codeIndexer | `JavaCodeExtractor` |
| Generate training pairs | datasetManager | `TrainingPairGenerator` |
| Dynamic batching | codeTokenizer | `DynamicBatcher` |
| LoRA fine-tuning | incrementalTraining | `CodeModelTrainer` |
| RAG retrieval | codeInference | `CodeRetriever` |
| Multi-project | projectManagement | `ProjectManager` |
| Continuous learning | projectManagement | `ChangeDetector` |
| End-to-end workflow | pipeline | `AICodeUnderstandingPipeline` |
| CLI commands | cli | `CodeUnderstandingCLI` |

---

## 🚀 Recommended Reading Order

### First Time? Start Here:
1. [README.md](README.md) - 5 min
2. [QUICKSTART.md](QUICKSTART.md) - 10 min
3. [EXAMPLES.py](EXAMPLES.py) - 10 min
4. Run a simple example

### Want Details?
5. [ARCHITECTURE.md](ARCHITECTURE.md) - 15 min
6. [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - 30 min
7. Review individual modules

### Advanced Users:
8. [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
9. Read source code in detail
10. Extend with new features

---

## 📞 Quick Help

**Q: How do I get started?**  
A: Read [QUICKSTART.md](QUICKSTART.md) (5 minutes)

**Q: How does the whole system work?**  
A: See [ARCHITECTURE.md](ARCHITECTURE.md) for diagrams and flow

**Q: What are the core modules?**  
A: See index above, each module has its own section

**Q: How do I use a specific feature?**  
A: Check [EXAMPLES.py](EXAMPLES.py) for working code

**Q: What if something goes wrong?**  
A: See troubleshooting in [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)

**Q: Can I extend it?**  
A: Yes! See "Extensibility" section in [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)

---

## 📦 Files Included

**Core Code** (8 modules, 4000+ lines)
- ✅ codeIndexer.py
- ✅ datasetManager.py
- ✅ codeTokenizer.py
- ✅ incrementalTraining.py
- ✅ codeInference.py
- ✅ projectManagement.py
- ✅ pipeline.py
- ✅ cli.py

**Documentation** (4000+ lines)
- ✅ IMPLEMENTATION_GUIDE.md
- ✅ QUICKSTART.md
- ✅ EXAMPLES.py
- ✅ ARCHITECTURE.md
- ✅ IMPLEMENTATION_SUMMARY.md (this file)
- ✅ README.md (redesigned)
- ✅ INDEX.md (navigation guide)

**Configuration**
- ✅ requirements.txt

---

**You're all set! Begin with [QUICKSTART.md](QUICKSTART.md) 🚀**
