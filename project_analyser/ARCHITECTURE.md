# System Architecture & Data Flow

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                              │
│  ┌──────────────────────────┐  ┌──────────────────────────┐                │
│  │  Python API              │  │  CLI Interface           │                │
│  │  (Direct imports)        │  │  (Command-line tool)     │                │
│  └────────┬─────────────────┘  └──────────┬───────────────┘                │
└───────────┼──────────────────────────────┼────────────────────────────────┘
            │                              │
            └──────────────┬───────────────┘
                           │
┌──────────────────────────▼────────────────────────────────────────────────┐
│              END-TO-END PIPELINE ORCHESTRATOR                              │
│         (AICodeUnderstandingPipeline - Main Entry Point)                   │
│                                                                             │
│  Coordinates 7 phases: Index → Dataset → Tokenize → Train → Inference    │
│  Maintains project configs, datasets, models, and logs                    │
│                                                                             │
└──────────────┬──────────────┬──────────────┬──────────────┬──────────────┘
               │              │              │              │
    ┌──────────▼──────┐ ┌─────▼──────┐ ┌────▼────────┐ ┌───▼──────────────┐
    │   PHASE 1       │ │   PHASE 2  │ │  PHASE 3    │ │   PHASE 4        │
    │   INDEXING      │ │  DATASET   │ │ TOKENIZING  │ │    TRAINING      │
    │                 │ │  CREATION  │ │             │ │                  │
    │  CodeIndexer    │ │ Dataset    │ │ Code        │ │  Code            │
    │  Extracts:      │ │ Manager    │ │ Tokenizer   │ │  ModelTrainer    │
    │  - Methods      │ │ Generates: │ │ Produces:   │ │  Performs:       │
    │  - Classes      │ │ - explain_ │ │ - token IDs │ │  - LoRA setup    │
    │  - Files        │ │   method   │ │ - attention │ │  - Fine-tuning   │
    │  - Code blocks  │ │ - explain_ │ │   masks     │ │  - Checkpoints   │
    │                 │ │   class    │ │ - labels    │ │  - Validation    │
    │  Output:        │ │ - explain_ │ │ - batches   │ │                  │
    │  chunks.jsonl   │ │   file     │ │             │ │  Output:         │
    │  metadata.json  │ │ - qa       │ │ Output:     │ │  checkpoints/    │
    │                 │ │ - logic    │ │ tokenized_  │ │  checkpoint_     │
    │  Input:         │ │            │ │ vN.jsonl    │ │  metadata.json   │
    │  Source code    │ │ Output:    │ │             │ │                  │
    │  files          │ │ versions/  │ │ Input:      │ │  Input:          │
    │                 │ │ project_vN │ │ versions/   │ │  tokenized_file  │
    │                 │ │ .jsonl     │ │ project_vN. │ │                  │
    │                 │ │            │ │ jsonl       │ │                  │
    └─────────────────┘ └────────────┘ └─────────────┘ └──────────────────┘
```

```
    ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────────────────┐
    │   PHASE 5        │ │   PHASE 6        │ │    PHASE 7 (Continuous)     │
    │   INFERENCE      │ │   PROJECT MGT    │ │    LEARNING                 │
    │                  │ │                  │ │                             │
    │  CodeExplainer   │ │  ProjectManager  │ │  ChangeDetector             │
    │  Performs:       │ │  Handles:        │ │  Monitors:                  │
    │  - Load model    │ │  - Multi-project │ │  - File changes             │
    │  - Retrieval     │ │  - Configs       │ │  - Triggers reindex         │
    │  - RAG fetch     │ │  - File handlers │ │  - Auto-retraining          │
    │  - Generate      │ │  - Analysis      │ │                             │
    │    answers       │ │                  │ │  Updates:                   │
    │  - Score         │ │  Output:         │ │  - New dataset versions     │
    │    confidence    │ │  projects.json   │ │  - Model checkpoints        │
    │  - Cache results │ │                  │ │  - Training history         │
    │                  │ │  Input:          │ │                             │
    │  Output:         │ │  Project configs │ │  Input:                     │
    │  QueryResponse   │ │                  │ │  File system changes        │
    │  - response      │ │                  │ │                             │
    │  - citations     │ │                  │ │                             │
    │  - confidence    │ │                  │ │                             │
    │                  │ │                  │ │                             │
    │  Input:          │ │                  │ │                             │
    │  Query string    │ │                  │ │                             │
    │  checkpoints/    │ │                  │ │                             │
    │  chunks.jsonl    │ │                  │ │                             │
    └──────────────────┘ └──────────────────┘ └─────────────────────────────┘
```

## Data Flow Diagram

```
SOURCE CODE FILES
       │
       │  (Extract structure)
       ▼
   INDEXING PHASE
       │
   ┌───▼───────────────────────────────────────────┐
   │  JavaCodeExtractor / MultiFormatExtractor     │
   │  - Regex-based parsing                        │
   │  - AST extraction for methods/classes         │
   │  - Metadata collection (line numbers, names)  │
   │  - Checksum calculation                       │
   └───┬───────────────────────────────────────────┘
       │
   ┌───▼──────────────────────────────────────────┐
   │  CodeChunk Objects                           │
   │  ├─ chunk_id, file_path, chunk_type         │
   │  ├─ code_content, description                │
   │  ├─ start_line, end_line                     │
   │  ├─ checksum (for change detection)          │
   │  └─ metadata (parent_class, etc.)            │
   └───┬──────────────────────────────────────────┘
       │  (Store as JSONL)
       │
   ┌───▼──────────────────────────────────────────┐
   │  chunks.jsonl                                │
   │  metadata.json (with checksums)              │
   └───┬──────────────────────────────────────────┘
       │
       │  (Generate training pairs)
       ▼
   DATASET CREATION PHASE
       │
   ┌───▼──────────────────────────────────────────┐
   │  TrainingPairGenerator                       │
   │  - Multiple pair types per chunk             │
   │  - Automatic description inference           │
   │  - Synthetic Q&A generation                  │
   └───┬──────────────────────────────────────────┘
       │
   ┌───▼──────────────────────────────────────────┐
   │  TrainingPair Objects                        │
   │  ├─ input_text (code + question)            │
   │  ├─ output_text (explanation)                │
   │  ├─ pair_type (explain_method, etc.)        │
   │  └─ metadata (tags, references)              │
   └───┬──────────────────────────────────────────┘
       │  (Store as versioned JSONL)
       │
   ┌───▼──────────────────────────────────────────┐
   │  versions/project_v1.jsonl                   │
   │  versions/project_v2.jsonl (incremental)     │
   │  dataset_metadata.json (version tracking)    │
   └───┬──────────────────────────────────────────┘
       │
       │  (Tokenize text to tokens)
       ▼
   TOKENIZATION PHASE
       │
   ┌───▼──────────────────────────────────────────┐
   │  CodeTokenizer                               │
   │  - GPT-2 tokenizer (or custom)              │
   │  - Dynamic batching by token count           │
   │  - Proper label masking                      │
   │  - Context preservation                      │
   └───┬──────────────────────────────────────────┘
       │
   ┌───▼──────────────────────────────────────────┐
   │  TokenizedPair Objects                       │
   │  ├─ input_ids (list of token IDs)           │
   │  ├─ attention_mask (masking)                 │
   │  ├─ labels (for loss computation)            │
   │  └─ token_count                              │
   └───┬──────────────────────────────────────────┘
       │  (Stored as JSONL)
       │
   ┌───▼──────────────────────────────────────────┐
   │  tokenized_v1.jsonl                          │
   │  (Ready for model training)                  │
   └───┬──────────────────────────────────────────┘
       │
       │  (Fine-tune language model)
       ▼
   TRAINING PHASE
       │
   ┌───▼──────────────────────────────────────────┐
   │  CodeModelTrainer                            │
   │  - Load base model (GPT-2)                   │
   │  - Apply LoRA config                         │
   │  - Setup training arguments                  │
   │  - Execute training loop                     │
   │  - Save checkpoints                          │
   └───┬──────────────────────────────────────────┘
       │
   ┌───▼──────────────────────────────────────────┐
   │  Model Checkpoints                           │
   │  ├─ checkpoint-v1/                           │
   │  │  ├─ pytorch_model.bin (LoRA weights)      │
   │  │  ├─ config.json (model config)            │
   │  │  └─ training_config.json                  │
   │  ├─ checkpoint-v2/ (incremental update)      │
   │  └─ checkpoint_metadata.json (history)       │
   └───┬──────────────────────────────────────────┘
       │
       │  (Load and use trained model)
       ▼
   INFERENCE PHASE
       │
   ┌───▼──────────────────────────────────────────┐
   │  User Query                                  │
   │  "What does UserController do?"              │
   └───┬──────────────────────────────────────────┘
       │
   ┌───▼──────────────────────────────────────────┐
   │  CodeRetriever (RAG)                         │
   │  - Query cache check (fast path)             │
   │  - Semantic search (keyword matching)        │
   │  - Retrieve top-k relevant chunks            │
   │  - Rank by relevance                         │
   └───┬──────────────────────────────────────────┘
       │
   ┌───▼──────────────────────────────────────────┐
   │  Prompt Engineering                          │
   │  ┌─────────────────────────────────────────┐ │
   │  │ RELEVANT CODE CONTEXT:                  │ │
   │  │ CLASS: UserController                   │ │
   │  │ [code snippet from chunk]               │ │
   │  │                                         │ │
   │  │ QUESTION: What does UserController do? │ │
   │  │                                         │ │
   │  │ ANSWER:                                 │ │
   │  └─────────────────────────────────────────┘ │
   └───┬──────────────────────────────────────────┘
       │
   ┌───▼──────────────────────────────────────────┐
   │  Model Generation                            │
   │  - Tokenize prompt                           │
   │  - Forward pass through model                │
   │  - Sampling (temperature, top-p)             │
   │  - Token generation (auto-regressive)        │
   │  - Decode tokens to text                     │
   └───┬──────────────────────────────────────────┘
       │
   ┌───▼──────────────────────────────────────────┐
   │  QueryResponse Object                        │
   │  ├─ response (generated text)               │
   │  ├─ confidence (source-based scoring)       │
   │  ├─ source_citations (which chunks used)    │
   │  ├─ generation_time_ms                      │
   │  └─ model_version                           │
   └───┬──────────────────────────────────────────┘
       │  (Cache result)
       │
   ┌───▼──────────────────────────────────────────┐
   │  QueryCache                                  │
   │  - Cache hit returns instantly (0.01s)       │
   │  - Persistent JSON file                      │
   │  - Reduces model invocations                 │
   └───┬──────────────────────────────────────────┘
       │
       ▼
    USER OUTPUT
```

## Directory Structure & Data Organization

```
workspace_root/
│
├── projects/                          # Project registry
│   ├── projects.json                 # All project configs
│   │
│   └── my_microservices/             # Single project
│       │
│       ├── .code_index/              # Phase 1: Indexing output
│       │   ├── chunks.jsonl          # Extracted code chunks (JSONL)
│       │   └── metadata.json         # File checksums & versions
│       │
│       └── datasets/                 # Phases 2-3: Dataset output
│           ├── dataset_metadata.json # Dataset tracking
│           ├── versions.json         # Version history
│           │
│           ├── versions/
│           │   ├── my_microservices_v1.jsonl      # Training pairs v1
│           │   ├── my_microservices_v2.jsonl      # Training pairs v2
│           │   └── ...
│           │
│           ├── my_microservices_v1_tokenized.jsonl  # Phase 3: Tokens
│           ├── my_microservices_v2_tokenized.jsonl
│           └── ...
│
├── models/                           # Phase 4: Training output
│   └── gpt2-code/
│       ├── checkpoints/
│       │   ├── checkpoint-v1/
│       │   │   ├── pytorch_model.bin # LoRA weights
│       │   │   ├── adapter_config.json
│       │   │   ├── config.json       # Model config
│       │   │   └── training_config.json
│       │   │
│       │   ├── checkpoint-v2/       # Incremental update
│       │   └── ...
│       │
│       ├── checkpoint_metadata.json # Training history
│       │
│       └── runs/                    # TensorBoard logs
│           ├── events.out.tfevents...
│           └── ...
│
└── artifacts/                       # System artifacts
    ├── cache/
    │   └── query_cache.json        # Phase 5: Cached queries
    │
    └── logs/
        └── pipeline.jsonl          # Phase 7: Event log
```

## State & Version Management

```
File Change Detection → Metadata Update → Dataset Versioning → Model Versioning

┌─────────────────────────────────────────────────────────────────┐
│  Metadata Tracking                                              │
│                                                                 │
│  chunks.metadata.json:                                         │
│  {                                                             │
│    "UserController.java": {                                    │
│      "checksum": "abc123def456...",                           │
│      "last_indexed": "2024-01-15T10:30:00",                   │
│      "chunks_count": 15                                        │
│    },                                                          │
│    ...                                                          │
│  }                                                              │
│                                                                 │
│  When file changes → checksum differs → file marked for reindex│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Dataset Versioning                                             │
│                                                                 │
│  dataset_metadata.json:                                        │
│  {                                                             │
│    "my_microservices": {                                       │
│      "current_version": 2,                                     │
│      "total_pairs": 1250,                                      │
│      "last_updated": "2024-01-15T11:00:00"                    │
│    }                                                            │
│  }                                                              │
│                                                                 │
│  versions.json:                                                │
│  {                                                             │
│    "my_microservices_v1": {                                    │
│      "version": 1,                                             │
│      "total_pairs": 1000,                                      │
│      "created_at": "2024-01-15T10:00:00"                      │
│    },                                                          │
│    "my_microservices_v2": {                                    │
│      "version": 2,                                             │
│      "total_pairs": 1250,  ← Incremental (250 new pairs)      │
│      "created_at": "2024-01-15T11:00:00"                      │
│    }                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Model Versioning & Checkpoints                                 │
│                                                                 │
│  checkpoint_metadata.json:                                     │
│  {                                                             │
│    "v1": {                                                      │
│      "version": 1,                                             │
│      "created_at": "2024-01-15T11:30:00",                     │
│      "train_loss": 2.15,                                       │
│      "eval_loss": 2.42,                                        │
│      "path": "./checkpoints/checkpoint-v1"                     │
│    },                                                          │
│    "v2": {                                                      │
│      "version": 2,                                             │
│      "created_at": "2024-01-15T12:45:00",                     │
│      "train_loss": 1.89,  ← Improvement from v1               │
│      "eval_loss": 2.05,                                        │
│      "path": "./checkpoints/checkpoint-v2"                     │
│    }                                                            │
│  }                                                              │
│                                                                 │
│  Each checkpoint contains full LoRA adapter weights             │
│  Can revert to any previous version                             │
└─────────────────────────────────────────────────────────────────┘
```

## Incremental Learning Cycle

```
Week 1: Initial Training
┌───────────────────────────┐
│ Index: 50 Java files      │ → 150 chunks
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ Dataset: Generate pairs   │ → 450 training pairs (v1)
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ Tokenize & Train          │ → Model v1
└────────────┬──────────────┘
             │
             ▼
        Model Ready ✓

Week 2: Code Changes
     (3 files modified, 2 new files)
             │
             ▼
┌───────────────────────────┐
│ Change Detection          │ → Detect 5 file changes
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ Incremental Index         │ → 150 unchanged + 25 new = 175 chunks
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ Incremental Dataset       │ → 450 unchanged + 75 new = 525 pairs (v2)
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ Incremental Training      │ → Load v1 checkpoint, train on 75 new pairs
│ (on new pairs only)       │ → Model v2 (preserves v1 knowledge)
└────────────┬──────────────┘
             │
             ▼
        Model Updated ✓

End Result: Model knows about both old AND new code
```

---

This architecture ensures:
- ✅ **Efficiency**: Incremental processing, caching, batching
- ✅ **Scalability**: Multi-project, extensible file handlers
- ✅ **Maintainability**: Clear separation of concerns
- ✅ **Continuous Learning**: Auto-update without full retraining
- ✅ **Reproducibility**: Full versioning and checkpoints
- ✅ **Transparency**: Source attribution and logging

