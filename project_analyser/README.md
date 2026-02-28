# AI Code Understanding System

A production-ready system for understanding microservices and other codebases using fine-tuned LLMs.

## 🚀 What It Does

**Feed Your Code → Ask Questions → Get Explanations**

Transform your Java microservices, documentation, or any codebase into an AI-powered knowledge base.

```
Java/Code Files
       ↓
   INDEX CODE        (Extract methods, classes, structure)
       ↓
  CREATE DATASET     (Generate training pairs)
       ↓
   TOKENIZE          (Convert to token sequences)
       ↓
   TRAIN MODEL       (Fine-tune GPT-2 with LoRA)
       ↓
   INFERENCE API     (Query with RAG & source attribution)
       ↓
Understand Any Code!
```

## ✨ Key Features

- **🔍 Intelligent Code Indexing**: Extract methods, classes, files with rich metadata
- **📚 Automatic Dataset Creation**: Generate training pairs from code chunks
- **⚡ LoRA Fine-Tuning**: Parameter-efficient model training (only 8MB overhead)
- **🎯 Multi-Type Queries**: Explain methods, classes, files, logic, find related code
- **🔄 Continuous Learning**: Auto-update model when code changes
- **🏗️ Multi-Project Support**: Manage multiple projects with different configs
- **📖 RAG (Retrieval-Augmented Generation)**: Fetch context before answering
- **🏷️ Source Attribution**: Know which code chunks were referenced
- **⚙️ Extensible Architecture**: Add file handlers for Python, JavaScript, YAML, etc.
- **💾 Incremental Updates**: Only re-process changed files

## 📋 System Components

| Component | Purpose | Key Classes |
|-----------|---------|------------|
| **Indexing** | Extract code chunks | `CodeIndexer`, `JavaCodeExtractor`, `MultiFormatExtractor` |
| **Dataset** | Generate training pairs | `IncrementalDatasetBuilder`, `TrainingPairGenerator` |
| **Tokenization** | Convert to tokens | `CodeDatasetTokenizer`, `DynamicBatcher` |
| **Training** | Fine-tune LLM | `CodeModelTrainer`, `LoRA` with PEFT |
| **Inference** | Answer questions | `CodeExplainerAPI`, `CodeModelInference` |
| **Management** | Multi-project support | `ProjectManager`, `ChangeDetector` |
| **Pipeline** | End-to-end orchestration | `AICodeUnderstandingPipeline` |
| **CLI** | Command-line interface | `CodeUnderstandingCLI` |

## 🎯 Use Cases

✅ **What does this method do?** - Instant code documentation  
✅ **Explain the UserController class** - Understand microservice structure  
✅ **What are all HTTP endpoints?** - Auto-generate API docs  
✅ **How does authentication work?** - Cross-file code understanding  
✅ **Onboard new developers** - Interactive codebase exploration  
✅ **Generate code comments** - AI-powered documentation  

## 📁 Project Structure

```
AI_Learnings/
├── project_analyser/ai/
│   ├── dataset/
│   │   ├── codeIndexer.py           # Code extraction
│   │   ├── datasetManager.py        # Dataset versioning
│   │   └── codeTokenizer.py         # Tokenization
│   ├── llm/
│   │   ├── incrementalTraining.py   # LoRA fine-tuning
│   │   └── codeInference.py         # Query interface
│   ├── projectManagement.py         # Multi-project support
│   ├── pipeline.py                  # End-to-end orchestration
│   └── cli.py                       # Command-line tool
├── IMPLEMENTATION_GUIDE.md          # Detailed technical docs
├── QUICKSTART.md                    # 5-minute setup
├── EXAMPLES.py                      # Usage examples
└── requirements.txt                 # Dependencies
```

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Project
```python
from project_analyser.ai.pipeline import AICodeUnderstandingPipeline

pipeline = AICodeUnderstandingPipeline('C:/Projects')

pipeline.setup_project(
    project_name='my_microservices',
    project_root='C:/path/to/code',
    project_type='microservices'
)
```

### 3. Run Complete Pipeline
```python
summary = pipeline.run_full_pipeline(
    project_name='my_microservices',
    project_root='C:/path/to/code'
)
```

### 4. Ask Questions
```python
response = pipeline.query_code(
    "What does the UserController class do?"
)
print(response.response)
print(f"Confidence: {response.confidence:.1%}")
```

### Or Use CLI
```powershell
python -m project_analyser.ai.cli run-pipeline \
  --name my_microservices \
  --root C:/path/to/code

python -m project_analyser.ai.cli query \
  --question "What does UserController do?"
```

## 📊 Performance

| Operation | Time (Single Project) |
|-----------|----------------------|
| Index 50 Java files | ~2s |
| Generate 450 training pairs | ~0.5s |
| Tokenize dataset | ~1s |
| Initial training | ~15min (GPU) / 45min (CPU) |
| First inference | ~2-5s |
| Subsequent queries | ~0.3-2s |
| Cached query | ~0.01s |

## 🔧 Advanced Usage

### Continuous Learning - Auto-Update on Code Changes
```python
while True:
    has_changes = pipeline.detect_and_update('my_microservices')
    if has_changes:
        print("✓ Model updated!")
    time.sleep(3600)  # Check hourly
```

### Step-by-Step Control
```python
chunks, _ = pipeline.index_project('my_microservices')
version_key, _ = pipeline.create_dataset('my_microservices', chunks)
stats = pipeline.tokenize_dataset('my_microservices', version_key)
metrics = pipeline.train_model('my_microservices', stats['tokenized_file'])
```

### Multi-Project Queries
```python
projects = pipeline.project_manager.list_projects()
for project in projects:
    response = pipeline.query_code(f"What does {project} do?")
```

### Batch Processing
```python
questions = [
    "What are the main services?",
    "How does authentication work?",
    "List all REST endpoints"
]
for q in questions:
    response = pipeline.query_code(q)
    print(response.response)
```

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup and basic usage
- **[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)** - Complete technical documentation
- **[EXAMPLES.py](EXAMPLES.py)** - Runnable example code

## 🛠️ Configuration

### Training Parameters
```python
from project_analyser.ai.llm.incrementalTraining import TrainingConfig

config = TrainingConfig(
    model_name="gpt2",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    use_lora=True,
    lora_r=8
)
```

### Project Setup
```python
from project_analyser.ai.projectManagement import ProjectConfig

config = ProjectConfig(
    project_name="my_project",
    project_root="/path/to/code",
    project_type="microservices",
    file_types=['.java', '.xml', '.properties']
)
```

## 🔌 Extensibility

### Add New File Type Handler
```python
from project_analyser.ai.projectManagement import FileTypeHandler

class PythonFileHandler(FileTypeHandler):
    def can_handle(self, file_path):
        return file_path.endswith('.py')
    
    def extract_metadata(self, file_path):
        # Your extraction logic
        pass
    
    def generate_training_description(self, content):
        # Your description logic
        pass
```

## 📈 Roadmap

- [ ] Vector embeddings for semantic search (FAISS)
- [ ] Web UI (FastAPI + React)
- [ ] Support for more models (LLaMA, Mistral)
- [ ] Code diff analysis (what changed between versions)
- [ ] Dependency mapping (visualize code relationships)
- [ ] Type inference (understand Java type systems)
- [ ] Cross-project learning
- [ ] Active learning (ask user for clarification)

## 🐛 Troubleshooting

**Q: Out of memory during training?**  
A: Reduce `per_device_train_batch_size` or use gradient accumulation

**Q: Poor model answers?**  
A: Train longer (more epochs), use more data, improve descriptions

**Q: Slow inference?**  
A: Check query caching, use RAG to reduce output length

See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for more troubleshooting.

## 📦 Dependencies

- Python 3.8+
- torch >= 2.0.0
- transformers >= 4.30.0
- peft >= 0.4.0 (LoRA)
- datasets >= 2.10.0

## 📄 License

MIT License - See LICENSE file

## 🤝 Contributing

Contributions welcome! Areas for help:
- [ ] Add file handlers (Python, JavaScript, Rust)
- [ ] Implement semantic search with embeddings
- [ ] Build web UI
- [ ] Improve training data generation
- [ ] Performance optimizations
- [ ] Documentation improvements

## 📞 Support

- Check [QUICKSTART.md](QUICKSTART.md) for quick help
- See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for detailed docs
- Review [EXAMPLES.py](EXAMPLES.py) for code examples
- Open an issue for bugs/features

---

**Happy Code Understanding! 🎉**
