# Quick Start Guide - AI Code Understanding System

## 5-Minute Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Your First Project

```bash
# Using Python
python -c "
from project_analyser.ai.pipeline import AICodeUnderstandingPipeline

pipeline = AICodeUnderstandingPipeline('C:/Sushant/AI Learning/Projects')

# Create your first project
pipeline.setup_project(
    project_name='my_microservices',
    project_root='C:/Sushant/testWorkspace/microservices/testProj',
    project_type='microservices'
)
"
```

OR using CLI:

```bash
python -m project_analyser.ai.cli setup-project \
  --name my_microservices \
  --root "C:\Sushant\testWorkspace\microservices\testProj"
```

### 3. Run Complete Pipeline

```python
from project_analyser.ai.pipeline import AICodeUnderstandingPipeline

pipeline = AICodeUnderstandingPipeline('C:/Sushant/AI Learning/Projects')

summary = pipeline.run_full_pipeline(
    project_name='my_microservices',
    project_root='C:/Sushant/testWorkspace/microservices/testProj'
)

print(f"✓ Training complete! Execution time: {summary['execution_time_seconds']:.2f}s")
```

OR using CLI:

```bash
python -m project_analyser.ai.cli run-pipeline \
  --name my_microservices \
  --root "C:\Sushant\testWorkspace\microservices\testProj"
```

### 4. Query Your Code

```python
from project_analyser.ai.pipeline import AICodeUnderstandingPipeline

pipeline = AICodeUnderstandingPipeline('C:/Sushant/AI Learning/Projects')

response = pipeline.query_code(
    "What does the UserController class do?"
)

print(f"Q: {response.query}")
print(f"A: {response.response}")
print(f"Confidence: {response.confidence:.1%}")
```

OR using CLI:

```bash
python -m project_analyser.ai.cli query \
  --question "What does the UserController class do?"
```

---

## Step-by-Step Execution

If you want to run each phase separately:

### Phase 1: Index Code

```python
from project_analyser.ai.pipeline import AICodeUnderstandingPipeline

pipeline = AICodeUnderstandingPipeline('C:/Sushant/AI Learning/Projects')

chunks, stats = pipeline.index_project('my_microservices')

print(f"Indexed {len(chunks)} code chunks")
print(f"Stats: {stats}")
```

### Phase 2: Create Dataset

```python
version_key, dataset_stats = pipeline.create_dataset('my_microservices', chunks)

print(f"Created dataset: {version_key}")
print(f"Training pairs: {dataset_stats['total_pairs']}")
```

### Phase 3: Tokenize

```python
tok_stats = pipeline.tokenize_dataset('my_microservices', version_key)

print(f"Total tokens: {tok_stats['total_tokens']}")
print(f"Avg tokens per pair: {tok_stats['avg_tokens_per_pair']:.2f}")
```

### Phase 4: Train Model

```python
train_metrics = pipeline.train_model(
    'my_microservices',
    tok_stats['tokenized_file']
)

print(f"Training loss: {train_metrics['train_loss']}")
print(f"Model checkpoint: {train_metrics['checkpoint']}")
```

### Phase 5: Setup Inference

```python
import os

project_info = pipeline.project_manager.get_project_info('my_microservices')
checkpoint_path = train_metrics['checkpoint']
chunks_file = os.path.join(project_info['project_dir'], '.code_index', 'chunks.jsonl')

pipeline.setup_inference(checkpoint_path, chunks_file)

response = pipeline.query_code("What does UserController do?")
print(response.response)
```

---

## Understanding the Output

### Indexing Output

```
Index Statistics:
  Total files: 12
  Processed files: 12
  New files: 12
  Total chunks: 48
  Chunks by type: 
    {'file': 12, 'class': 24, 'method': 120}
```

### Dataset Creation Output

```
Generated Training Pairs:
  Total pairs: 360
  Pairs by type: 
    {'explain_method': 120, 'explain_class': 48, 'qa': 120, 'explain_file': 12, 'explain_logic': 60}
```

### Tokenization Output

```
Tokenization Statistics:
  Total pairs: 360
  Total tokens: 125,400
  Avg tokens per pair: 348.33
  Total batches: 45
  Avg batch size: 8
```

### Training Output

```
Training:
  Model version: 1
  Train loss: 2.15
  Checkpoint: ./models/gpt2-code/checkpoints/checkpoint-v1
```

---

## Testing & Verification

### Test 1: Basic Index

```python
from project_analyser.ai.dataset.codeIndexer import CodeIndexer

indexer = CodeIndexer('C:/Sushant/testWorkspace/microservices/testProj')
chunks, stats = indexer.incremental_index()

assert len(chunks) > 0, "No chunks extracted!"
assert stats['total_chunks'] == len(chunks)
print("✓ Indexing test passed")
```

### Test 2: Dataset Generation

```python
from project_analyser.ai.dataset.datasetManager import IncrementalDatasetBuilder

builder = IncrementalDatasetBuilder('C:/Sushant/AI Learning/Datasets')
version_key, stats = builder.build_dataset(chunks, 'test_project')

assert stats['total_pairs'] > 0, "No training pairs generated!"
print("✓ Dataset generation test passed")
```

### Test 3: Tokenization

```python
from project_analyser.ai.dataset.codeTokenizer import TokenizationPipeline

pipeline = TokenizationPipeline()
stats = pipeline.process_dataset(input_file, output_file)

assert stats['total_pairs'] > 0
assert stats['total_tokens'] > 0
print("✓ Tokenization test passed")
```

---

## Common Tasks

### Task 1: Analyze Your Project

```python
from project_analyser.ai.pipeline import AICodeUnderstandingPipeline

pipeline = AICodeUnderstandingPipeline('C:/Sushant/AI Learning/Projects')
analysis = pipeline.analyze_project('my_microservices')

print(f"File types found: {analysis['file_types'].keys()}")
print(f"Total files by type: {analysis['total_files_by_type']}")
```

### Task 2: Check Training Progress

```python
from project_analyser.ai.llm.incrementalTraining import CodeModelTrainer

trainer = CodeModelTrainer()
history = trainer.checkpoint_manager.get_checkpoint_history()

for checkpoint in history:
    print(f"v{checkpoint['version']}: {checkpoint['metrics']}")
```

### Task 3: Update Model After Code Changes

```python
from project_analyser.ai.pipeline import AICodeUnderstandingPipeline

pipeline = AICodeUnderstandingPipeline('C:/Sushant/AI Learning/Projects')

# Detect and auto-update
has_changes = pipeline.detect_and_update('my_microservices')

if has_changes:
    print("✓ Model updated with new code changes!")
else:
    print("No code changes detected")
```

### Task 4: Batch Query

```python
from project_analyser.ai.pipeline import AICodeUnderstandingPipeline

pipeline = AICodeUnderstandingPipeline('C:/Sushant/AI Learning/Projects')

questions = [
    "What does UserController do?",
    "Explain the UserService class",
    "What are the main methods in UserRepository?",
]

for question in questions:
    response = pipeline.query_code(question)
    print(f"Q: {question}")
    print(f"A: {response.response}\n")
```

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution**: Make sure you're running from the workspace root and Python path includes the project:

```bash
set PYTHONPATH=%PYTHONPATH%;C:\Sushant\AI Learning\AI_Learnings
```

### Issue: CUDA/GPU not detected

**Solution**: This is normal. System will fall back to CPU (slower but works fine)

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Issue: Out of memory during training

**Solution**: Reduce batch size:

```python
from project_analyser.ai.llm.incrementalTraining import TrainingConfig

config = TrainingConfig(
    per_device_train_batch_size=2,  # Reduce from 4
    gradient_accumulation_steps=2    # But accumulate gradients
)
```

### Issue: Model generating poor quality answers

**Solution**: This is normal for initial training. Try:
1. Train for more epochs
2. Use more training data
3. Improve training descriptions in dataset
4. Use a larger base model (gpt2-medium, gpt2-large)

---

## Next Steps

1. **Read the full guide**: See `IMPLEMENTATION_GUIDE.md` for detailed documentation
2. **Explore different projects**: Set up multiple projects and compare
3. **Fine-tune the system**: Adjust training hyperparameters for better results
4. **Extend file handlers**: Add support for Python, JavaScript, or YAML files
5. **Build custom queries**: Create domain-specific question prompts

---

## Getting Help

**Check the implementation guide**: `IMPLEMENTATION_GUIDE.md`

**Key modules**:
- `project_analyser/ai/dataset/codeIndexer.py` - Code extraction
- `project_analyser/ai/dataset/datasetManager.py` - Dataset management
- `project_analyser/ai/dataset/codeTokenizer.py` - Tokenization
- `project_analyser/ai/llm/incrementalTraining.py` - Model training
- `project_analyser/ai/llm/codeInference.py` - Query interface
- `project_analyser/ai/projectManagement.py` - Multi-project support
- `project_analyser/ai/pipeline.py` - End-to-end orchestration
- `project_analyser/ai/cli.py` - Command-line interface

---

**Happy coding! 🚀**
