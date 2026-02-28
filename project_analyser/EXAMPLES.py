"""
Example Notebook: AI Code Understanding System
Demonstrates complete workflow with real examples
"""

# ============================================================================
# EXAMPLE 1: Quick Setup & Training on Your Microservices
# ============================================================================

print("="*80)
print("EXAMPLE 1: Quick Setup & Training")
print("="*80)

from project_analyser.ai.pipeline import AICodeUnderstandingPipeline

# Initialize pipeline with your workspace
workspace = "C:/Sushant/AI Learning/Projects"
pipeline = AICodeUnderstandingPipeline(workspace)

# Setup project
pipeline.setup_project(
    project_name="my_microservices",
    project_root="C:/Sushant/testWorkspace/microservices/testProj",
    project_type="microservices"
)

# Run complete pipeline (this takes time on first run)
summary = pipeline.run_full_pipeline(
    project_name="my_microservices",
    project_root="C:/Sushant/testWorkspace/microservices/testProj"
)

print(f"\n✓ Pipeline complete in {summary['execution_time_seconds']:.2f}s")
print(f"  Index stats: {summary['index_stats']}")
print(f"  Dataset size: {summary['dataset_stats']['total_pairs']} pairs")

# ============================================================================
# EXAMPLE 2: Querying Your Code
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 2: Querying Your Code")
print("="*80)

questions = [
    "What does the UserController class do?",
    "Explain the login method",
    "What is the purpose of the AuthService?",
    "How does the database connection work?",
]

for question in questions:
    print(f"\n📝 Question: {question}")
    response = pipeline.query_code(question)
    print(f"💡 Answer: {response.response}")
    print(f"📊 Confidence: {response.confidence:.1%}")

    if response.source_citations:
        print(f"📚 Sources:")
        for cite in response.source_citations:
            print(f"   - {cite['chunk_type'].upper()}: {cite['name']} ({cite['file']})")

# ============================================================================
# EXAMPLE 3: Step-by-Step Analysis
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 3: Step-by-Step Code Analysis")
print("="*80)

# Analyze project structure
analysis = pipeline.analyze_project("my_microservices")

print(f"\nProject Structure:")
print(f"  File types recognized: {list(analysis['file_types'].keys())}")

if 'JavaFileHandler' in analysis['file_types']:
    java_info = analysis['file_types']['JavaFileHandler']
    print(f"\n  Java Files ({java_info['count']} total):")
    for example in java_info['examples'][:3]:
        print(f"    - {example}")

# Get project info
project_info = pipeline.project_manager.get_project_info("my_microservices")
print(f"\nDatasets created: {len(project_info['datasets'])}")
for dataset in project_info['datasets']:
    print(f"  - {dataset}")

# ============================================================================
# EXAMPLE 4: Code Chunk Inspection
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 4: Inspecting Code Chunks")
print("="*80)

# Get indexed chunks
project_info = pipeline.project_manager.get_project_info("my_microservices")
chunks_file = f"{project_info['project_dir']}/.code_index/chunks.jsonl"

import json
chunks = []
try:
    with open(chunks_file, 'r') as f:
        for line in f:
            chunks.append(json.loads(line))
except:
    print("Chunks file not found yet. Run pipeline first.")
    chunks = []

if chunks:
    print(f"\nTotal chunks: {len(chunks)}")

    # Group by type
    by_type = {}
    for chunk in chunks:
        chunk_type = chunk.get('chunk_type', 'unknown')
        if chunk_type not in by_type:
            by_type[chunk_type] = []
        by_type[chunk_type].append(chunk)

    print(f"\nChunks by type:")
    for chunk_type, type_chunks in by_type.items():
        print(f"  {chunk_type}: {len(type_chunks)}")

    # Show example method chunk
    method_chunks = [c for c in chunks if c.get('chunk_type') == 'method']
    if method_chunks:
        print(f"\nExample Method Chunk:")
        example = method_chunks[0]
        print(f"  Name: {example['chunk_name']}")
        print(f"  Class: {example.get('parent_class', 'N/A')}")
        print(f"  Lines: {example['start_line']}-{example['end_line']}")
        print(f"  Description: {example.get('description', 'N/A')}")

# ============================================================================
# EXAMPLE 5: Training Dataset Inspection
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 5: Inspecting Training Dataset")
print("="*80)

from project_analyser.ai.dataset.datasetManager import IncrementalDatasetBuilder

dataset_dir = f"{project_info['project_dir']}/datasets"
builder = IncrementalDatasetBuilder(dataset_dir)

# Get training pairs
pairs = builder.get_training_dataset("my_microservices")

if pairs:
    print(f"\nTotal training pairs: {len(pairs)}")

    # Group by type
    by_pair_type = {}
    for pair in pairs:
        pair_type = pair.pair_type
        if pair_type not in by_pair_type:
            by_pair_type[pair_type] = []
        by_pair_type[pair_type].append(pair)

    print(f"\nPairs by type:")
    for pair_type, type_pairs in by_pair_type.items():
        print(f"  {pair_type}: {len(type_pairs)}")

    # Show example pair
    if by_pair_type:
        example_type = list(by_pair_type.keys())[0]
        example_pair = by_pair_type[example_type][0]
        print(f"\nExample Training Pair ({example_type}):")
        print(f"  Input: {example_pair.input_text[:100]}...")
        print(f"  Output: {example_pair.output_text[:100]}...")

# ============================================================================
# EXAMPLE 6: Continuous Learning (File Change Detection)
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 6: Continuous Learning - Change Detection")
print("="*80)

# Check for code changes
print("Checking for code changes...")
has_changes = pipeline.detect_and_update("my_microservices")

if has_changes:
    print("✓ Changes detected! Model has been updated.")
else:
    print("No changes detected in code.")

# ============================================================================
# EXAMPLE 7: Multi-Project Management
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 7: Multi-Project Management")
print("="*80)

# List all projects
projects = pipeline.project_manager.list_projects()

print(f"\nManaged Projects ({len(projects)} total):")
for project in projects:
    status = "✓ Indexed" if project.get('total_chunks', 0) > 0 else "○ Not indexed"
    chunks = project.get('total_chunks', 0)
    print(f"  {status}: {project['project_name']:<30} ({chunks} chunks)")

# ============================================================================
# EXAMPLE 8: Advanced Querying with RAG
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 8: Advanced Querying with Source Attribution")
print("="*80)

# Complex query with expected context
query = "What are all the HTTP endpoints defined in the REST controllers?"

print(f"\nQuery: {query}")
response = pipeline.query_code(query)

print(f"\nResponse:")
print(f"  {response.response}")

print(f"\nConfidence: {response.confidence:.1%}")
print(f"Generation time: {response.generation_time_ms:.2f}ms")

if response.source_citations:
    print(f"\nSource Code Referenced:")
    for citation in response.source_citations:
        print(f"  - {citation['chunk_type'].upper()}: {citation['name']}")
        print(f"    File: {citation['file']}")
        print(f"    Relevance: {citation['relevance']}")

# ============================================================================
# EXAMPLE 9: Model Performance Inspection
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 9: Model Information & Training History")
print("="*80)

model_info = pipeline.orchestrator.get_model_info()

print(f"\nModel Configuration:")
print(f"  Base model: {model_info['config']['model_name']}")
print(f"  Max length: {model_info['config']['max_length']}")
print(f"  Using LoRA: {model_info['config']['use_lora']}")

print(f"\nTraining History:")
print(f"  Latest version: v{model_info['latest_version']}")

for checkpoint in model_info['checkpoint_history']:
    print(f"  Version {checkpoint['version']}: {checkpoint['created_at']}")

# ============================================================================
# EXAMPLE 10: Batch Processing
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 10: Batch Query Processing")
print("="*80)

batch_queries = [
    {
        "question": "What is the main purpose of this microservices project?",
        "expected_answer_type": "high_level_overview"
    },
    {
        "question": "List all the REST API endpoints",
        "expected_answer_type": "api_documentation"
    },
    {
        "question": "How is authentication implemented?",
        "expected_answer_type": "implementation_detail"
    },
]

print(f"\nProcessing {len(batch_queries)} queries...\n")

results = []
for i, query_item in enumerate(batch_queries, 1):
    question = query_item['question']
    print(f"[{i}/{len(batch_queries)}] {question}")

    response = pipeline.query_code(question)
    results.append({
        "question": question,
        "answer": response.response,
        "confidence": response.confidence,
        "sources": len(response.source_citations)
    })

print(f"\n✓ Batch processing complete!")
print(f"  Average confidence: {sum(r['confidence'] for r in results)/len(results):.1%}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "#"*80)
print("# EXAMPLES COMPLETE")
print("#"*80)

print("""
You've seen how to:
1. ✓ Setup projects and run the complete pipeline
2. ✓ Query your code with confidence scores
3. ✓ Analyze project structure
4. ✓ Inspect code chunks and training data
5. ✓ Implement continuous learning on code changes
6. ✓ Manage multiple projects
7. ✓ Use RAG for better answers with citations
8. ✓ Check model performance and history
9. ✓ Batch process queries
10. ✓ Review results

Next steps:
- Modify hyperparameters for better accuracy
- Add support for new file types (Python, JavaScript, etc.)
- Build a web interface for querying
- Deploy model as a service
- Integrate with CI/CD pipeline for auto-training

Happy coding! 🚀
""")
