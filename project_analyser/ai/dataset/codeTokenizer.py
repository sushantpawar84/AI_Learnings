"""
Code-Specific Tokenizer
Handles tokenization with awareness of code syntax and structure.
Implements dynamic batching and context-aware chunking.
"""

import json
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, PreTrainedTokenizer
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class TokenizedPair:
    """Represents a tokenized training pair"""
    pair_id: str
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    token_count: int
    original_length: int


class CodeTokenizer:
    """Specialized tokenizer for code with syntax awareness"""

    def __init__(self, model_name: str = "gpt2", max_length: int = 1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.code_delimiters = {
            "java": ["```java", "```", "{", "}", "[CODE]", "[/CODE]"],
            "generic": ["```", "```", "[CODE]", "[/CODE]"]
        }

    def tokenize_pair(self, input_text: str, output_text: str,
                     code_type: str = "java", truncate: bool = True) -> TokenizedPair:
        """Tokenize a training pair with syntax awareness"""

        # Prepare combined text with special formatting
        combined_text = f"{input_text}\n\nEXPLANATION:\n{output_text}"

        # Tokenize input and output separately for proper loss computation
        input_tokens = self.tokenizer(
            input_text,
            truncation=truncate,
            max_length=self.max_length,
            add_special_tokens=True
        )

        output_tokens = self.tokenizer(
            output_text,
            truncation=truncate,
            max_length=self.max_length,
            add_special_tokens=True
        )

        # Tokenize combined text for full context
        combined_tokens = self.tokenizer(
            combined_text,
            truncation=truncate,
            max_length=self.max_length,
            add_special_tokens=True
        )

        # Create labels - only predict on output portion
        input_length = len(input_tokens["input_ids"])
        labels = [-100] * input_length + combined_tokens["input_ids"][input_length:]

        # Pad or truncate to max_length
        if len(combined_tokens["input_ids"]) < self.max_length:
            pad_length = self.max_length - len(combined_tokens["input_ids"])
            combined_tokens["input_ids"] += [self.tokenizer.pad_token_id] * pad_length
            combined_tokens["attention_mask"] += [0] * pad_length
            labels += [-100] * pad_length
        else:
            combined_tokens["input_ids"] = combined_tokens["input_ids"][:self.max_length]
            combined_tokens["attention_mask"] = combined_tokens["attention_mask"][:self.max_length]
            labels = labels[:self.max_length]

        return TokenizedPair(
            pair_id="",  # Will be set by caller
            input_ids=combined_tokens["input_ids"],
            attention_mask=combined_tokens["attention_mask"],
            labels=labels,
            token_count=len([x for x in combined_tokens["attention_mask"] if x == 1]),
            original_length=len(combined_text)
        )

    def tokenize_pairs_batch(self, pairs: List[Dict[str, str]],
                            code_type: str = "java") -> List[TokenizedPair]:
        """Tokenize a batch of pairs"""
        tokenized = []
        for pair in pairs:
            try:
                tok_pair = self.tokenize_pair(
                    pair["input_text"],
                    pair["output_text"],
                    code_type=code_type
                )
                tok_pair.pair_id = pair.get("pair_id", "")
                tokenized.append(tok_pair)
            except Exception as e:
                print(f"Error tokenizing pair: {e}")
                continue
        return tokenized

    def get_max_length(self) -> int:
        """Get configured maximum sequence length"""
        return self.max_length

    def get_vocab_size(self) -> int:
        """Get tokenizer vocabulary size"""
        return self.tokenizer.vocab_size


class DynamicBatcher:
    """Handles dynamic batching for variable-length sequences"""

    def __init__(self, max_tokens: int = 2048, max_sequences: int = 16):
        self.max_tokens = max_tokens  # Total tokens per batch
        self.max_sequences = max_sequences  # Max sequences per batch

    def create_batches(self, tokenized_pairs: List[TokenizedPair],
                      sort_by_length: bool = True) -> List[List[TokenizedPair]]:
        """Create dynamic batches based on token count"""

        # Sort by length for efficient packing
        if sort_by_length:
            pairs = sorted(tokenized_pairs, key=lambda x: x.token_count, reverse=True)
        else:
            pairs = tokenized_pairs

        batches = []
        current_batch = []
        current_tokens = 0

        for pair in pairs:
            # Check if adding this pair exceeds limits
            if (current_batch and
                (len(current_batch) >= self.max_sequences or
                 current_tokens + pair.token_count > self.max_tokens)):
                # Save current batch and start new one
                batches.append(current_batch)
                current_batch = [pair]
                current_tokens = pair.token_count
            else:
                current_batch.append(pair)
                current_tokens += pair.token_count

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def collate_batch(self, batch: List[TokenizedPair],
                     device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Convert batch to tensors for model input"""

        input_ids = torch.tensor([p.input_ids for p in batch], dtype=torch.long)
        attention_mask = torch.tensor([p.attention_mask for p in batch], dtype=torch.long)
        labels = torch.tensor([p.labels for p in batch], dtype=torch.long)

        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device)
        }


class CodeDatasetTokenizer:
    """Main class orchestrating tokenization of code datasets"""

    def __init__(self, model_name: str = "gpt2", max_length: int = 1024,
                 output_dir: str = None):
        self.tokenizer = CodeTokenizer(model_name, max_length)
        self.batcher = DynamicBatcher()
        self.output_dir = output_dir
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    def tokenize_dataset_file(self, input_file: str, output_file: str = None,
                             code_type: str = "java") -> Tuple[List[TokenizedPair], Dict[str, Any]]:
        """Tokenize a JSONL dataset file"""

        pairs = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        pair_dict = json.loads(line)
                        pairs.append(pair_dict)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading input file: {e}")
            return [], {}

        print(f"Loaded {len(pairs)} pairs from {input_file}")

        # Tokenize
        print(f"Tokenizing {len(pairs)} pairs...")
        tokenized = self.tokenizer.tokenize_pairs_batch(pairs, code_type)

        # Calculate statistics
        stats = {
            "total_pairs": len(tokenized),
            "total_tokens": sum(p.token_count for p in tokenized),
            "avg_tokens_per_pair": sum(p.token_count for p in tokenized) / len(tokenized) if tokenized else 0,
            "max_tokens": max(p.token_count for p in tokenized) if tokenized else 0,
            "min_tokens": min(p.token_count for p in tokenized) if tokenized else 0,
        }

        # Save if output file specified
        if output_file:
            self._save_tokenized(tokenized, output_file)

        return tokenized, stats

    def create_batches(self, tokenized_pairs: List[TokenizedPair]) -> List[List[TokenizedPair]]:
        """Create dynamic batches"""
        return self.batcher.create_batches(tokenized_pairs)

    def _save_tokenized(self, tokenized: List[TokenizedPair], output_file: str):
        """Save tokenized pairs to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for tok_pair in tokenized:
                    f.write(json.dumps(asdict(tok_pair)) + '\n')
            print(f"Saved {len(tokenized)} tokenized pairs to {output_file}")
        except Exception as e:
            print(f"Error saving tokenized pairs: {e}")


class TokenizationPipeline:
    """End-to-end tokenization pipeline"""

    def __init__(self, model_name: str = "gpt2", max_length: int = 1024):
        self.dataset_tokenizer = CodeDatasetTokenizer(model_name, max_length)
        self.batcher = DynamicBatcher()

    def process_dataset(self, input_file: str, output_file: str,
                       code_type: str = "java") -> Dict[str, Any]:
        """Process entire dataset through tokenization pipeline"""

        print("=" * 60)
        print("CODE TOKENIZATION PIPELINE")
        print("=" * 60)

        # Tokenize
        tokenized, stats = self.dataset_tokenizer.tokenize_dataset_file(
            input_file, output_file, code_type
        )

        # Create batches
        batches = self.batcher.create_batches(tokenized)

        # Add batch statistics
        stats["total_batches"] = len(batches)
        stats["avg_batch_size"] = len(tokenized) / len(batches) if batches else 0
        batch_sizes = [len(b) for b in batches]
        stats["min_batch_size"] = min(batch_sizes) if batch_sizes else 0
        stats["max_batch_size"] = max(batch_sizes) if batch_sizes else 0

        print("\n" + "=" * 60)
        print("TOKENIZATION STATISTICS")
        print("=" * 60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

        return stats


if __name__ == "__main__":
    # Test tokenization
    pipeline = TokenizationPipeline(max_length=1024)

    input_file = "C:/Sushant/AI Learning/Datasets/microservices/versions/microservices_project_v1.jsonl"
    output_file = "C:/Sushant/AI Learning/Datasets/microservices/tokenized_v1.jsonl"

    stats = pipeline.process_dataset(input_file, output_file)

