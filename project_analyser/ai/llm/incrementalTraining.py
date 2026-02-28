"""
Incremental Fine-Tuning Module
Implements efficient fine-tuning using LoRA (Low-Rank Adaptation) and PEFT.
Supports continuous learning with checkpoint management and validation.
"""

import os
import json
import torch
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
from transformers.integrations import TensorBoardCallback
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning"""
    model_name: str = "gpt2"
    output_dir: str = "./models/gpt2-code"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    weight_decay: float = 0.01
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 500
    max_length: int = 1024
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingMetrics:
    """Metrics from training run"""
    epoch: int
    train_loss: float
    eval_loss: float
    learning_rate: float
    timestamp: str
    checkpoint: str


class ModelCheckpointManager:
    """Manages model checkpoints and version control"""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.checkpoints_dir = os.path.join(base_dir, "checkpoints")
        self.metadata_file = os.path.join(base_dir, "checkpoint_metadata.json")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.metadata = self._load_metadata()

    def save_checkpoint(self, model, tokenizer, config: TrainingConfig,
                       metrics: Dict[str, Any], version: int) -> str:
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.checkpoints_dir, f"checkpoint-v{version}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model and tokenizer
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

        # Save config
        config_file = os.path.join(checkpoint_dir, "training_config.json")
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Update metadata
        checkpoint_info = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "path": checkpoint_dir
        }
        self.metadata[f"v{version}"] = checkpoint_info
        self._save_metadata()

        return checkpoint_dir

    def load_checkpoint(self, version: int) -> Tuple[str, Dict[str, Any]]:
        """Load checkpoint info"""
        checkpoint_dir = os.path.join(self.checkpoints_dir, f"checkpoint-v{version}")
        if os.path.exists(checkpoint_dir):
            metadata_key = f"v{version}"
            metadata = self.metadata.get(metadata_key, {})
            return checkpoint_dir, metadata
        return None, {}

    def get_latest_version(self) -> int:
        """Get latest checkpoint version"""
        versions = [int(k.replace('v', '')) for k in self.metadata.keys() if k.startswith('v')]
        return max(versions) if versions else 0

    def get_checkpoint_history(self) -> List[Dict[str, Any]]:
        """Get all checkpoints in chronological order"""
        checkpoints = []
        for key in sorted(self.metadata.keys()):
            checkpoints.append(self.metadata[key])
        return checkpoints

    def _load_metadata(self) -> Dict[str, Any]:
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}

    def _save_metadata(self):
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}")


class CodeModelTrainer:
    """Main trainer for code understanding models"""

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.checkpoint_manager = ModelCheckpointManager(self.config.output_dir)
        self.tokenizer = None
        self.model = None
        self.training_history = []

    def prepare_model(self) -> Tuple[Any, AutoTokenizer]:
        """Load and prepare model for training"""
        print(f"Loading model: {self.config.model_name}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Apply LoRA if configured
        if self.config.use_lora:
            print(f"Applying LoRA configuration...")
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["c_attn"]  # For GPT-2
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        return self.model, self.tokenizer

    def train_on_dataset(self, train_dataset_path: str,
                        eval_dataset_path: str = None,
                        resume_from_checkpoint: bool = False) -> Dict[str, Any]:
        """Train or fine-tune on code dataset"""

        if self.model is None or self.tokenizer is None:
            self.prepare_model()

        print(f"Loading training dataset from {train_dataset_path}")

        # Load dataset
        try:
            train_dataset = load_dataset(
                'json',
                data_files=train_dataset_path,
                split='train'
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return {}

        # Load eval dataset if provided
        eval_dataset = None
        if eval_dataset_path:
            try:
                eval_dataset = load_dataset(
                    'json',
                    data_files=eval_dataset_path,
                    split='train'
                )
            except:
                pass

        # Prepare datasets
        def prepare_data(examples):
            # For tokenized data
            return {
                'input_ids': torch.tensor(examples['input_ids']) if isinstance(examples['input_ids'], list) else examples['input_ids'],
                'attention_mask': torch.tensor(examples['attention_mask']) if isinstance(examples['attention_mask'], list) else examples['attention_mask'],
                'labels': torch.tensor(examples['labels']) if isinstance(examples['labels'], list) else examples['labels']
            }

        train_dataset = train_dataset.map(prepare_data, batched=True, batch_size=100)
        if eval_dataset:
            eval_dataset = eval_dataset.map(prepare_data, batched=True, batch_size=100)

        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        if eval_dataset:
            eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.config.fp16 and torch.cuda.is_available(),
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            report_to=["tensorboard"],
            remove_unused_columns=False
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[TensorBoardCallback()]
        )

        # Train
        print("Starting training...")
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save checkpoint
        version = self.checkpoint_manager.get_latest_version() + 1
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.model, self.tokenizer, self.config,
            result.training_result if hasattr(result, 'training_result') else {},
            version
        )

        metrics = {
            "train_loss": result.training_loss,
            "version": version,
            "checkpoint": checkpoint_path,
            "timestamp": datetime.now().isoformat()
        }
        self.training_history.append(metrics)

        return metrics

    def incremental_train(self, new_dataset_path: str,
                         previous_checkpoint_version: int = None) -> Dict[str, Any]:
        """Perform incremental training on new data"""

        # Load previous checkpoint if available
        if previous_checkpoint_version:
            checkpoint_dir, metadata = self.checkpoint_manager.load_checkpoint(previous_checkpoint_version)
            if checkpoint_dir:
                print(f"Loading checkpoint v{previous_checkpoint_version}")
                self.model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
                self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        else:
            self.prepare_model()

        # Train on new data
        return self.train_on_dataset(new_dataset_path)

    def generate_from_prompt(self, prompt: str, max_length: int = 100) -> str:
        """Generate text from prompt using trained model"""
        if self.model is None or self.tokenizer is None:
            self.prepare_model()

        inputs = self.tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class TrainingOrchestrator:
    """Orchestrates the entire training pipeline"""

    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.trainer = CodeModelTrainer(self.config)

    def train_on_new_project(self, dataset_path: str,
                           eval_dataset_path: str = None) -> Dict[str, Any]:
        """Train model on a new project dataset"""
        print("\n" + "="*60)
        print("TRAINING ON NEW PROJECT")
        print("="*60)

        self.trainer.prepare_model()
        metrics = self.trainer.train_on_dataset(dataset_path, eval_dataset_path)

        return metrics

    def incremental_retrain(self, new_dataset_path: str) -> Dict[str, Any]:
        """Incrementally retrain on new/updated data"""
        print("\n" + "="*60)
        print("INCREMENTAL RETRAINING")
        print("="*60)

        previous_version = self.trainer.checkpoint_manager.get_latest_version()
        metrics = self.trainer.incremental_train(new_dataset_path, previous_version)

        return metrics

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about current model"""
        return {
            "config": self.config.to_dict(),
            "checkpoint_history": self.trainer.checkpoint_manager.get_checkpoint_history(),
            "latest_version": self.trainer.checkpoint_manager.get_latest_version()
        }


if __name__ == "__main__":
    # Example usage
    config = TrainingConfig(
        model_name="gpt2",
        output_dir="./models/code-gpt2",
        num_train_epochs=3,
        per_device_train_batch_size=4
    )

    orchestrator = TrainingOrchestrator(config)

    # Train on initial dataset
    # metrics = orchestrator.train_on_new_project(
    #     "path/to/tokenized/dataset.jsonl"
    # )


