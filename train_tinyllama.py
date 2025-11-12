#!/usr/bin/env python3
"""
Train TinyLlama on Python Q&A Dataset
Optimized for CPU training with memory-efficient techniques
"""

import os
import csv
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import warnings
from pathlib import Path
from tqdm import tqdm
import logging

# Setup
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PythonQADataset(Dataset):
    """Dataset for Python Q&A pairs"""
    
    def __init__(self, csv_file, tokenizer, max_length=512, use_chat_template=True):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        
        logger.info(f"Loading dataset from {csv_file}")
        
        # Load and parse CSV
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):
                try:
                    # Get question and answer
                    question = row.get('question', row.get('Question', ''))
                    answer = row.get('answer', row.get('Answer', ''))
                    
                    if not question or not answer:
                        continue
                    
                    # Parse answer (it's a JSON list)
                    try:
                        answer_list = json.loads(answer)
                        if isinstance(answer_list, list):
                            answer_text = ' '.join(str(item) for item in answer_list)
                        else:
                            answer_text = str(answer_list)
                    except:
                        answer_text = str(answer)
                    
                    # Format for training
                    if self.use_chat_template:
                        # TinyLlama chat format
                        text = f"<|system|>\nYou are a helpful programming assistant specializing in Python.</s>\n<|user|>\n{question}</s>\n<|assistant|>\n{answer_text}</s>"
                    else:
                        # Simple Q&A format
                        text = f"Question: {question}\nAnswer: {answer_text}"
                    
                    self.data.append(text)
                    
                except Exception as e:
                    logger.warning(f"Error parsing row {row_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.data)} training samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Labels are the same as input_ids for causal LM
        labels = input_ids.clone()
        
        # Replace padding token id's in labels with -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class TinyLlamaTrainer:
    """Trainer for TinyLlama with LoRA"""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize trainer with TinyLlama"""
        logger.info(f"Loading {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        
        logger.info("Model loaded successfully")
    
    def prepare_model_for_training(self):
        """Apply LoRA for efficient training"""
        logger.info("Applying LoRA configuration...")
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # rank
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        return self.model
    
    def train(self, train_dataset, output_dir, epochs=3, batch_size=1, learning_rate=2e-4):
        """Train the model"""
        logger.info("Starting training...")
        
        # Training arguments optimized for CPU
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,  # Simulate larger batch
            warmup_steps=100,
            learning_rate=learning_rate,
            fp16=False,  # No mixed precision on CPU
            fp16_full_eval=False,
            logging_steps=10,
            save_steps=500,
            eval_strategy="no",
            save_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=False,
            push_to_hub=False,
            report_to="none",
            remove_unused_columns=False,
            use_cpu=True,
            dataloader_num_workers=2,
            gradient_checkpointing=False,  # Disabled for CPU
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        # Train
        logger.info("Training started...")
        trainer.train()
        
        # Save model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save LoRA weights separately
        lora_output = os.path.join(output_dir, "adapter_model")
        os.makedirs(lora_output, exist_ok=True)
        self.model.save_pretrained(lora_output)
        
        logger.info("Training completed!")


def merge_lora_weights(base_model_name, lora_model_path, output_path):
    """Merge LoRA weights with base model for easier inference"""
    logger.info("Merging LoRA weights with base model...")
    
    from peft import PeftModel
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    
    # Merge and unload
    model = model.merge_and_unload()
    
    # Save merged model
    model.save_pretrained(output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_path)
    
    logger.info(f"Merged model saved to {output_path}")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune TinyLlama on Python Q&A')
    parser.add_argument('--dataset', default='~/tinyllama/data/dataset_clean.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--output-dir', default='~/tinyllama/models/tinyllama-python-qa',
                        help='Output directory for fine-tuned model')
    parser.add_argument('--base-model', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='Base model name or path')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (keep small for CPU)')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--merge-weights', action='store_true',
                        help='Merge LoRA weights after training')
    parser.add_argument('--test-prompts', action='store_true',
                        help='Test model with sample prompts after training')
    
    args = parser.parse_args()
    
    # Expand paths
    dataset_path = os.path.expanduser(args.dataset)
    output_dir = os.path.expanduser(args.output_dir)
    
    # Check dataset exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 50)
    logger.info("TinyLlama Fine-tuning for Python Q&A")
    logger.info("=" * 50)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info("=" * 50)
    
    # Initialize trainer
    trainer = TinyLlamaTrainer(args.base_model)
    
    # Prepare model with LoRA
    trainer.prepare_model_for_training()
    
    # Load dataset
    train_dataset = PythonQADataset(
        dataset_path,
        trainer.tokenizer,
        max_length=args.max_length
    )
    
    # Train
    trainer.train(
        train_dataset,
        output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Merge weights if requested
    if args.merge_weights:
        lora_path = os.path.join(output_dir, "adapter_model")
        merged_path = output_dir + "-merged"
        
        if os.path.exists(lora_path):
            merge_lora_weights(args.base_model, lora_path, merged_path)
            logger.info(f"Merged model saved to {merged_path}")
    
    # Test the model
    if args.test_prompts:
        logger.info("\n" + "=" * 50)
        logger.info("Testing fine-tuned model...")
        
        test_prompts = [
            "What is a Python list?",
            "How do I read a file in Python?",
            "Explain decorators",
            "What's the difference between append and extend?"
        ]
        
        # Load the fine-tuned model
        from peft import PeftModel
        
        tokenizer = trainer.tokenizer
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        
        lora_path = os.path.join(output_dir, "adapter_model")
        if os.path.exists(lora_path):
            model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            model = trainer.model
        
        model.eval()
        
        for prompt in test_prompts:
            logger.info(f"\nQ: {prompt}")
            
            # Format prompt
            chat_prompt = f"<|system|>\nYou are a helpful programming assistant specializing in Python.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
            
            # Generate
            inputs = tokenizer.encode(chat_prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("<|assistant|>")[-1].strip()
            
            logger.info(f"A: {answer[:200]}...")
    
    logger.info("\n✅ Fine-tuning completed!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("\nTo use the fine-tuned model:")
    logger.info(f"python ~/tinyllama/tinyllama_chat.py --model-path {output_dir}")


if __name__ == "__main__":
    main()
