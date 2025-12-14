# pipeline_train_lora.py

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from pipeline_dataset import build_hf_dataset, tokenize_and_mask


class MetricsCallback(TrainerCallback):
    """Callback per salvare le metriche durante il training."""
    
    def __init__(self, metrics_file="training_metrics.json"):
        self.metrics_file = metrics_file
        self.metrics = {
            "train_loss": [],
            "learning_rate": [],
            "epoch": [],
            "step": []
        }
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Salva le metriche ad ogni log."""
        if logs is None:
            return
        
        if "loss" in logs:
            self.metrics["train_loss"].append(logs["loss"])
            self.metrics["step"].append(state.global_step)
            self.metrics["epoch"].append(state.epoch)
        
        if "learning_rate" in logs:
            self.metrics["learning_rate"].append(logs["learning_rate"])
        
        # Salva le metriche in tempo reale
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
    
    def on_train_end(self, args, state, control, **kwargs):
        """Salva le metriche finali."""
        with open(self.metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Metriche salvate in {self.metrics_file}")


def main():
    # Parametri (puoi leggerli da config.py, ma qui li metto espliciti)
    MODEL_NAME = "microsoft/phi-2"   # come nel tuo config :contentReference[oaicite:7]{index=7}
    JSON_PATH = "degustazioni.json"  # come nel tuo config :contentReference[oaicite:8]{index=8}
    OUTPUT_DIR = "./fine_tuned_lora"
    MAX_LENGTH = 768
    BATCH_SIZE = 1
    EPOCHS = 5
    LR = 2e-4

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Carico base model in FP16 se GPU disponibile (più stabile e leggero)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # LoRA: target_modules robusto. "all-linear" è la scelta più resiliente.
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    dataset = build_hf_dataset(JSON_PATH)

    def tok_map(batch):
        return tokenize_and_mask(batch, tokenizer=tokenizer, max_length=MAX_LENGTH)

    tokenized = dataset.map(tok_map, batched=True, remove_columns=dataset.column_names)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        return_tensors="pt",
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=LR,
        warmup_ratio=0.05,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        bf16=False,
        dataloader_pin_memory=False,
    )

    # Callback per salvare le metriche
    metrics_callback = MetricsCallback(metrics_file=os.path.join(OUTPUT_DIR, "training_metrics.json"))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[metrics_callback],
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"OK: LoRA salvata in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
