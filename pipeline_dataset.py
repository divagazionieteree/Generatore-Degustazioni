# pipeline_dataset.py

import json
from typing import List, Dict, Any
from datasets import Dataset

from pipeline_prompts import build_prompt, build_target_json

# IMPORTA la tua funzione esistente (già nel tuo fine_tuning.py) :contentReference[oaicite:5]{index=5}
# Se la funzione sta in fine_tuning.py, ti conviene spostarla in un modulo dedicato (es. narrative.py).
from fine_tuning import json_to_narrative  # oppure dal file corretto


def load_degustazioni(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_training_examples(degustazioni: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out = []
    for deg in degustazioni:
        scheda = (deg.get("scheda_giornalistica") or "").strip()
        abb = (deg.get("abbinamento") or "").strip()
        if not scheda and not abb:
            continue

        input_data = dict(deg)
        input_data.pop("scheda_giornalistica", None)
        input_data.pop("abbinamento", None)

        input_text = json_to_narrative(input_data, include_output=False)
        prompt = build_prompt(input_text)
        target = build_target_json(scheda, abb)

        out.append({"prompt": prompt, "target": target})
    return out


def tokenize_and_mask(batch, tokenizer, max_length: int):
    """
    Costruisce: [prompt + target + eos]
    e maschera i token del prompt (labels=-100) così la loss viene calcolata SOLO sul target.
    """
    prompts = batch["prompt"]
    targets = batch["target"]

    input_ids_list = []
    attention_masks = []
    labels_list = []

    eos = tokenizer.eos_token or ""
    for prompt, target in zip(prompts, targets):
        full_text = prompt + target + eos

        enc_full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )
        enc_prompt = tokenizer(
            prompt,
            truncation=True,
            max_length=max_length,
            padding=False,
            add_special_tokens=False,
        )

        input_ids = enc_full["input_ids"]
        attn = enc_full["attention_mask"]

        prompt_len = len(enc_prompt["input_ids"])
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[: len(input_ids)]  # sicurezza

        input_ids_list.append(input_ids)
        attention_masks.append(attn)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "attention_mask": attention_masks, "labels": labels_list}


def build_hf_dataset(json_path: str) -> Dataset:
    degustazioni = load_degustazioni(json_path)
    examples = make_training_examples(degustazioni)
    return Dataset.from_list(examples)
