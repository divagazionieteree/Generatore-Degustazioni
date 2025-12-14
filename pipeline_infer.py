# pipeline_infer.py

import json
import re
from typing import Dict, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from pipeline_prompts import build_prompt
from fine_tuning import json_to_narrative  # come ora :contentReference[oaicite:10]{index=10}


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> Dict[str, Any]:
    """
    Estrae il primo blocco {...} e tenta json.loads.
    Gestisce anche JSON malformati con tentativi di pulizia.
    """
    # Prima prova con il pattern standard
    m = JSON_RE.search(text)
    if not m:
        # Prova a trovare qualsiasi blocco che inizia con { e contiene le chiavi che ci servono
        alt_pattern = re.compile(r'\{[^{}]*(?:scheda_giornalistica|abbinamento)[^{}]*\}', re.DOTALL)
        m = alt_pattern.search(text)
        if not m:
            raise ValueError(f"Nessun JSON trovato nell'output. Testo: {text[:300]}")
    
    raw = m.group(0)
    
    # Prova a parsare
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Prova a pulire il JSON rimuovendo caratteri problematici
        # Rimuove eventuali caratteri non validi prima della {
        cleaned = raw.strip()
        if not cleaned.startswith('{'):
            # Trova la prima {
            idx = cleaned.find('{')
            if idx >= 0:
                cleaned = cleaned[idx:]
        
        # Prova a rimuovere eventuali virgolette non bilanciate o caratteri strani alla fine
        while cleaned and not cleaned.endswith('}'):
            cleaned = cleaned[:-1]
        
        try:
            return json.loads(cleaned)
        except:
            raise ValueError(f"Impossibile parsare JSON. Errore: {e}. Testo: {raw[:500]}")


def load_model(base_model_name: str, lora_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    return tokenizer, model


def generate_for_degustazione(deg: Dict[str, Any], tokenizer, model,
                             max_input_tokens: int = 800,
                             max_new_tokens: int = 220,
                             debug: bool = True) -> Tuple[str, str, str]:

    input_data = dict(deg)
    input_data.pop("scheda_giornalistica", None)
    input_data.pop("abbinamento", None)

    input_text = json_to_narrative(input_data, include_output=False)
    prompt = build_prompt(input_text)

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.25,
            top_p=0.7,
            top_k=40,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    
    if debug:
        print(f"\n[DEBUG] Testo generato grezzo:\n{gen}\n")

    # Prova a estrarre il JSON
    try:
        data = extract_json(gen)
    except ValueError as e:
        # Se l'estrazione fallisce, prova a pulire il testo
        if debug:
            print(f"[DEBUG] Errore estrazione JSON: {e}")
            print(f"[DEBUG] Tentativo di pulizia...")
        
        # Prova a trovare JSON anche se malformato
        # Cerca pattern più flessibili
        json_patterns = [
            r'\{[^{}]*"scheda_giornalistica"[^{}]*"abbinamento"[^{}]*\}',
            r'\{.*?"scheda_giornalistica".*?"abbinamento".*?\}',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, gen, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                    break
                except:
                    continue
        
        # Se ancora non funziona, solleva l'errore originale
        if 'data' not in locals():
            raise ValueError(f"Impossibile estrarre JSON dal testo generato. Testo: {gen[:200]}...")
    
    scheda = str(data.get("scheda_giornalistica", "")).strip()
    abb = str(data.get("abbinamento", "")).strip()
    
    if debug:
        print(f"[DEBUG] Scheda estratta: {scheda}")
        print(f"[DEBUG] Abbinamento estratto: {abb}")
        print(f"[DEBUG] Lunghezza scheda: {len(scheda)}")

    # Safety minima - più flessibile
    if not scheda:
        raise ValueError(f"Scheda vuota. Testo generato: {gen[:300]}")
    if len(scheda) < 20:  # Ridotto da 40 a 20 per essere meno rigido
        raise ValueError(f"Scheda troppo corta ({len(scheda)} caratteri, minimo 20). Contenuto: {scheda}")
    if not abb:
        raise ValueError(f"Abbinamento vuoto. Testo generato: {gen[:300]}")
    if len(abb) > 100:  # Aumentato da 80 a 100
        raise ValueError(f"Abbinamento troppo lungo ({len(abb)} caratteri, massimo 100). Contenuto: {abb}")

    return scheda, abb, prompt


def main():
    """
    Funzione principale per eseguire l'inferenza su un file JSON di degustazioni.
    """
    # Parametri configurabili
    BASE_MODEL_NAME = "microsoft/phi-2"
    LORA_PATH = "./fine_tuned_lora"  # Path alla LoRA addestrata
    INPUT_JSON = "input.json"  # File di input con le degustazioni
    OUTPUT_JSON = "output_infer.json"  # File di output
    DEBUG = True  # Attiva debug per vedere cosa genera il modello
    
    print(f"Caricamento modello base: {BASE_MODEL_NAME}")
    print(f"Caricamento LoRA da: {LORA_PATH}")
    
    try:
        tokenizer, model = load_model(BASE_MODEL_NAME, LORA_PATH)
        print("✓ Modello caricato con successo")
    except Exception as e:
        print(f"✗ Errore nel caricamento del modello: {e}")
        return
    
    # Carica i dati di input
    print(f"\nCaricamento dati da: {INPUT_JSON}")
    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            degustazioni = json.load(f)
        print(f"✓ Trovate {len(degustazioni)} degustazioni")
    except FileNotFoundError:
        print(f"✗ File {INPUT_JSON} non trovato")
        return
    except Exception as e:
        print(f"✗ Errore nel caricamento del file: {e}")
        return
    
    # Genera le schede per ogni degustazione
    results = []
    print("\nGenerazione schede...")
    
    for i, deg in enumerate(degustazioni, 1):
        deg_id = deg.get("id", str(i))
        print(f"\n[{i}/{len(degustazioni)}] Elaborazione degustazione ID: {deg_id}")
        
        try:
            scheda, abbinamento, prompt = generate_for_degustazione(
                deg, tokenizer, model, debug=DEBUG
            )
            
            # Crea il risultato
            result = {
                "id": deg_id,
                "scheda_giornalistica": scheda,
                "abbinamento": abbinamento,
            }
            results.append(result)
            
            print(f"✓ Scheda generata:")
            print(f"  - Scheda: {scheda[:80]}...")
            print(f"  - Abbinamento: {abbinamento}")
            
        except Exception as e:
            print(f"✗ Errore nella generazione: {e}")
            # Aggiungi comunque un risultato vuoto per tracciare l'errore
            results.append({
                "id": deg_id,
                "scheda_giornalistica": "",
                "abbinamento": "",
                "errore": str(e)
            })
    
    # Salva i risultati
    print(f"\nSalvataggio risultati in: {OUTPUT_JSON}")
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Risultati salvati: {len(results)} schede generate")
    except Exception as e:
        print(f"✗ Errore nel salvataggio: {e}")
    
    # Stampa un riepilogo
    print("\n" + "="*60)
    print("RIEPILOGO")
    print("="*60)
    successful = sum(1 for r in results if r.get("scheda_giornalistica"))
    print(f"Schede generate con successo: {successful}/{len(results)}")
    print(f"File di output: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
