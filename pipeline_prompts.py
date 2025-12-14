# pipeline_prompts.py

def build_prompt(input_text: str) -> str:
    """
    Prompt identico per training e inferenza.
    Delimitatori semplici (Phi-2 regge bene), niente token speciali tipo <|endoftext|>.
    """
    return (
        "### INPUT\n"
        f"{input_text.strip()}\n"
        "\n### ISTRUZIONI\n"
        "- Scrivi una scheda giornalistica in italiano, 2-4 frasi, tono professionale.\n"
        "- Suggerisci un abbinamento (un piatto o una tipologia di piatto), massimo 10 parole.\n"
        "- Non inventare fatti non presenti nell'input.\n"
        "- Rispondi SOLO con un JSON valido.\n"
        "\n### OUTPUT (JSON)\n"
    )


def build_target_json(scheda: str, abbinamento: str) -> str:
    # Output deterministico e facile da parsare
    scheda = (scheda or "").strip().replace("\n", " ")
    abbinamento = (abbinamento or "").strip().replace("\n", " ")
    return (
        "{"
        f"\"scheda_giornalistica\": \"{escape_json_str(scheda)}\", "
        f"\"abbinamento\": \"{escape_json_str(abbinamento)}\""
        "}"
    )


def escape_json_str(s: str) -> str:
    # Escape minimo per non rompere il JSON
    return (
        s.replace("\\", "\\\\")
         .replace("\"", "\\\"")
         .replace("\t", " ")
         .replace("\r", " ")
    )
