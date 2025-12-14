# fine_tuning.py

from typing import Dict, Any, List


def json_to_narrative(data: Dict[str, Any], include_output: bool = False) -> str:
    """
    Converte i dati strutturati di una degustazione in un testo narrativo.
    
    Args:
        data: Dizionario con i dati della degustazione
        include_output: Se True, include anche scheda_giornalistica e abbinamento
    
    Returns:
        Testo narrativo formattato
    """
    parts = []
    
    # Informazioni base
    info_base = data.get("informazioni_base", {})
    if info_base:
        nome = info_base.get("nome_vino", "")
        produttore = info_base.get("produttore", "")
        denominazione = info_base.get("Denominazione", "")
        anno = info_base.get("anno", "")
        colore = info_base.get("colore", "")
        vitigni = info_base.get("vitigni", [])
        
        if nome:
            parts.append(f"Vino: {nome}")
        if produttore:
            parts.append(f"Produttore: {produttore}")
        if denominazione:
            parts.append(f"Denominazione: {denominazione}")
        if anno:
            parts.append(f"Anno: {anno}")
        if colore:
            parts.append(f"Colore: {colore}")
        if vitigni:
            vitigni_str = ", ".join([f"{v.get('nome', '')} {v.get('percentuale', '')}%" for v in vitigni])
            parts.append(f"Vitigni: {vitigni_str}")
    
    # Scheda analitico-descrittiva
    scheda_analitica = data.get("scheda_analitico_descrittiva", {})
    if scheda_analitica:
        # Esame visivo
        esame_visivo = scheda_analitica.get("esame_visivo", {})
        if esame_visivo:
            vis_parts = []
            if esame_visivo.get("limpidezza"):
                vis_parts.append(f"Limpidezza: {esame_visivo['limpidezza']}")
            colore_b = esame_visivo.get("colore_bianchi") or esame_visivo.get("colore_rose") or esame_visivo.get("colore_rossi")
            if colore_b:
                vis_parts.append(f"Colore: {colore_b}")
            if esame_visivo.get("consistenza"):
                vis_parts.append(f"Consistenza: {esame_visivo['consistenza']}")
            if vis_parts:
                parts.append("Esame visivo: " + ", ".join(vis_parts))
        
        # Esame olfattivo
        esame_olfattivo = scheda_analitica.get("esame_olfattivo", {})
        if esame_olfattivo:
            olf_parts = []
            if esame_olfattivo.get("intensita_olfattiva"):
                olf_parts.append(f"Intensità: {esame_olfattivo['intensita_olfattiva']}")
            if esame_olfattivo.get("complessita_olfattiva"):
                olf_parts.append(f"Complessità: {esame_olfattivo['complessita_olfattiva']}")
            if esame_olfattivo.get("qualita_olfattiva"):
                olf_parts.append(f"Qualità: {esame_olfattivo['qualita_olfattiva']}")
            
            descrittori = esame_olfattivo.get("descrittori", {})
            desc_list = []
            for categoria, valori in descrittori.items():
                if valori and isinstance(valori, list) and len(valori) > 0:
                    desc_list.append(f"{categoria}: {', '.join(valori)}")
            if desc_list:
                olf_parts.append("Descrittori: " + "; ".join(desc_list))
            
            if olf_parts:
                parts.append("Esame olfattivo: " + " | ".join(olf_parts))
        
        # Esame gusto-olfattivo
        esame_gusto = scheda_analitica.get("esame_gusto_olfattivo", {})
        if esame_gusto:
            gusto_parts = []
            if esame_gusto.get("dolcezza"):
                gusto_parts.append(f"Dolcezza: {esame_gusto['dolcezza']}")
            if esame_gusto.get("acidita"):
                gusto_parts.append(f"Acidità: {esame_gusto['acidita']}")
            if esame_gusto.get("alcolicita"):
                gusto_parts.append(f"Alcolicità: {esame_gusto['alcolicita']}")
            if esame_gusto.get("tannicita"):
                gusto_parts.append(f"Tannicità: {esame_gusto['tannicita']}")
            if esame_gusto.get("rotondita"):
                gusto_parts.append(f"Rotondità: {esame_gusto['rotondita']}")
            if esame_gusto.get("sapidita"):
                gusto_parts.append(f"Sapidità: {esame_gusto['sapidita']}")
            if esame_gusto.get("intensita_gusto"):
                gusto_parts.append(f"Intensità: {esame_gusto['intensita_gusto']}")
            if esame_gusto.get("struttura"):
                gusto_parts.append(f"Struttura: {esame_gusto['struttura']}")
            if esame_gusto.get("equilibrio"):
                gusto_parts.append(f"Equilibrio: {esame_gusto['equilibrio']}")
            if esame_gusto.get("persistenza"):
                gusto_parts.append(f"Persistenza: {esame_gusto['persistenza']}")
            if esame_gusto.get("qualita_gusto_olfattiva"):
                gusto_parts.append(f"Qualità: {esame_gusto['qualita_gusto_olfattiva']}")
            
            if gusto_parts:
                parts.append("Esame gusto-olfattivo: " + " | ".join(gusto_parts))
        
        # Considerazioni finali
        considerazioni = scheda_analitica.get("considerazioni_finali", {})
        if considerazioni:
            cons_parts = []
            if considerazioni.get("stato_evolutivo"):
                cons_parts.append(f"Stato evolutivo: {considerazioni['stato_evolutivo']}")
            if considerazioni.get("armonia"):
                cons_parts.append(f"Armonia: {considerazioni['armonia']}")
            if considerazioni.get("qualita_complessiva"):
                cons_parts.append(f"Qualità complessiva: {considerazioni['qualita_complessiva']}")
            if cons_parts:
                parts.append("Considerazioni finali: " + " | ".join(cons_parts))
    
    # Scheda valutazione punti (opzionale)
    scheda_punti = data.get("scheda_valutazione_punti", {})
    if scheda_punti and include_output:
        punteggio = scheda_punti.get("punteggio_totale", "")
        if punteggio:
            parts.append(f"Punteggio totale: {punteggio}/100")
    
    # Output (solo se richiesto)
    if include_output:
        scheda_giornalistica = data.get("scheda_giornalistica", "")
        abbinamento = data.get("abbinamento", "")
        if scheda_giornalistica:
            parts.append(f"Scheda giornalistica: {scheda_giornalistica}")
        if abbinamento:
            parts.append(f"Abbinamento: {abbinamento}")
    
    return "\n".join(parts)

