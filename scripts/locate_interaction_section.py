import argparse
import csv
import re
import sqlite3
 
# --- Vocabulário heurístico (ajustável) -------------------------------
 
ANT_TERMS = [
    r"\bformicidae\b", r"\bant\b", r"\bants\b",
    r"\batta\b", r"\bacromyrmex\b", r"\bcamponotus\b", r"\bformica\b",
    r"\bpolyergus\b", r"\bsolenopsis\b", r"\blasius\b",
    r"\boecophylla\b", r"\bazteca\b", r"\bmyrmecophil\w*",
]
 
MICROBE_TERMS = [
    r"\bfungus\b", r"\bfungi\b", r"\bfungal\b",
    r"\bbacteri\w*", r"\bmicrob\w*",
    r"\bescovopsis\b", r"\bophiocordyceps\b", r"\bleucoagaricus\b",
    r"\bmetarhizium\b", r"\bbeauveria\b", r"\bwolbachia\b",
    r"\bpseudonocardia\b", r"\bsymbiont\w*", r"\bpathogen\w*",
    r"\bmicrobiome\b", r"\bmicrobiota\b",
]
 
INTERACTION_TERMS = [
    r"\bassociat\w*", r"\bsymbios\w*", r"\bmutualis\w*",
    r"\bparasit\w*", r"\bpathogenic\w*", r"\binfect\w*",
    r"\bisolat\w*\s+from\b", r"\bcultivat\w*", r"\binocula\w*",
    r"\bcoloniz\w*", r"\bhost[- ]specific\b", r"\bvirulen\w*",
]
 
SECTION_HEADER_RE = re.compile(r'(?m)^([A-Z][A-Z0-9 \-:]{2,60})\n\n')
CONTEXT_WINDOW = 220  # caracteres antes/depois da primeira ocorrência
 
 
def compile_group(terms):
    return re.compile("|".join(terms), re.IGNORECASE)
 
 
ANT_RE = compile_group(ANT_TERMS)
MICROBE_RE = compile_group(MICROBE_TERMS)
INTERACTION_RE = compile_group(INTERACTION_TERMS)
 
 
def split_sections(content):
    """Divide o content em (nome_da_secao, texto) usando o padrão de
    cabeçalho gravado pelo extract_pmc.py. Se nenhum cabeçalho for
    encontrado, retorna uma única seção 'FULL_TEXT'."""
    matches = list(SECTION_HEADER_RE.finditer(content))
    if not matches:
        return [("FULL_TEXT", content)]
 
    sections = []
    for i, m in enumerate(matches):
        name = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        text = content[start:end].strip()
        if text:
            sections.append((name, text))
    return sections
 
 
def score_section(text):
    ant_hits = len(ANT_RE.findall(text))
    microbe_hits = len(MICROBE_RE.findall(text))
    interaction_hits = len(INTERACTION_RE.findall(text))
    # Co-ocorrência pesa mais que menções isoladas
    score = min(ant_hits, microbe_hits) * 2 + interaction_hits
    return ant_hits, microbe_hits, interaction_hits, score
 
 
SENTENCE_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+')
 
 
def closest_pair(ant_positions, microbe_positions):
    """Encontra o par (posição de termo de formiga, posição de termo de
    microorganismo) com a menor distância entre si no texto."""
    best_pair = None
    best_dist = None
    for a in ant_positions:
        for m in microbe_positions:
            dist = abs(a - m)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_pair = (a, m)
    return best_pair
 
 
def expand_to_sentences(text, start, end):
    """Expande [start, end) até os limites de frase mais próximos, sem
    ultrapassar CONTEXT_WINDOW de folga extra em cada lado."""
    left_limit = max(0, start - CONTEXT_WINDOW)
    right_limit = min(len(text), end + CONTEXT_WINDOW)
 
    # anda para trás até achar um limite de frase ('. ', '! ', '? ')
    # ou até left_limit
    seg_before = text[left_limit:start]
    boundaries = list(SENTENCE_BOUNDARY_RE.finditer(seg_before))
    if boundaries:
        new_start = left_limit + boundaries[-1].end()
    else:
        new_start = left_limit
 
    seg_after = text[end:right_limit]
    boundaries = list(SENTENCE_BOUNDARY_RE.finditer(seg_after))
    if boundaries:
        new_end = end + boundaries[0].start()
    else:
        new_end = right_limit
 
    return new_start, new_end
 
 
def best_context(text):
    """Localiza o trecho onde termo de formiga e termo de microorganismo
    aparecem mais próximos um do outro (a co-ocorrência mais provável de
    descrever a relação), e retorna a(s) frase(s) ao redor, com o maior
    peso possível dado também à presença de um termo de interação entre
    eles."""
    ant_positions = [m.start() for m in ANT_RE.finditer(text)]
    microbe_positions = [m.start() for m in MICROBE_RE.finditer(text)]
 
    if not ant_positions and not microbe_positions:
        return text[:2 * CONTEXT_WINDOW].replace("\n", " ").strip()
    if not ant_positions or not microbe_positions:
        pos = (ant_positions or microbe_positions)[0]
        start, end = expand_to_sentences(text, pos, pos)
        return text[start:end].replace("\n", " ").strip()
 
    # Entre os pares mais próximos, prefere aquele que também tem um
    # termo de interação no meio, se existir mais de um candidato próximo
    pairs = []
    for a in ant_positions:
        for m in microbe_positions:
            pairs.append((abs(a - m), a, m))
    pairs.sort(key=lambda p: p[0])
 
    top_candidates = [p for p in pairs if p[0] <= pairs[0][0] + 150] or pairs[:1]
 
    def has_interaction_between(a, m):
        lo, hi = min(a, m), max(a, m)
        return bool(INTERACTION_RE.search(text[lo:hi]))
 
    chosen = next((p for p in top_candidates if has_interaction_between(p[1], p[2])),
                  top_candidates[0])
 
    _, a_pos, m_pos = chosen
    lo, hi = min(a_pos, m_pos), max(a_pos, m_pos)
    start, end = expand_to_sentences(text, lo, hi)
    snippet = text[start:end].replace("\n", " ").strip()
    return snippet
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Localiza a seção com a relação ecológica formiga-microorganismo")
    parser.add_argument('--database', dest='db', required=True)
    parser.add_argument('--output', dest='output', required=True,
                        help='CSV de saída')
    parser.add_argument('--top_n', dest='top_n', type=int, default=2,
                        help='Quantas seções (melhor pontuadas) reportar por artigo')
    args = parser.parse_args()
 
    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    rows = cur.execute(
        'SELECT pmid, title, doi, content FROM pcw_literature').fetchall()
 
    results = []
    for pmid, title, doi, content in rows:
        if not content:
            results.append({
                'pmid': pmid, 'title': title, 'doi': doi,
                'section': '', 'ant_hits': 0, 'microbe_hits': 0,
                'interaction_hits': 0, 'score': 0, 'snippet': '(sem content)'
            })
            continue
 
        sections = split_sections(content)
        scored = []
        for name, text in sections:
            ant_hits, microbe_hits, interaction_hits, score = score_section(text)
            scored.append((name, text, ant_hits, microbe_hits, interaction_hits, score))
 
        scored.sort(key=lambda x: x[-1], reverse=True)
 
        top = [s for s in scored if s[-1] > 0][:args.top_n]
        if not top:
            results.append({
                'pmid': pmid, 'title': title, 'doi': doi,
                'section': '(nenhuma seção com co-ocorrência)', 'ant_hits': 0,
                'microbe_hits': 0, 'interaction_hits': 0, 'score': 0,
                'snippet': ''
            })
            continue
 
        for name, text, ant_hits, microbe_hits, interaction_hits, score in top:
            results.append({
                'pmid': pmid, 'title': title, 'doi': doi,
                'section': name, 'ant_hits': ant_hits,
                'microbe_hits': microbe_hits,
                'interaction_hits': interaction_hits, 'score': score,
                'snippet': best_context(text)
            })
 
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pmid', 'title', 'doi', 'section', 'ant_hits', 'microbe_hits',
            'interaction_hits', 'score', 'snippet'])
        writer.writeheader()
        writer.writerows(results)
 
    print(f"{len(results)} linhas gravadas em {args.output}")
 
 
if __name__ == '__main__':
    main()