#!/usr/bin/env python3

import argparse
import csv
import re
import sqlite3
import requests
import spacy

__version__ = "2.0.0"

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "Modelo 'en_core_web_sm' do spaCy não encontrado. "
        "Execute no terminal: python3 -m spacy download en_core_web_sm"
    )

taxonomy_cache = {}

GLOBI_RELATION_LEMMAS = {
    "parasiteof": {
        "parasite", "parasitic", "parasitize", "ectoparasite", "endoparasite", 
        "infection", "infect", "attack", "endosymbiont", "found", "host", "isolate",
        "susceptibility", "susceptible", "parasitoid", "antagonist", "harbour", "harbor"
    },
    "hasparasite": {"parasite", "parasitic", "parasitize", "host"},
    "endoparasiteof": {"endoparasite", "parasite", "endosymbiont"},
    "ectoparasiteof": {"ectoparasite", "parasite"},
    "pathogenof": {
        "pathogen", "pathogenic", "infect", "infection", "cause", "disease", 
        "attack", "antagonist", "agent", "collect", "isolate", "endosymbiont", 
        "type", "host", "susceptibility", "susceptible", "biting", "fungus", "die"
    },
    "haspathogen": {"pathogen", "infect", "infection"},
    "hashost": {"host", "infect", "associate", "association", "dwell", "find", "stomach", "gut"},
    "hosts": {"host"},
    "symbiontof": {"symbiont", "symbiosis", "symbiotic", "associate", "culture", "endosymbiont", "fungiculture"},
    "hassymbiont": {"symbiont", "symbiosis", "symbiotic", "culture"},
    "mutualistof": {"mutualist", "mutualism", "mutualistic", "symbiosis"},
    "hasmutualist": {"mutualist", "mutualism"},
    "commensalistof": {"commensal", "commensalism"},
    "hascommensalist": {"commensal"},
    "eats": {"feed", "eat", "consume", "predate", "prey", "harvest", "collect", "ingest", "forage", "fungus-growing"},
    "preyson": {"predate", "prey", "consume", "feed"},
    "eatenby": {"feed", "eat", "consume"},
    "dispersalvectorof": {"dispersal", "vector", "transport", "carry", "dispersion", "act"},
    "hasvector": {"vector", "transmit", "transmission"},
    "pollinates": {"pollinate", "pollination", "pollinator"},
    "pollinatedby": {"pollinate", "pollination"},
    "visitsfloweringplantof": {"visit", "forage"},
    "vectorof": {"vector", "transmit", "transmission"},
    "interactswith": {
        "interact", "interaction", "associate", "association", "challenge", 
        "inhibit", "inhibition", "find", "occur", "detect", "stomach", "gut", "observe", "specific", "harbour"
    },
    "cooccurswith": {"cooccur", "co-occur", "find", "associate", "sympatric", "occur"},
    "isolatedfrom": {"isolate", "isolation", "obtain", "sample", "extract", "detect", "find"},
    "livesinside": {"live", "inhabit", "dwell", "stomach", "gut"},
    "livesnear": {"live", "find"},
    "epiphyteof": {"epiphyte", "epiphytic"},
    "hasepiphyte": {"epiphyte"},
    "farms": {"farm", "cultivate", "fungiculture"},
    "farmedby": {"farm", "cultivate"},
    "kill": {"kill", "destroy", "inhibit", "antagonist", "antagonistic", "die"}
}

DEFAULT_RELATION_LEMMAS = {"associate", "association", "find", "relate", "relation", "interact", "occur"}

TAXON_SYNONYMS_MANUAL = {
    "Ophiocordyceps unilateralis": ["Torrubia unilateralis", "Cordyceps unilateralis"],
    "Ophiocordyceps": ["Cordyceps", "Torrubia", "Hirsutella"],
    "Colobopsis leonardi": ["Camponotus leonardi", "C. leonardi"],
    "Colobopsis saundersi": ["Camponotus saundersi", "C. saundersi"],
    "Colobopsis": ["Camponotus"],
    "Hirsutella stilbelliformis": ["Hirsutella stilbelliformis var. stilbelliformis", "Ophiocordyceps"]
}


def get_relation_lemmas(interaction_type):
    key = re.sub(r'[\s_\-]', '', interaction_type.strip().lower())
    return GLOBI_RELATION_LEMMAS.get(key, DEFAULT_RELATION_LEMMAS)


def get_gbif_taxonomic_hierarchy(taxon_name):
    taxon_name = taxon_name.strip()
    if taxon_name in taxonomy_cache:
        return taxonomy_cache[taxon_name]

    matched_names = set()
    if taxon_name:
        matched_names.add(taxon_name)

    if taxon_name in TAXON_SYNONYMS_MANUAL:
        matched_names.update(TAXON_SYNONYMS_MANUAL[taxon_name])

    try:
        url_match = f"https://api.gbif.org/v1/species/match?name={requests.utils.quote(taxon_name)}"
        resp = requests.get(url_match, timeout=5)

        if resp.status_code == 200:
            data = resp.json()
            usage_key = data.get('usageKey')
            rank = data.get('rank', '').upper()

            if usage_key:
                accepted_name = data.get('acceptedUsage', {}).get('canonicalName') or data.get('species')
                if accepted_name:
                    matched_names.add(accepted_name)

                url_syn = f"https://api.gbif.org/v1/species/{usage_key}/synonyms"
                resp_syn = requests.get(url_syn, timeout=4)
                if resp_syn.status_code == 200:
                    for item in resp_syn.json().get('results', []):
                        if item.get('canonicalName'):
                            matched_names.add(item['canonicalName'])

                if rank in ['FAMILY', 'ORDER', 'SUPERFAMILY', 'SUBFAMILY', 'TRIBE', 'GENUS']:
                    url_children = f"https://api.gbif.org/v1/species/{usage_key}/children?limit=150"
                    resp_child = requests.get(url_children, timeout=5)
                    if resp_child.status_code == 200:
                        for child in resp_child.json().get('results', []):
                            child_name = child.get('canonicalName')
                            if child_name:
                                matched_names.add(child_name)

    except Exception:
        pass

    taxonomy_cache[taxon_name] = matched_names
    return matched_names


def taxon_to_regex(taxon_name):
    all_names = get_gbif_taxonomic_hierarchy(taxon_name)
    patterns = []

    for name in all_names:
        name = name.strip()
        if not name:
            continue

        parts = name.split()
        if len(parts) >= 2:
            genus = re.escape(parts[0])
            species = re.escape(parts[1])
            patterns.append(rf"\b{genus}(?:\s+(?:var\.|subsp\.|s\.l\.|sensu\s+lato|[A-Z][a-z]+))?\s+{species}\b")
            
            initial = re.escape(parts[0][0])
            patterns.append(rf"\b{initial}\s*\.\s*{species}\b")
        else:
            patterns.append(rf"\b{re.escape(name)}\b")

    if not patterns:
        patterns.append(rf"\b{re.escape(taxon_name.strip())}\b")

    return re.compile("|".join(patterns), re.IGNORECASE)


def detect_section_name(text):
    
    lower_text = text.lower().strip()
    if lower_text.startswith(("abstract", "summary")):
        return "Abstract"
    elif "introduction" in lower_text[:30]:
        return "Introduction"
    elif any(k in lower_text[:40] for k in ["materials and methods", "methodology", "methods"]):
        return "Materials and Methods"
    elif "results" in lower_text[:30]:
        return "Results"
    elif "discussion" in lower_text[:30]:
        return "Discussion"
    elif "conclusion" in lower_text[:30]:
        return "Conclusion"
    return "Main Content"


def find_best_evidence_spacy_with_section(abstract_text, content_text, subject_re, object_re, relation_lemmas):
    
    blocks = []
    if abstract_text.strip():
        blocks.append(("Abstract", abstract_text))
    if content_text.strip():
        # Divide o conteúdo por parágrafos para identificar mudanças de seção
        paragraphs = [p.strip() for p in content_text.split("\n\n") if p.strip()]
        current_sec = "Main Content"
        for p in paragraphs:
            sec_candidate = detect_section_name(p)
            if sec_candidate != "Main Content":
                current_sec = sec_candidate
            blocks.append((current_sec, p))

    matched_candidates = []

    for section_name, block_text in blocks:
        doc = nlp(block_text)
        for sent in doc.sents:
            sent_text = sent.text.replace("\n", " ").strip()
            has_subject = bool(subject_re.search(sent_text))
            has_object = bool(object_re.search(sent_text))

            if has_subject and has_object:
                combined_lemmas = {token.lemma_.lower() for token in sent}
                combined_raw = {token.text.lower() for token in sent}
                has_relation = bool(combined_lemmas.intersection(relation_lemmas) or combined_raw.intersection(relation_lemmas))

                if has_relation:
                    return sent_text, section_name, True

                matched_candidates.append((sent_text, section_name, False))

    if not matched_candidates:
        return None

    best_snippet, best_section, has_relation_term = matched_candidates[0]
    return best_snippet, best_section, has_relation_term


def main():
    parser = argparse.ArgumentParser(
        description="Search for textual evidence of SRO relationships and report exact paper section."
    )
    parser.add_argument('--database', dest='db', required=True)
    parser.add_argument('--relations_csv', dest='relations_csv', required=True)
    parser.add_argument('--output', dest='output', required=True)
    parser.add_argument('--pmid_col', default='pmid')
    parser.add_argument('--subject_col', default='sourceTaxonName')
    parser.add_argument('--relation_col', default='interactionTypeName')
    parser.add_argument('--object_col', default='targetTaxonName')
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    results = []

    with open(args.relations_csv, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            pmcid = row[args.pmid_col].strip()
            subject = row[args.subject_col].strip()
            relation = row[args.relation_col].strip()
            obj = row[args.object_col].strip()

            db_row = cur.execute(
                'SELECT abstract, content FROM pcw_literature WHERE pmcid = ?', (pmcid,)
            ).fetchone()

            if db_row is None or (not db_row[0] and not db_row[1]):
                continue

            abstract_text = db_row[0] or ""
            content_text = db_row[1] or ""

            subject_re = taxon_to_regex(subject)
            object_re = taxon_to_regex(obj)
            relation_lemmas = get_relation_lemmas(relation)

            evidence = find_best_evidence_spacy_with_section(
                abstract_text, content_text, subject_re, object_re, relation_lemmas
            )

            if evidence is not None:
                snippet, section, has_relation_term = evidence
                results.append({
                    'pmid': pmcid,
                    'subject': subject,
                    'relation': relation,
                    'object': obj,
                    'section': section,
                    'relation_term_matched': has_relation_term,
                    'snippet': snippet
                })

    results.sort(key=lambda r: (not r['relation_term_matched'], r['pmid']))

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pmid', 'subject', 'relation', 'object', 'section',
            'relation_term_matched', 'snippet'])
        writer.writeheader()
        writer.writerows(results)

    print(f"Processed and saved {len(results)} confirmed relations -> {args.output}")


if __name__ == '__main__':
    main()