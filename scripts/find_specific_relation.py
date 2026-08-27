#!/usr/bin/env python

import argparse
import csv
import re
import sqlite3
import requests
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. "
        "Run in terminal: python3 -m spacy download en_core_web_sm"
    )

taxonomy_cache = {}
parsed_articles_cache = {}

GLOBI_RELATION_LEMMAS = {
    "parasiteof": {"parasite", "parasitic", "parasitize", "ectoparasite", "endoparasite"},
    "hasparasite": {"parasite", "parasitic", "parasitize"},
    "endoparasiteof": {"endoparasite", "parasite"},
    "ectoparasiteof": {"ectoparasite", "parasite"},
    "pathogenof": {"pathogen", "pathogenic", "infect", "infection", "cause", "disease"},
    "haspathogen": {"pathogen", "infect", "infection"},
    "hashost": {"host", "infect", "associate", "association"},
    "hosts": {"host"},
    "symbiontof": {"symbiont", "symbiosis", "symbiotic", "associate"},
    "hassymbiont": {"symbiont", "symbiosis", "symbiotic"},
    "mutualistof": {"mutualist", "mutualism", "mutualistic", "symbiosis"},
    "hasmutualist": {"mutualist", "mutualism"},
    "commensalistof": {"commensal", "commensalism"},
    "hascommensalist": {"commensal"},
    "eats": {"feed", "eat", "consume", "predate", "prey", "harvest", "collect"},
    "preyson": {"predate", "prey", "consume", "feed"},
    "eatenby": {"feed", "eat", "consume"},
    "dispersalvectorof": {"dispersal", "vector", "transport", "carry"},
    "hasdispersalvector": {"dispersal", "vector"},
    "pollinates": {"pollinate", "pollination", "pollinator"},
    "pollinatedby": {"pollinate", "pollination"},
    "visitsfloweringplantof": {"visit", "forage"},
    "hasvector": {"vector", "transmit", "transmission"},
    "vectorof": {"vector", "transmit", "transmission"},
    "interactswith": {"interact", "interaction", "associate", "association"},
    "cooccurswith": {"cooccur", "co-occur", "find", "associate"},
    "isolatedfrom": {"isolate", "isolation"},
    "livesinside": {"live", "inhabit", "dwell"},
    "livesnear": {"live", "find"},
    "epiphyteof": {"epiphyte", "epiphytic"},
    "hasepiphyte": {"epiphyte"},
    "farms": {"farm", "cultivate", "fungiculture"},
    "farmedby": {"farm", "cultivate"},
}

DEFAULT_RELATION_LEMMAS = {"associate", "association", "find", "relate", "relation"}


def get_relation_lemmas(interaction_type):
    key = re.sub(r'[\s_\-]', '', interaction_type.strip().lower())
    return GLOBI_RELATION_LEMMAS.get(key, DEFAULT_RELATION_LEMMAS)


def get_gbif_taxonomic_hierarchy(taxon_name):
    taxon_name = taxon_name.strip()
    if taxon_name in taxonomy_cache:
        return taxonomy_cache[taxon_name]

    matched_names = set()
    matched_names.add(taxon_name)

    try:
        url_match = f"https://api.gbif.org/v1/species/match?name={requests.utils.quote(taxon_name)}"
        resp = requests.get(url_match, timeout=4)

        if resp.status_code == 200:
            data = resp.json()
            usage_key = data.get('usageKey')
            rank = data.get('rank', '').upper()

            if usage_key:
                accepted_name = data.get('acceptedUsage', {}).get('canonicalName') or data.get('species')
                if accepted_name:
                    matched_names.add(accepted_name)

                url_syn = f"https://api.gbif.org/v1/species/{usage_key}/synonyms"
                resp_syn = requests.get(url_syn, timeout=3)
                if resp_syn.status_code == 200:
                    for item in resp_syn.json().get('results', []):
                        if item.get('canonicalName'):
                            matched_names.add(item['canonicalName'])

                if rank in ['FAMILY', 'ORDER', 'SUPERFAMILY', 'SUBFAMILY']:
                    url_children = f"https://api.gbif.org/v1/species/{usage_key}/children?limit=100"
                    resp_child = requests.get(url_children, timeout=4)
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
        patterns.append(rf"\b{re.escape(name)}\b")
        parts = name.split()

        if len(parts) >= 2:
            genus = parts[0]
            species = " ".join(parts[1:])
            initial = genus[0]
            patterns.append(rf"\b{initial}\s*\.\s*{re.escape(species)}\b")
            patterns.append(rf"\b{re.escape(genus)}\b")

    return re.compile("|".join(patterns), re.IGNORECASE)


def find_best_evidence_spacy(doc, subject_re, object_re, relation_lemmas, window_size=3):
    sents = list(doc.sents)
    matched_candidates = []

    for i in range(len(sents)):
        window_sents = sents[i : i + window_size + 1]
        window_text = " ".join([s.text for s in window_sents]).replace("\n", " ").strip()

        has_subject = bool(subject_re.search(window_text))
        has_object = bool(object_re.search(window_text))

        if has_subject and has_object:
            combined_lemmas = {token.lemma_.lower() for s in window_sents for token in s}
            has_relation = bool(combined_lemmas.intersection(relation_lemmas))
            distance_score = len(window_sents) - 1

            if distance_score == 0 and has_relation:
                return window_text, 0, True

            matched_candidates.append((window_text, distance_score, has_relation))

    if not matched_candidates:
        return None

    matched_candidates.sort(key=lambda c: (not c[2], c[1]))
    best_snippet, distance_score, has_relation_term = matched_candidates[0]

    return best_snippet, distance_score, has_relation_term


def main():
    parser = argparse.ArgumentParser(
        description="Search for textual evidence of SRO relationships using spaCy and GBIF API."
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
                'SELECT content FROM pcw_literature WHERE pmcid = ?', (pmcid,)
            ).fetchone()

            if db_row is None or not db_row[0]:
                results.append({
                    'pmid': pmcid, 'subject': subject, 'relation': relation,
                    'object': obj, 'found': False, 'relation_term_matched': False,
                    'distance_chars': 999, 'snippet': '(article not found or missing content)'
                })
                continue

            content = db_row[0]

            if pmcid not in parsed_articles_cache:
                parsed_articles_cache[pmcid] = nlp(content)
            doc = parsed_articles_cache[pmcid]

            subject_re = taxon_to_regex(subject)
            object_re = taxon_to_regex(obj)
            relation_lemmas = get_relation_lemmas(relation)

            evidence = find_best_evidence_spacy(doc, subject_re, object_re, relation_lemmas)

            if evidence is None:
                results.append({
                    'pmid': pmcid, 'subject': subject, 'relation': relation,
                    'object': obj, 'found': False, 'relation_term_matched': False,
                    'distance_chars': 999, 'snippet': '(subject and/or object not mentioned in text)'
                })
                continue

            snippet, distance_score, has_relation_term = evidence
            results.append({
                'pmid': pmcid, 'subject': subject, 'relation': relation,
                'object': obj, 'found': True,
                'relation_term_matched': has_relation_term,
                'distance_chars': distance_score, 'snippet': snippet
            })

    # Sort results: found matches first, then matched relations, sentence distance, and PMCID
    results.sort(key=lambda r: (
        not r['found'],
        not r['relation_term_matched'],
        r['distance_chars'] if isinstance(r['distance_chars'], int) else 999,
        r['pmid']
    ))

    for r in results:
        if not r['found']:
            r['distance_chars'] = ''

    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'pmid', 'subject', 'relation', 'object', 'found',
            'relation_term_matched', 'distance_chars', 'snippet'])
        writer.writeheader()
        writer.writerows(results)

    found_count = sum(1 for r in results if r['found'])
    print(f"Processed {len(results)} relations ({found_count} matches found at top) -> {args.output}")


if __name__ == '__main__':
    main()