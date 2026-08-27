#!/usr/bin/env python

import argparse
import json
import sqlite3
import time
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError
from crossref.restful import Works
from lxml import etree

start_time = time.time()

BUCKET = "pmc-oa-opendata"


def get_args():
    parser = argparse.ArgumentParser(
        description="Stores PMC OA articles (S3/JATS) in a SQLite database",
        add_help=True,
    )
    parser.add_argument("--database", dest="db", required=True)
    parser.add_argument(
        "--pmcid_list",
        dest="pmcid_file",
        required=True,
        help="Text file with one PMCID per line, e.g., PMC1234567",
    )
    parser.add_argument("--doi_to_keep", dest="doi_file", required=False)
    parser.add_argument(
        "--min_year", dest="min_year", type=int, required=False
    )
    parser.add_argument(
        "--buffer_size", dest="buffer_size", type=int, default=400
    )
    args = parser.parse_args()
    return (
        args.db,
        args.pmcid_file,
        args.doi_file or "",
        args.min_year,
        args.buffer_size,
    )


def set_database(db):
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS pcw_literature
                    (pmcid text PRIMARY KEY, pmid integer, title text, year integer, doi text,
                     journal_name text, first_author text, abstract text, content text)"""
    )
    conn.commit()
    return conn, c


def find_latest_version(s3, pmcid):
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=f"{pmcid}.", Delimiter="/")
    prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
    if not prefixes:
        return None
    versions = [p.rstrip("/") for p in prefixes]
    versions.sort(key=lambda v: int(v.split(".")[-1]))
    return versions[-1]


def fetch_object(s3, key):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=key)
        return obj["Body"].read()
    except ClientError:
        return None


def get_first_author(root):
    contrib = root.find('.//front//contrib[@contrib-type="author"]/name')
    if contrib is None:
        return ""
    surname = contrib.findtext("surname") or ""
    given = contrib.findtext("given-names") or ""
    return f"{given} {surname}".strip()


def get_year(root):
    for pub_date in root.findall(".//front//pub-date"):
        y = pub_date.findtext("year")
        if y:
            try:
                return int(y)
            except ValueError:
                continue
    return 0


def extract_sections(root):
    abs_text = ""
    body_text = ""
    abstract_el = root.find(".//front//abstract")
    if abstract_el is not None:
        abs_text = " ".join(
            t.strip() for t in abstract_el.itertext() if t.strip()
        )

    body = root.find(".//body")
    if body is not None:
        secs = body.findall(".//sec")
        if secs:
            for sec in secs:
                sec_type = sec.get("sec-type", "")
                title_el = sec.find("title")
                title_text = (
                    title_el.text.strip()
                    if title_el is not None and title_el.text
                    else sec_type
                )
                paras = sec.findall(".//p")
                sec_text = " ".join(
                    t.strip() for p in paras for t in p.itertext() if t.strip()
                )
                if sec_text:
                    body_text += (
                        f"{title_text.upper()}\n\n{sec_text}\n\n"
                        if title_text
                        else f"{sec_text}\n\n"
                    )
        else:
            paras = body.findall("p")
            body_text = " ".join(
                t.strip() for p in paras for t in p.itertext() if t.strip()
            )
    return abs_text, body_text


def main():
    db, pmcid_file, doi_list_file, min_year, max_buffer_size = get_args()

    doi_list_to_keep = {}
    if doi_list_file:
        with open(doi_list_file) as f:
            for list_doi in f:
                list_doi = list_doi.rstrip()
                prefix, _, suffix = list_doi.partition("/")
                doi_list_to_keep.setdefault(prefix, []).append(suffix)

    db_conn, db_cursor = set_database(db)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    with open(pmcid_file) as f:
        pmcids = [line.strip() for line in f if line.strip()]

    report_stats = {
        "missing_pmid": 0,
        "missing_doi": 0,
        "normal_cases_pmid": 0,
        "normal_cases_doi": 0,
        "filtered_doi_prefix": 0,
        "not_found": 0,
    }
    data_buffer = []

    for pmcid in pmcids:
        if len(data_buffer) >= max_buffer_size:
            db_cursor.executemany(
                "INSERT OR REPLACE INTO pcw_literature VALUES (?,?,?,?,?,?,?,?,?)",
                data_buffer,
            )
            db_conn.commit()
            data_buffer = []

        version_prefix = find_latest_version(s3, pmcid)
        if version_prefix is None:
            report_stats["not_found"] += 1
            continue

        meta_raw = fetch_object(s3, f"metadata/{version_prefix}.json")
        if meta_raw is None:
            report_stats["not_found"] += 1
            continue
        meta = json.loads(meta_raw)

        pmid = meta.get("pmid") or 0
        doi = meta.get("doi") or ""
        if pmid:
            report_stats["normal_cases_pmid"] += 1
        else:
            report_stats["missing_pmid"] += 1
        if doi:
            report_stats["normal_cases_doi"] += 1
        else:
            report_stats["missing_doi"] += 1

        exists = db_cursor.execute(
            "SELECT 1 FROM pcw_literature WHERE pmcid = ?", (pmcid,)
        ).fetchone()
        if exists is not None:
            continue

        xml_raw = fetch_object(s3, f"{version_prefix}/{version_prefix}.xml")
        if xml_raw is None:
            continue

        try:
            root = etree.fromstring(xml_raw)
        except etree.XMLSyntaxError:
            continue

        year = get_year(root)
        if min_year and year and year < min_year:
            continue

        title = meta.get("title", "")
        first_author = get_first_author(root)
        abs_text, article_text = extract_sections(root)

        if doi_list_file:
            if doi:
                prefix, _, suffix = doi.partition("/")
                for doi_journal in doi_list_to_keep.get(prefix, []):
                    if suffix.startswith(doi_journal):
                        try:
                            works_res = Works().doi(doi)
                            time.sleep(0.5)
                            journal_name = (
                                works_res["container-title"][0]
                                if works_res and "container-title" in works_res
                                else ""
                            )
                        except Exception:
                            journal_name = ""
                        report_stats["filtered_doi_prefix"] += 1
                        data_buffer.append(
                            (
                                pmcid,
                                pmid,
                                title,
                                year,
                                doi,
                                journal_name,
                                first_author,
                                abs_text,
                                article_text,
                            )
                        )
                        break
        else:
            journal_name = meta.get("citation", "")
            data_buffer.append(
                (
                    pmcid,
                    pmid,
                    title,
                    year,
                    doi,
                    journal_name,
                    first_author,
                    abs_text,
                    article_text,
                )
            )

        if (
            report_stats["normal_cases_pmid"] % 1000 == 0
            and report_stats["normal_cases_pmid"] != 0
        ):
            print(
                "Seconds:",
                time.time() - start_time,
                "Normal cases:",
                report_stats["normal_cases_pmid"],
            )

    if data_buffer:
        db_cursor.executemany(
            "INSERT OR REPLACE INTO pcw_literature VALUES (?,?,?,?,?,?,?,?,?)",
            data_buffer,
        )
        db_conn.commit()

    print(report_stats)


if __name__ == "__main__":
    main()