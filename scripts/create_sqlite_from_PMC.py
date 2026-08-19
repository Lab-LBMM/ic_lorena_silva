#!/usr/bin/env python

from Bio import Entrez

import json
import argparse
import sqlite3
import tarfile
import time
from crossref.restful import Works

start_time = time.time()

def get_args():
    print('Getting data from arguments.')
    version = 0.01
    parser = argparse.ArgumentParser(description='Stores articles in sqlite database',
                                    add_help=True)
    parser.add_argument('-v', '--version', action='version', version=version)
    parser.add_argument('--database', dest='db',
                        metavar='literature.db', type=str,
                        help='Database name',
                        required=True)
    parser.add_argument('--pmc_gz', dest='gz_file',
                        metavar='PMC000XXXXX_json_unicode.tar.gz', type=str,
                        help='BioC PMC articles (gz)',
                        required=True)
    parser.add_argument('--doi_to_keep', dest='doi_file',
                        metavar='doi.txt', type=str,
                        help='DOI list for selected journals',
                        required=False)
    parser.add_argument('--min_year', dest='min_year',
                        metavar='2005', type=int,
                        help='Consider only articles published this year or after',
                        required=False)
    parser.add_argument('--log_file', dest='log_file',
                        metavar='log.txt', type=str,
                        help='Log file',
                        required=False)
    parser.add_argument('--buffer_size', dest='buffer_size',
                        metavar='400', type=int,
                        help='Buffer size (number of inserts in batch)',
                        required=False, default=400)

    args = parser.parse_args()
    database = args.db
    pmc_file = args.gz_file
    min_year = args.min_year
    buffer_size = args.buffer_size

    if args.doi_file:
        doi_list_file = args.doi_file
    else:
        doi_list_file = ''

    if args.log_file:
        log_file = args.log_file
    else:
        log_file = ''

    return database, pmc_file, doi_list_file, min_year, log_file, buffer_size

def set_database(db):
    # Create sqlite3 database if it does not exist
    conn = sqlite3.connect(db)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS pcw_literature
                    (pmid integer, title text, year integer, doi text,\
                    journal_name text, first_author text, abstract text, content text)''')
    conn.commit()
    return conn, c

def get_first_author_name(infos):
    if 'name_0' in infos:
        author_infos = infos['name_0'].split(";")
        first_author_infos = author_infos[0].split(":")
        first_author = first_author_infos[1]
    else:
        first_author = ''
    return first_author

# LABEL: Reading Args
db, pmc_file, doi_list_file, min_year, log_file, max_buffer_size = get_args()

doi_list_to_keep = {}

sections_in_db = []

if doi_list_file:
    with open(doi_list_file) as f:
        dois = [doi.rstrip() for doi in f]
        
        # print(dois)

        for list_doi in dois:
            suffix = list_doi.split("/")
            prefix = suffix.pop(0)
            suffix = "/".join(suffix)

            if prefix in doi_list_to_keep.keys():
                doi_list_to_keep[prefix].append(suffix)
            else:
                doi_list_to_keep[prefix] = [suffix]

# LABEL: Configure Database Connection
# print(f'Configuring database: {db}')
db_conn, db_cursor = set_database(db)

#print(f'Uncompressing gzip file: {pmc_file}')
#gzip.decompress(gzip.open(pmc_file, 'rb'))

report_stats = {"missing_pmid":0,
                "missing_doi":0,
                "normal_cases_pmid":0,
                "normal_cases_doi":0,
                "filtered_doi_prefix":0}

data_buffer = []

with tarfile.open(pmc_file, 'r:gz') as tar:
    for member in tar.getmembers():

        if len(data_buffer) >= max_buffer_size:

            db_cursor.executemany('INSERT INTO pcw_literature VALUES (?,?,?,?,?,?,?,?)', data_buffer)
            db_conn.commit()
            # print(f'{len(data_buffer)} articles successfully added to the database.')
            data_buffer = []

        f = tar.extractfile(member)
        if f is not None:
            data_json = json.loads(f.read())
            pas = data_json['documents'][0]['passages']
            if "article-id_pmid" in pas[0]['infons'].keys():
                pmid = pas[0]['infons']['article-id_pmid']
                report_stats['normal_cases_pmid']+=1
            else:
                pmid = 0
                report_stats['missing_pmid']+=1

            teste_file_out = open("teste_file_out.txt", "a")
            start = time.time()
            pmid_exists = db_cursor.execute(f'SELECT * FROM pcw_literature WHERE pmid = {pmid}')

            existing_record = pmid_exists.fetchone()

            if existing_record is not None:
                continue
            end = time.time()
            teste_file_out.write("Time to select check condition: " + str(end - start) + "\n")
            teste_file_out.close()

            title = pas[0]['text']
            if "year" in pas[0]['infons']:
                try:
                    year = int(pas[0]['infons']['year'].replace("(", "").replace(")", "").replace(";", ""))
                except ValueError as ve:
                    year = 0
                    if log_file:
                        log_file_obj = open(log_file, "a")
                        log_file_obj.write(f"Year value error:\t{year}\n")
                        log_file_obj.close()
            else:
                year = 0
            
            if min_year:
                if year < min_year:
                    continue

            if "article-id_doi" in pas[0]['infons']:
                doi = pas[0]['infons']['article-id_doi']
                report_stats['normal_cases_doi']+=1
            else:
                doi = ''
                report_stats['missing_doi']+=1

            
            
            first_author = get_first_author_name(pas[0]['infons'])

            abs = ''
            article_text = ''
            section_title = ''

            for p in pas:
                if 'section_type' not in p['infons'].keys():
                        print(f'section_type does not exist for DOI: {doi}')
                else:
                    if(p['infons']['section_type'] not in ["ABSTRACT", "ACK_FUND", "COMP_INT",
                                                           "REF", "ABBR", "REVIEW_INFO",
                                                           "SUPPL", "TABLE", "TITLE", "APPENDIX",
                                                           "AUTH_CONT", "CASE", "KEYWORD"]):

                        if p['infons']['section_type'] not in sections_in_db:
                            sections_in_db.append(p['infons']['section_type'])

                        if section_title != p['infons']['section_type']:
                            article_text += p['infons']['section_type'] + '\n\n'
                            section_title = p['infons']['section_type']
                        article_text += p['text'] + '\n\n'
                    elif p['infons']['section_type'] == "ABSTRACT":
                        abs += p['text'] + '\n\n' # When it finds the abstract, it appends it to a string
            
            if doi_list_file:
                if doi:
                    suffix = doi.split("/")
                    prefix = suffix.pop(0)
                    suffix = "/".join(suffix)

                    if prefix in doi_list_to_keep.keys():
                        for doi_journal in doi_list_to_keep[prefix]:
                            if suffix.startswith(doi_journal):
                                try:
                                    works = Works()
                                    works_res = works.doi(doi)
                                    time.sleep(0.5)
                                    if works_res is not None:
                                        if 'container-title' in works_res.keys():
                                            journal_name = works_res['container-title'][0]
                                        else:
                                            journal_name = ''
                                        report_stats["filtered_doi_prefix"]+=1
                                        data_buffer.append((pmid,title,year,doi,journal_name,first_author,abs,article_text))
                                    else:
                                        journal_name = ''
                                        report_stats["filtered_doi_prefix"]+=1
                                        data_buffer.append((pmid,title,year,doi,journal_name,first_author,abs,article_text))
                                except:
                                    continue

            else:
                #Precisa inserir o journal_name
                journal_name = ' '
                data_buffer.append((pmid,title,year,doi,journal_name,first_author,abs,article_text))

        if report_stats['normal_cases_pmid'] % 1000 == 0:
            print('Seconds: ', time.time() - start_time, 'Normal cases: ', report_stats['normal_cases_pmid'])
        if (report_stats['missing_pmid'] % 1000 == 0) and (report_stats['missing_pmid'] != 0):
            print('Seconds: ', time.time() - start_time, 'Missing pmid: ', report_stats['missing_pmid'])

    if len(data_buffer) > 0:
        db_cursor.executemany('INSERT INTO pcw_literature VALUES (?,?,?,?,?,?,?,?)', data_buffer)
        db_conn.commit()
        # print(f'{len(data_buffer)} articles successfully added to the database.')
        data_buffer = []

if log_file:
    log_file_obj = open(log_file, "a")
    
    for section in sections_in_db:
        log_file_obj.write(f"Observed section:\t{section}\n")

    log_file_obj.close()
            
print('normal cases: ',report_stats['normal_cases_pmid'])
print('missing pmid: ',report_stats['missing_pmid'])

print('normal cases: ',report_stats['normal_cases_doi'])
print('missing doi: ',report_stats['missing_doi'])

print('filtered prefix doi: ',report_stats['filtered_doi_prefix'])
