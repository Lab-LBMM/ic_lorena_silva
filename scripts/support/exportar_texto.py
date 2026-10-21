import sqlite3
import os

def exportar_artigos(caminho_db, ficheiro_saida, limite=5):
    print(f"[*] A extrair os primeiros {limite} artigos da base de dados...")
    
    try:
        conn = sqlite3.connect(caminho_db)
        cursor = conn.cursor()
        
        # Usa exatamente a mesma query do seu script de extração
        query = f"SELECT pmcid, title, abstract, content FROM pcw_literature LIMIT {limite}"
        cursor.execute(query)
        artigos = cursor.fetchall()
        
        with open(ficheiro_saida, "w", encoding="utf-8") as f:
            for i, (pmcid, title, abstract, content) in enumerate(artigos, 1):
                f.write(f"# ==========================================\n")
                f.write(f"# ARTIGO {i} | PMCID: {pmcid}\n")
                f.write(f"# ==========================================\n\n")
                
                f.write(f"## Título\n{title}\n\n")
                
                f.write(f"## Abstract\n{abstract if abstract else 'Sem abstract disponível.'}\n\n")
                
                f.write(f"## Content (Texto Completo)\n{content if content else 'Sem conteúdo disponível.'}\n\n")
                
                f.write(f"---\n\n\n")
                
        conn.close()
        print(f"[+] Sucesso! Textos exportados para: {ficheiro_saida}")
        
    except Exception as e:
        print(f"[!] Erro ao exportar: {e}")

if __name__ == "__main__":
    # Certifique-se de que o caminho aponta para onde a sua db realmente está
    CAMINHO_DB = "data/data_base/literature.db"
    SAIDA = "artigos_para_leitura.md"
    
    exportar_artigos(CAMINHO_DB, SAIDA, limite=5)