import json
from pathlib import Path
from collections import Counter

def auditar_logs_falhas(pasta_logs="."):
    arquivos_jsonl = list(Path(pasta_logs).rglob("metricas_blocos_*.jsonl"))
    
    if not arquivos_jsonl:
        print("[!] Nenhum arquivo de métricas .jsonl encontrado.")
        return

    print(f"[*] Encontrados {len(arquivos_jsonl)} arquivos de log .jsonl\n")

    for arq in arquivos_jsonl:
        erros_contagem = Counter()
        total_blocos = 0
        blocos_com_erro = 0
        
        with open(arq, "r", encoding="utf-8") as f:
            for line in f:
                total_blocos += 1
                try:
                    dado = json.loads(line)
                    if dado.get("erro"):
                        blocos_com_erro += 1
                        # Categoriza o tipo de erro
                        msg_erro = str(dado["erro"])
                        if "finish_reason=length" in msg_erro:
                            erros_contagem["Truncado (max_tokens estourado)"] += 1
                        elif "JSON/schema" in msg_erro:
                            erros_contagem["JSON/Esquema Inválido"] += 1
                        elif "servidor" in msg_erro or "timeout" in msg_erro:
                            erros_contagem["Timeout no Ollama"] += 1
                        else:
                            erros_contagem[msg_erro[:40]] += 1
                except Exception:
                    continue
        
        if blocos_com_erro > 0:
            nome_teste = arq.stem.replace("metricas_blocos_", "")
            print(f"=== ❌ Falhas em: {nome_teste} ===")
            print(f"    Blocos afetados: {blocos_com_erro}/{total_blocos}")
            for tipo_erro, quant in erros_contagem.items():
                print(f"    -> {tipo_erro}: {quant} bloco(s)")
            print("-" * 50)

if __name__ == "__main__":
    # Ajuste para a pasta onde seus arquivos .jsonl foram salvos
    auditar_logs_falhas("data/processed/metricas/")