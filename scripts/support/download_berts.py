from transformers import AutoTokenizer, AutoModel

modelos = [
    "allenai/scibert_scivocab_uncased",
    "dmis-lab/biobert-v1.1",
    "microsoft/mdeberta-v3-base"
]

for modelo in modelos:
    print(f"A descarregar {modelo}...")
    AutoTokenizer.from_pretrained(modelo)
    # Adicionado o parâmetro use_safetensors=True para contornar o bloqueio de segurança
    AutoModel.from_pretrained(modelo, use_safetensors=True)
    print(f"[OK] Modelo guardado localmente!\n")