import pandas as pd

print("Carregando dados...")
df = pd.read_csv("data/raw/interactions.csv", low_memory=False)

print(f"Total de linhas: {len(df)}")
print("\nColunas disponíveis:")
print(df.columns.tolist())

print("\nTipos de interação encontrados:")
print(df["interaction_type"].value_counts())

df.to_csv("data/processed/formigas_interactions.csv", index=False)
print("\nArquivo salvo em data/processed/formigas_interactions.csv")