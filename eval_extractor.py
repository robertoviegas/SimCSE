import os
import re

# Caminho base onde estão os diretórios dos modelos
base_dir = '/raid/robertoviegas/SimCSE/result'

# Lista todos os diretórios dentro do base_dir
model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Dicionário para armazenar os resultados
results = {}

for model_dir in model_dirs:
    eval_path = os.path.join(base_dir, model_dir, 'eval_results.txt')
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            content = f.read()
            match = re.search(r'eval_avg_sts\s*=\s*([0-9.]+)', content)
            if match:
                results[model_dir] = float(match.group(1))
            else:
                results[model_dir] = None  # Valor inválido
    else:
        results[model_dir] = None  # Arquivo não encontrado

# Filtra apenas os resultados válidos (com valor float) e ordena do maior para o menor
sorted_results = sorted(
    ((model, value) for model, value in results.items() if isinstance(value, float)),
    key=lambda x: x[1],
    reverse=True
)

# Print dos modelos ranqueados
print("\nRanking dos modelos por eval_avg_sts (maior para menor):\n")
for rank, (model, value) in enumerate(sorted_results, start=1):
    print(f"{rank:2d}. {model:<40} -> eval_avg_sts = {value:.4f}")

# Opcional: mostra também os que não foram encontrados ou com erro
print("\nModelos com erro ou sem eval_avg_sts válido:")
for model, value in results.items():
    if not isinstance(value, float):
        print(f"- {model}: {value}")
