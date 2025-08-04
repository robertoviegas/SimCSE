
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
                results[model_dir] = 'eval_avg_sts não encontrado'
    else:
        results[model_dir] = 'eval_results.txt não encontrado'

# Imprime os resultados
for model, value in sorted(results.items()):
    print(f'{model}: {value}')
