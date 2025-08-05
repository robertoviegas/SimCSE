from huggingface_hub import hf_hub_download, list_repo_files
import os
import shutil

model_name = "neuralmind/bert-base-portuguese-cased"
local_dir = "./bertimbau_local"

model_name = "neuralmind/bert-large-portuguese-cased"
local_dir = "./bertimbau_local_large"

os.makedirs(local_dir, exist_ok=True)

files = list_repo_files(model_name)

print(f"Arquivos no modelo {model_name}:")
for f in files:
    print(f)

for file in files:
    # Faz o download do arquivo e retorna o caminho local no cache
    cached_path = hf_hub_download(repo_id=model_name, filename=file)
    # Copia para a pasta local
    dest_path = os.path.join(local_dir, file)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copyfile(cached_path, dest_path)
    print(f"Arquivo {file} copiado para {dest_path}")
