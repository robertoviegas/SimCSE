import subprocess
import os
from itertools import product

learning_rates = [1e-5, 3e-5, 5e-5]
batch_sizes = [64, 128, 256, 512]

combinacoes = list(product(learning_rates, batch_sizes))

for i, (lr, bs) in enumerate(combinacoes):
    output_dir = f"result/sup_bertimbau_dataset_trad_lr{lr}_bs{bs}"
    results_file = os.path.join(output_dir, "train_results.txt")

    # Verifica se o resultado já existe
    if os.path.exists(results_file):
        print(f"[SKIP] Já existe: {results_file}")
        continue  # Pula para a próxima combinação

    cmd = [
        "python", "train.py",
        "--model_name_or_path", "bertimbau_local",
        "--train_file", "data/portuguese_nli_trad/pt_nli_for_simcse.csv",
        "--output_dir", output_dir,
        "--num_train_epochs", "3",
        "--per_device_train_batch_size", str(bs),
        "--learning_rate", str(lr),
        "--max_seq_length", "32",
        "--evaluation_strategy", "steps",
        "--metric_for_best_model", "stsb_spearman",
        "--load_best_model_at_end",
        "--eval_steps", "250",
        "--pooler_type", "cls",
        "--overwrite_output_dir",
        "--temp", "0.05",
        "--do_train",
        "--do_eval",
        "--fp16"
    ]

    print(f"[RUN] Treinando com lr={lr}, bs={bs}")
    subprocess.run(cmd)
