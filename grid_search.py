import subprocess
from itertools import product

learning_rates = [1e-5, 3e-5, 5e-5]
batch_sizes = [64,128, 256, 512]

combinacoes = list(product(learning_rates, batch_sizes))

for i, (lr, bs) in enumerate(combinacoes):
    output_dir = f"result/sup_bertimbau_dataset_trad_lr{lr}_bs{bs}"
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
        '--fp16'
    ]
    subprocess.run(cmd)
