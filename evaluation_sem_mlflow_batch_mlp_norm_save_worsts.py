import sys
import os
import numpy as np
import logging
import argparse
from prettytable import PrettyTable
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data/downstream/STS'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# Lista global para armazenar pares avaliados
pares_avaliados = []

# Mapeamento de task → pasta real
TASK_TO_FOLDER = {
    'STS12': 'STS12-en-test',
    'STS13': 'STS13-en-test',
    'STS14': 'STS14-en-test',
    'STS15': 'STS15-en-test',
    'STS16': 'STS16-en-test',
    'STSBenchmark': 'STSBenchmark',
    'SICKRelatedness': 'SICK'
}

def run_eval(args, model, tokenizer, device):
    tasks = list(TASK_TO_FOLDER.keys())

    if args.mode in ['dev', 'fasttest']:
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                'tenacity': 3, 'epoch_size': 2}
    else:
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                'tenacity': 5, 'epoch_size': 4}

    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        global pares_avaliados
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]
        batch_enc = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=128,
            truncation=True
        )
        for k in batch_enc:
            batch_enc[k] = batch_enc[k].to(device)

        with torch.no_grad():
            outputs = model(**batch_enc, output_hidden_states=True, return_dict=True)

            if args.pooler == 'cls':
                if hasattr(model, 'pooler'):
                    emb = model.pooler(outputs.last_hidden_state)
                else:
                    emb = outputs.pooler_output
            elif args.pooler == 'cls_before_pooler':
                emb = outputs.last_hidden_state[:, 0]
            elif args.pooler == "avg":
                emb = ((outputs.last_hidden_state * batch_enc['attention_mask'].unsqueeze(-1)).sum(1) /
                       batch_enc['attention_mask'].sum(-1).unsqueeze(-1))
            elif args.pooler == "avg_first_last":
                first_hidden = outputs.hidden_states[1]
                last_hidden = outputs.hidden_states[-1]
                emb = ((first_hidden + last_hidden) / 2.0 *
                       batch_enc['attention_mask'].unsqueeze(-1)).sum(1) / \
                       batch_enc['attention_mask'].sum(-1).unsqueeze(-1)
            elif args.pooler == "avg_top2":
                second_last = outputs.hidden_states[-2]
                last_hidden = outputs.hidden_states[-1]
                emb = ((last_hidden + second_last) / 2.0 *
                       batch_enc['attention_mask'].unsqueeze(-1)).sum(1) / \
                       batch_enc['attention_mask'].sum(-1).unsqueeze(-1)
            else:
                raise NotImplementedError

            emb = F.normalize(emb, p=2, dim=1)

        # Salvar embeddings e frases
        for i, sent in enumerate(sentences):
            pares_avaliados.append({
                "sentence": sent,
                "embedding": emb[i].cpu().numpy()
            })

        return emb.cpu()

    results = {}
    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Scores agregados
    scores = []
    for task in tasks:
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                scores.append(results[task]['all']['spearman']['all'] * 100)
            else:
                scores.append(results[task]['test']['spearman'].correlation * 100)
        else:
            scores.append(0.0)
    avg = sum(scores) / len(scores)
    scores.append(avg)
    return scores, results

def salvar_piores_exemplos(tasks, top_k=5, args=None):
    exemplos = []
    os.makedirs("outputs", exist_ok=True)
    saida = os.path.join("outputs",
                         f"worst_sts_examples_{os.path.basename(args.model_name_or_path).replace('/', '_')}.txt")

    for task in tasks:
        folder = TASK_TO_FOLDER[task]
        task_dir = os.path.join(PATH_TO_DATA, folder)

        # Listar arquivos input/gs que correspondem
        input_files = [f for f in os.listdir(task_dir) if f.startswith("STS.input")]
        gs_files = [f for f in os.listdir(task_dir) if f.startswith("STS.gs")]

        # Mapear pares input <-> gs
        for in_file, gs_file in zip(sorted(input_files), sorted(gs_files)):
            in_path = os.path.join(task_dir, in_file)
            gs_path = os.path.join(task_dir, gs_file)

            sentences1, sentences2 = [], []
            with open(in_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 2:
                        continue
                    s1, s2 = parts
                    sentences1.append(s1)
                    sentences2.append(s2)

            labels = []
            with open(gs_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        labels.append(float(line.strip()))
                    except:
                        continue

            for s1, s2, gold in zip(sentences1, sentences2, labels):
                emb1 = next((p["embedding"] for p in pares_avaliados if p["sentence"] == s1), None)
                emb2 = next((p["embedding"] for p in pares_avaliados if p["sentence"] == s2), None)
                if emb1 is None or emb2 is None:
                    continue
                sim_pred = float(np.dot(emb1, emb2))
                diff = abs(gold - sim_pred)
                exemplos.append({
                    "task": task,
                    "s1": s1,
                    "s2": s2,
                    "gold": gold,
                    "pred": sim_pred,
                    "diff": diff
                })

    piores = sorted(exemplos, key=lambda x: x["diff"], reverse=True)[:top_k]

    with open(saida, "w", encoding="utf-8") as f:
        for w in piores:
            f.write(f"Task: {w['task']}\n")
            f.write(f"Sentence 1: {w['s1']}\n")
            f.write(f"Sentence 2: {w['s2']}\n")
            f.write(f"Gold similarity: {w['gold']:.3f}\n")
            f.write(f"Predicted similarity: {w['pred']:.3f}\n")
            f.write(f"Difference: {w['diff']:.3f}\n")
            f.write("-"*60 + "\n")

    logging.info(f"Piores exemplos salvos em {saida}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Transformers model name or path")
    parser.add_argument("--pooler", type=str,
                        choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'],
                        default='cls')
    parser.add_argument("--mode", type=str,
                        choices=['dev', 'test', 'fasttest'],
                        default='test')
    args = parser.parse_args()

    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logging.info(f"Rodando avaliação ...")
    scores, results = run_eval(args, model, tokenizer, device)

    # Montar tabela final
    tb = PrettyTable()
    tb.field_names = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'STSBenchmark', 'SICKRelatedness', 'Avg.']
    tb.add_row([f"{s:.2f}" for s in scores])
    print(f"Modelo carregado: {args.model_name_or_path}")
    print("\nResultados finais:")
    print(tb)

    # Salvar os piores exemplos com gold labels
    salvar_piores_exemplos(list(TASK_TO_FOLDER.keys()), top_k=5, args=args)

if __name__ == "__main__":
    main()
