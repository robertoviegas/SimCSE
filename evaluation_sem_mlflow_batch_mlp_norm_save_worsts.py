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
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def run_eval(args, model, tokenizer, device):
    # Set up the tasks (apenas STS)
    tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
             'STSBenchmark', 'SICKRelatedness']

    # Set params for SentEval
    if args.mode in ['dev', 'fasttest']:
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                         'tenacity': 3, 'epoch_size': 2}
    else:  # test
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                         'tenacity': 5, 'epoch_size': 4}

    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]

        sentences = [' '.join(s) for s in batch]

        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
            max_length=128,
            truncation=True
        )

        for k in batch:
            batch[k] = batch[k].to(device)

        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)

            # Pooling
            if args.pooler == 'cls':
                if hasattr(model, 'pooler'):
                    emb = model.pooler(outputs.last_hidden_state)   # CLS + MLP treinado
                else:
                    emb = outputs.pooler_output                    # fallback (sem MLP)
            elif args.pooler == 'cls_before_pooler':
                emb = outputs.last_hidden_state[:, 0]
            elif args.pooler == "avg":
                emb = ((outputs.last_hidden_state * batch['attention_mask'].unsqueeze(-1)).sum(1) /
                        batch['attention_mask'].sum(-1).unsqueeze(-1))
            elif args.pooler == "avg_first_last":
                first_hidden = outputs.hidden_states[1]
                last_hidden = outputs.hidden_states[-1]
                emb = ((first_hidden + last_hidden) / 2.0 *
                       batch['attention_mask'].unsqueeze(-1)).sum(1) / \
                       batch['attention_mask'].sum(-1).unsqueeze(-1)
            elif args.pooler == "avg_top2":
                second_last = outputs.hidden_states[-2]
                last_hidden = outputs.hidden_states[-1]
                emb = ((last_hidden + second_last) / 2.0 *
                       batch['attention_mask'].unsqueeze(-1)).sum(1) / \
                       batch['attention_mask'].sum(-1).unsqueeze(-1)
            else:
                raise NotImplementedError

            # Normalização L2 (essencial no SimCSE)
            emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu()

    # Rodar todas as tasks
    results = {}
    for task in tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    # Coletar scores agregados
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


def salvar_piores_exemplos(results, top_k=5, saida="worst_sts_examples.txt"):
    """Extrai os piores pares das STS tasks e salva em um txt"""
    exemplos = []

    for task, result in results.items():
        # Algumas tasks têm resultados em 'all', outras em 'test'
        if "test" in result:
            y_true = np.array(result["test"]["y"])
            y_pred = np.array(result["test"]["x"])
            sents = result["test"]["sentences"]
        else:
            y_true = np.array(result["all"]["y"])
            y_pred = np.array(result["all"]["x"])
            sents = result["all"]["sentences"]

        # Similaridade prevista via produto escalar
        sim_pred = [np.dot(x[0], x[1]) for x in y_pred]

        # Diferença absoluta
        diffs = np.abs(y_true - sim_pred)

        for i, d in enumerate(diffs):
            exemplos.append({
                "task": task,
                "s1": sents[i][0],
                "s2": sents[i][1],
                "gold": y_true[i],
                "pred": sim_pred[i],
                "diff": d
            })

    # Ordenar piores casos
    piores = sorted(exemplos, key=lambda x: x["diff"], reverse=True)[:top_k]

    # Salvar em TXT
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

    # Load model/tokenizer once
    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Executar avaliação
    all_results = []
    logging.info(f"Rodando avaliação ...")
    scores, results = run_eval(args, model, tokenizer, device)
    all_results.append([f"{s:.2f}" for s in scores])

    # Montar tabela final
    tb = PrettyTable()
    tb.field_names = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'STSBenchmark', 'SICKRelatedness', 'Avg.']
    for row in all_results:
        tb.add_row(row)
    print(f"Modelo carregado: {args.model_name_or_path}")
    print("\nResultados finais:")
    print(tb)

    # Salvar os piores exemplos
    salvar_piores_exemplos(results, top_k=5)


if __name__ == "__main__":
    main()
