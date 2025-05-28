import sys
import io, os
import numpy as np
import logging
import argparse
import torch
import transformers
from transformers import AutoModel, AutoTokenizer
from prettytable import PrettyTable
import mlflow

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = './SentEval/data'

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

def print_table(task_names, scores):
    tb = PrettyTable()
    tb.field_names = task_names
    tb.add_row(scores)
    print(tb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="Transformers' model name or path")
    parser.add_argument("--pooler", type=str, choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], default='cls')
    parser.add_argument("--mode", type=str, choices=['dev', 'test', 'fasttest'], default='test')
    parser.add_argument("--task_set", type=str, choices=['sts', 'transfer', 'full', 'na'], default='sts')
    parser.add_argument("--tasks", type=str, nargs='+', default=['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness'])
    
    args = parser.parse_args()

    # Start MLflow run
    mlflow.start_run()
    mlflow.log_param("model_name_or_path", args.model_name_or_path)
    mlflow.log_param("pooler", args.pooler)
    mlflow.log_param("mode", args.mode)
    mlflow.log_param("task_set", args.task_set)

    model = AutoModel.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.task_set == 'sts':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
    elif args.task_set == 'transfer':
        args.tasks = ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
    elif args.task_set == 'full':
        args.tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']
        args.tasks += ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']

    if args.mode == 'dev' or args.mode == 'fasttest':
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
        params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}
    elif args.mode == 'test':
        params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
        params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 5, 'epoch_size': 4}
    else:
        raise NotImplementedError

    def prepare(params, samples):
        return

    def batcher(params, batch, max_length=None):
        if len(batch) >= 1 and len(batch[0]) >= 1 and isinstance(batch[0][0], bytes):
            batch = [[word.decode('utf-8') for word in s] for s in batch]
        sentences = [' '.join(s) for s in batch]
        if max_length is not None:
            batch = tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True, max_length=max_length, truncation=True)
        else:
            batch = tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True)
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            pooler_output = outputs.pooler_output
            hidden_states = outputs.hidden_states
        if args.pooler == 'cls':
            return pooler_output.cpu()
        elif args.pooler == 'cls_before_pooler':
            return last_hidden[:, 0].cpu()
        elif args.pooler == "avg":
            return ((last_hidden * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
        elif args.pooler == "avg_first_last":
            first_hidden = hidden_states[1]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        elif args.pooler == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * batch['attention_mask'].unsqueeze(-1)).sum(1) / batch['attention_mask'].sum(-1).unsqueeze(-1)
            return pooled_result.cpu()
        else:
            raise NotImplementedError

    results = {}

    for task in args.tasks:
        se = senteval.engine.SE(params, batcher, prepare)
        result = se.eval(task)
        results[task] = result

    print("------ %s ------" % (args.mode))
    task_names = []
    scores = []
    for task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness']:
        task_names.append(task)
        if task in results:
            if task in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
                score = results[task]['all']['spearman']['all'] * 100
            else:
                score = results[task]['test']['spearman'].correlation * 100
        else:
            score = 0.0
        valor_formatado = "%.2f" % score
        scores.append(valor_formatado)
        mlflow.log_param(task, valor_formatado)

    avg_score = float(np.mean([float(s) for s in scores]))
    avg_formatado = "%.2f" % avg_score
    scores.append(avg_formatado)
    mlflow.log_param("Avg.", avg_formatado)
    # print_table(task_names, scores)
    mlflow.end_run()

if __name__ == "__main__":
    main()
