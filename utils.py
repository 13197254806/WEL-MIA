import copy
import json
import random
import os
import shutil
import torch
import bisect
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve


def add_directory(dir):
    parts = dir.rstrip('/').split('/')
    now_dir = ''
    for part in parts:
        now_dir += part
        if not os.path.exists(now_dir):
            os.mkdir(now_dir)
        now_dir += '/'

def load_json_file(position):
    with open(position, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_data_in_json_file(data, position):
    with open(position, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def sort_attack_names(attacks_dic):
    priority = ['Loss Attack', 'Neighbor Attack', 'Min-K% Attack', 'LiRA-Base', 'LiRA-Align', 'Ours-Base', 'Ours-Align']
    priority = dict(zip(priority, list(range(len(priority)))))
    sorted_attacks_dic = dict(sorted(attacks_dic.items(), key=lambda x: priority[x[0]]))
    return sorted_attacks_dic

def get_sequence_probs(confidence, ids):
    """
    Get the token-level prediction probabilities from the vocab.

    Parameters:
        confidence: torch.Tensor of shape (batch_size, max_length, vocab_size), the confidence output on the whole vocab.
        ids: torch.LongTensor of shape (batch_size, max_length), the encoded text ids.

    Returns:
        torch.Tensor of shape (batch_size, max_length), probability predictions for a batch of text.
    """
    index = torch.unsqueeze(ids, dim=-1)
    probs = torch.gather(confidence, dim=2, index=index)
    probs = torch.squeeze(probs, dim=2)
    probs[torch.isnan(probs)] = 1e-9
    probs[torch.isinf(probs)] = 1e-9
    probs[probs <= 1e-9] = 1e-9
    return probs.cpu().numpy()


def generate_paraphrased_text(data_list, mask_model, mask_tokenizer, neighbour_number, device):
    """
    Get the paraphrased text (neighbours).

    Parameters:
        data_list: List, text list for paraphrasing.
        mask_model: paraphrase (or mask) model.
        mask_tokenizer: tokenizer.
        neighbour_number: int, the number of neighbours to generate for a sentence.
        device: device.

    Returns:
        torch.Tensor of shape (batch_size, max_length), probability predictions for a batch of text.
    """
    paraphrased_list = []
    for sentence in tqdm(data_list, desc='generating neighbours'):
        text = "paraphrase: " + sentence
        encoding = mask_tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
        outputs = mask_model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=True,
            top_k=40,
            top_p=0.9,
            max_new_tokens=len(input_ids[0]),
            num_return_sequences=neighbour_number
        )
        for output in outputs:
            line = mask_tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            paraphrased_list.append(line)
    return paraphrased_list


def draw_roc_curves(labels_list, scores_list, plt, scale='log', desc=''):
    '''
    Draw roc and return metrics.

    Parameters:
        labels_list: List, list of labels.
        scores_list: List, list of membership scores.
        plt: matplotlib.pyplot.
        scale: str, draw roc curves in log scale or linear scale
        desc: str, description.

    Returns:
        tuple of (plt, metrics)
    '''
    fpr, tpr, thresholds = roc_curve(labels_list, scores_list, pos_label=1)
    auc = 0
    for i in range(1, len(fpr)):
        auc += tpr[i] * (fpr[i] - fpr[i-1])

    if scale == 'log':
        plt.xlim([1e-3, 1])
        plt.ylim([1e-3, 1])
        plt.xscale('log')
        plt.yscale('log')

    plt.plot(fpr, tpr, label=desc + f'(auc={round(auc, 3)})')

    idx_at_point_01 = bisect.bisect_left(fpr, 0.01)
    idx_at_point_001 = bisect.bisect_left(fpr, 0.001)

    # the metrics including AUC, TPR @1%FPR, TPR@0.1%FPR
    metric = {
        'auc': round(auc, 3),
        'tpr_at_point_01_fpr': round(tpr[idx_at_point_01], 4),
        'tpr_at_point_001_fpr': round(tpr[idx_at_point_001], 5)
    }
    return plt, metric

