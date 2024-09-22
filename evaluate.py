from datasets import Dataset
from torch.utils.data import DataLoader
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *
from attacks import *


parser = argparse.ArgumentParser()
parser.add_argument('--tokenizer_path', default='model/target_model', type=str, required=False, help='path of the tokenizer')
parser.add_argument('--eval_batch_size', default=2, type=int, required=False, help='batch size for evaluation')
parser.add_argument('--target_model', default='model/target_model', type=str, required=False, help='path of the target model')
parser.add_argument('--reference_model_base', default='model/reference_model_base', type=str, required=False, help='path of the base reference model')
parser.add_argument('--reference_model_align', default='model/reference_model_align', type=str, required=False, help='path of the align reference model')
parser.add_argument('--paraphrase_model', default='model/paraphrase_model', type=str, required=False, help='path of the align paraphrase model')

parser.add_argument('--max_length', default=128, type=int, required=False, help='the max sentence length')
parser.add_argument('--bins_number', default=10, type=int, required=False, help='the number of groups')

parser.add_argument('--no_evaluate_loss_attack', action='store_true', required=False, help='whether to evaluate Loss Attack')
parser.add_argument('--no_evaluate_min_k_attack', action='store_true', required=False, help='whether to evaluate Min-k% Attack')
parser.add_argument('--no_evaluate_neighbour_attack', action='store_true', required=False, help='whether to evaluate Neighbour Attack')
parser.add_argument('--no_evaluate_base', action='store_true', required=False, help='whether to evaluate the Base version of reference-based attacks(LiRA and WEL-MIA)')
parser.add_argument('--no_evaluate_align', action='store_true', help='whether to evaluate the Align version of reference-based attacks(LiRA and WEL-MIA)')

parser.add_argument('--neighbour_number', default=10, type=int, required=False, help='the number of neighbours in Neighbour Attack')
parser.add_argument('--k', default=0.2, type=float, required=False, help='the parameter k in Min-k% Attack')

parser.add_argument('--packed_data', default='data/packed_data', type=str, required=False, help='path of the packed data')
parser.add_argument('--paraphrased_data', default='data/paraphrased_data', type=str, required=False, help='options: no/path of the paraphrased target dataset')
parser.add_argument('--evaluated_data', default='data/evaluated_data', type=str, required=False, help='path of the evaluated data')

parser.add_argument('--save_result', default='result', type=str, required=False, help='options: no/path of the results to save')
parser.add_argument('--overwrite_result', action='store_true', help='whether to overwrite the results')
parser.add_argument('--no_showing_result', action='store_true', help='whether to show ROC curves')
parser.add_argument('--random_seed', default=12345, type=int, required=False)

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scores_dict = {}


if __name__ == "__main__":
    '''
        load the target dataset for evaluation
    '''
    dataset = Dataset.from_dict({
        'text': load_json_file(f'{args.evaluated_data}/text.json'),
        'label': load_json_file(f'{args.evaluated_data}/label.json'),
    })
    labels = dataset['label']
    data_copy = copy.deepcopy(dataset['text'])
    dataset = dataset.map(lambda x: {'text': '<|endoftext|>' + x['text']})
    data_loader = DataLoader(dataset, batch_size=args.eval_batch_size)

    '''
        get the token-level prediction probabilities in models
    '''
    # dict for saving the token-level prediction probabilities
    probabilities = {
        'target': np.ndarray((0, args.max_length)),         # `numpy.Array` of shape (target_dataset_size, max_length)
        'reference_base': np.ndarray((0, args.max_length)),     # `numpy.Array` of shape (target_dataset_size, max_length)
        'reference_align': np.ndarray((0, args.max_length)),    # `numpy.Array` of shape (target_dataset_size, max_length)
        'paraphrased': np.ndarray((0, args.max_length))     # `numpy.Array` of shape (target_dataset_size * neighbour_number, max_length)
    }
    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model, trust_remote_code=True)
    target_model.eval()
    target_model.to(device)

    # get the token-level prediction probabilities in the target model
    for batch in tqdm(data_loader, desc='running on the target model...'):
        ids_batch = torch.LongTensor(tokenizer.batch_encode_plus(batch['text'], padding='max_length', truncation='longest_first', max_length=args.max_length + 2)['input_ids']).to(device)
        ids_batch = ids_batch[:, :args.max_length + 2]
        with torch.no_grad():
            logits = target_model.forward(input_ids=ids_batch, labels=ids_batch)[1]
            confidence = torch.softmax(logits[:, :-2, :], dim=-1)
            probabilities_batch = get_sequence_probs(confidence, ids_batch[:, 1: -1])
            probabilities['target'] = np.vstack((probabilities['target'], probabilities_batch))

    if not args.no_evaluate_base:
        reference_model_base = AutoModelForCausalLM.from_pretrained(args.reference_model_base, trust_remote_code=True)
        reference_model_base.eval()
        reference_model_base.to(device)
        # get the token-level prediction probabilities in the reference model base
        for batch in tqdm(data_loader, desc='running on the reference model base...'):
            ids_batch = torch.LongTensor(
                tokenizer.batch_encode_plus(batch['text'], padding='max_length', truncation='longest_first',
                                            max_length=args.max_length + 2)['input_ids']).to(device)
            ids_batch = ids_batch[:, :args.max_length + 2]
            with torch.no_grad():
                logits = reference_model_base.forward(input_ids=ids_batch, labels=ids_batch)[1]
                confidence = torch.softmax(logits[:, :-2, :], dim=-1)
                probabilities_batch = get_sequence_probs(confidence, ids_batch[:, 1: -1])
                probabilities['reference_base'] = np.vstack((probabilities['reference_base'], probabilities_batch))

    if not args.no_evaluate_align:
        reference_model_align = AutoModelForCausalLM.from_pretrained(args.reference_model_align, trust_remote_code=True)
        reference_model_align.eval()
        reference_model_align.to(device)
        # get the token-level prediction probabilities in the reference model align
        for batch in tqdm(data_loader, desc='running on the reference model align...'):
            ids_batch = torch.LongTensor(
                tokenizer.batch_encode_plus(batch['text'], padding='max_length', truncation='longest_first', max_length=args.max_length + 2)[
                    'input_ids']).to(device)
            ids_batch = ids_batch[:, :args.max_length + 2]
            with torch.no_grad():
                logits = reference_model_align.forward(input_ids=ids_batch, labels=ids_batch)[1]
                confidence = torch.softmax(logits[:, : -2, :], dim=-1)
                probabilities_batch = get_sequence_probs(confidence, ids_batch[:, 1: -1])
                probabilities['reference_align'] = np.vstack((probabilities['reference_align'], probabilities_batch))


    if not args.no_evaluate_neighbour_attack:
        if args.paraphrased_data == 'no':
            # run paraphrase model for generating neighbours when this option is 'no'
            paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(args.paraphrase_model).to(device)
            paraphrase_tokenizer = T5Tokenizer.from_pretrained(args.paraphrase_model)
            paraphrase_model.eval()
            paraphrased_list = generate_paraphrased_text(data_copy, paraphrase_model, paraphrase_tokenizer,
                                                         args.neighbour_number, device)

        else:
            # load neighbours from file when this option is the path of paraphrased data
            paraphrased_list = load_json_file(f'{args.paraphrased_data}/paraphrased.json')

        paraphrased_dataset = Dataset.from_dict({'text': paraphrased_list})
        paraphrased_dataloader = DataLoader(paraphrased_dataset, batch_size=args.eval_batch_size)

        # get the token-level prediction probabilities of paraphrased target dataset in the target model
        for batch in tqdm(paraphrased_dataloader, desc='paraphrased text is running on the target model...'):
            ids_batch = torch.LongTensor(
                tokenizer.batch_encode_plus(batch['text'], padding='max_length', truncation='longest_first',
                                            max_length=args.max_length + 2)['input_ids']).to(device)
            ids_batch = ids_batch[:, :args.max_length + 2]
            with torch.no_grad():
                logits = target_model.forward(input_ids=ids_batch, labels=ids_batch)[1]
                confidence = torch.softmax(logits[:, :-2, :], dim=-1)
                probabilities_batch = get_sequence_probs(confidence, ids_batch[:, 1: -1])
                probabilities['paraphrased'] = np.vstack((probabilities['paraphrased'], probabilities_batch))

    '''
    calculate membership scores
    '''
    if not args.no_evaluate_loss_attack:
        scores_dict['Loss Attack'] = loss_attack(probabilities['target'])

    if not args.no_evaluate_base:
        scores_dict['LiRA-Base'] = lira(probabilities['target'], probabilities['reference_base'])
        scores_dict['Ours-Base'] = wel_mia(probabilities['target'], probabilities['reference_base'], args.bins_number)

    if not args.no_evaluate_align:
        scores_dict['LiRA-Align'] = lira(probabilities['target'], probabilities['reference_align'])
        scores_dict['Ours-Align'] = wel_mia(probabilities['target'], probabilities['reference_align'], args.bins_number)

    if not args.no_evaluate_min_k_attack:
        scores_dict['Min-K% Attack'] = min_k(probabilities['target'], args.k)

    if not args.no_evaluate_neighbour_attack:
        scores_dict['Neighbor Attack'] = neighbour_attack(probabilities['target'], probabilities['paraphrased'], args.neighbour_number)

    '''
    show and save_results
    '''
    add_directory(f'{args.save_result}/scores')
    add_directory(f'{args.save_result}/roc_curves')
    add_directory(f'{args.save_result}/metrics')
    metrics = []
    plt.figure(figsize=(12, 5))
    for attack_name, scores in scores_dict.items():
        plt.subplot(1, 2, 1)
        plt, metric = draw_roc_curves(labels, scores, plt, desc=attack_name, scale='log')
        plt.subplot(1, 2, 2)
        plt, _ = draw_roc_curves(labels, scores, plt, desc=attack_name, scale='linear')
        result = {
            'score': scores.tolist(),
            'label': labels,
        }
        metrics.append({
            'attack_name': attack_name,
            'metric': metric
        })
        if args.save_result != 'no':
            if not (os.path.exists(f'{args.save_result}/scores/{attack_name}.json') and not args.overwrite_result):
                save_data_in_json_file(result, f'{args.save_result}/scores/{attack_name}.json')

    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], color='brown', ls='--')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot([0, 1], [0, 1], color='brown', ls='--')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend()
    if args.save_result != 'no':
        if not (os.path.exists(f'{args.save_result}/roc_curves/roc.png') and not args.overwrite_result):
            plt.savefig(f'{args.save_result}/roc_curves/roc.png')
        if not (os.path.exists(f'{args.save_result}/metrics/metrics.json') and not args.overwrite_result):
            save_data_in_json_file(metrics, f'{args.save_result}/metrics/metrics.json')

    if not args.no_showing_result:
        plt.show()



