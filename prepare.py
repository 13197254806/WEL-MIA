import json
import os
from tqdm import tqdm
import argparse
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, T5Tokenizer, AutoModelForSeq2SeqLM
from utils import *


def pack_raw_data(tokenizer, raw_data_path, save_path, member_number, non_member_number, max_length, seed):
    """
    Packing raw text dataset into fixed length, then splitting packed dataset into members and non-members.

    Parameters:
        tokenizer: transformers.AutoTokenizer.
        raw_data_path: str, local dir or dataset name in hugging face.
        save_path: str, local dir to save members and non-members.
        member_number:int, the size of member space.
        non_member_number:int, the size of non-member space.
        max_length:int, the packed text length.
        seed: int, random seed.

    Returns:
        None.
    """
    full_dataset = load_dataset(raw_data_path)
    raw_dataset = full_dataset['train'].shuffle(seed=seed)
    test_dataset = full_dataset['test'].shuffle(seed=seed)
    column_name = ''

    if 'text' in raw_dataset.features.keys():
        column_name = 'text'
    elif 'data' in raw_dataset.features.keys():
        column_name = 'data'
    elif 'document' in raw_dataset.features.keys():
        column_name = 'document'
    elif 'dialogue' in raw_dataset.features.keys():
        column_name = 'dialogue'

    all_tokens = []
    packed_data = []
    count_token = 0
    for record in tqdm(raw_dataset[column_name], desc='spilting raw data'):
        tmp = tokenizer.tokenize(str(record))
        all_tokens.extend(tmp)
        count_token += len(tmp)
        if count_token >= max_length * (member_number + non_member_number):
            break
    packed_number = len(all_tokens) // max_length
    for cnt in tqdm(range(packed_number), desc='packing raw data'):
        ids = tokenizer.convert_tokens_to_ids(all_tokens[int(cnt * max_length): int((cnt + 1) * max_length)])
        packed_data.append(tokenizer.decode(ids))


    members, non_members = packed_data[: member_number], packed_data[member_number: member_number + non_member_number]
    with open(save_path + '/member.json', 'w', encoding='utf-8') as f:
        json.dump(members, f, indent=2)
    with open(save_path + '/non_member.json', 'w', encoding='utf-8') as f:
        json.dump(non_members, f, indent=2)

    all_tokens = []
    packed_data = []
    for record in test_dataset[column_name]:
        all_tokens.extend(tokenizer.tokenize(str(record)))
    packed_number = len(all_tokens) // max_length
    for cnt in range(min(packed_number, 500)):
        ids = tokenizer.convert_tokens_to_ids(all_tokens[int(cnt * max_length): int((cnt + 1) * max_length)])
        packed_data.append(tokenizer.decode(ids))
    with open(save_path + '/validation.json', 'w', encoding='utf-8') as f:
        json.dump(packed_data, f, indent=2)


def load_full_packed_data(data_dir):
    """
    Load packed dataset (members and non-members) from dir.

    Parameters:
        data_dir: str, local dir of packed dataset.

    Returns:
        datasets.Dataset.
    """
    data_list = []
    with open(data_dir + '/member.json', 'r', encoding='utf-8') as f:
        data_list.extend(json.load(f))
        member_number = len(data_list)
    with open(data_dir + '/non_member.json', 'r', encoding='utf-8') as f:
        data_list.extend(json.load(f))
        non_member_number = len(data_list) - member_number
    label_list = [1] * member_number
    label_list.extend([0] * non_member_number)
    dataset_dict = {
        'text': data_list,
        'label': label_list
    }
    return Dataset.from_dict(dataset_dict)


def random_sampling(full_dataset, max_nums, gamma, seed):
    """
    Randomly sample members and non-members in full packed dataset for evaluation.

    Parameters:
        full_dataset: datasets.Dataset, The full packed dataset.
        max_nums: int, number of sampled non-members.
        gamma: float, the ratio of the members to non-members.
        seed: int, random seed.

    Returns:
        datasets.Dataset.
    """
    members = full_dataset.filter(lambda x: x['label'] == 1)
    non_members = full_dataset.filter(lambda x: x['label'] == 0)
    sampled_members = members.shuffle(seed).select(range(min(round(max_nums * gamma), len(members))))
    sampled_non_members = non_members.shuffle(seed).select(range(min(max_nums, len(non_members))))
    return concatenate_datasets([sampled_members, sampled_non_members])




parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='knkarthick/xsum', type=str, required=False, help='the raw dataset name')
parser.add_argument('--tokenizer_path', default='openai-community/gpt2', type=str, required=False, help='the tokenizer path')
parser.add_argument('--max_length', default=128, type=int, required=False, help='the packed length')
parser.add_argument('--member_space_size', default=10000, type=int, required=False, help='the number of all packed members')
parser.add_argument('--non_member_space_size', default=10000, type=int, required=False, help='the number of all packed non-members')
parser.add_argument('--save_packed_data', default='data/packed_data', type=str, required=False, help='the save path of packed data')
parser.add_argument('--random_seed', default=12345, type=int, required=False)
parser.add_argument('--packing_data', action='store_true', help='whether to pack raw data')
parser.add_argument('--evaluated_data', default='data/evaluated_data', type=str, required=False, help='no/saved_path')
parser.add_argument('--paraphrased_data', default='data/paraphrased_data', type=str, required=False, help='path of the paraphrased target dataset')

parser.add_argument('--generating_neighbours', action='store_true', help='whether to generate neighbours')
parser.add_argument('--neighbour_number', default=10, type=int, required=False, help='the number of neighbours in Neighbour Attack')
parser.add_argument('--paraphrase_model', default='model/paraphrase_model', type=str, required=False, help='path of the align paraphrase model')



parser.add_argument('--max_number', default=1000, type=int, required=False, help='the number of evaluated non-member samples')
parser.add_argument('--gamma', default=1.0, type=float, required=False, help='the ratio of members to non-members')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    if args.packing_data:
        # packing raw data
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        add_directory(args.save_packed_data)
        pack_raw_data(tokenizer,
                      args.dataset,
                      args.save_packed_data,
                      member_number=args.member_space_size,
                      non_member_number=args.non_member_space_size,
                      max_length=args.max_length,
                      seed=args.random_seed)

    # load full packed dataset and randomly sample target dataset
    packed_dataset = load_full_packed_data(args.save_packed_data)
    dataset = random_sampling(packed_dataset, args.max_number, args.gamma, args.random_seed)
    add_directory(args.evaluated_data)
    save_data_in_json_file(dataset['text'], f'{args.evaluated_data}/text.json')
    save_data_in_json_file(dataset['label'], f'{args.evaluated_data}/label.json')

    if args.generating_neighbours:
        # generating neighbours of evaluated dataset and saving neighbours in json files
        paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(args.paraphrase_model).to(device)
        paraphrase_tokenizer = T5Tokenizer.from_pretrained(args.paraphrase_model)
        paraphrased_list = generate_paraphrased_text(dataset['text'], paraphrase_model, paraphrase_tokenizer,
                                                 args.neighbour_number, device)
        add_directory(args.paraphrased_data)
        save_data_in_json_file(paraphrased_list, args.paraphrased_data + '/paraphrased.json')





