import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', default=128, type=int, required=False, help='the packed data length')
parser.add_argument('--overwrite_models', action='store_true', help='whether to store models')
parser.add_argument('--overwrite_data', action='store_true', help='whether to store data')
parser.add_argument('--save_strategy', default='no', type=str, help='save strategy of checkpoints')
parser.add_argument('--member_space_size', default=10000, type=int, required=False, help='the number of all packed members')
parser.add_argument('--non_member_space_size', default=10000, type=int, required=False, help='the number of all packed non-members')
parser.add_argument('--max_number', default=1000, type=int, required=False, help='the number of evaluated non-member samples')
parser.add_argument('--gamma', default=1.0, type=float, required=False, help='the ratio of members to non-members')
parser.add_argument('--neighbour_number', default=10, type=int, required=False, help='the number of neighbours in Neighbour Attack')
parser.add_argument('--paraphrase_model', default='Vamsi/T5_Paraphrase_Paws', type=str, required=False, help='path of the align paraphrase model')
args = parser.parse_args()


if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
    cuda_option = 'CUDA_VISIBLE_DEVICES=' + os.environ['CUDA_VISIBLE_DEVICES'] + ' '
else:
    cuda_option = ''


'''
    Datasets for evaluation
'''
datasets_list = [
    'knkarthick/xsum',
    'SetFit/ag_news',
    'iohadrubin/wikitext-103-raw-v1',
]


'''
    Models for evaluation
'''
models_dict = {
    'openai-community/gpt2': {
        'batch_size': 2,
        'num_train_epochs': 2,
        'learning_rate': 2e-5
    },
    'openai-community/gpt2-medium': {
        'batch_size': 2,
        'num_train_epochs': 2,
        'learning_rate': 2e-5
    },
    'openai-community/gpt2-large': {
        'batch_size': 1,
        # 'batch_size': 2,
        'num_train_epochs': 2,
        'learning_rate': 5e-6
    },
    'EleutherAI/pythia-160m': {
        'batch_size': 2,
        'num_train_epochs': 2,
        'learning_rate': 5e-6
    },
    'EleutherAI/gpt-neo-125m': {
        'batch_size': 2,
        'num_train_epochs': 2,
        'learning_rate': 2e-5
    }
}

for dataset_name in datasets_list:
    data_path = 'data' if args.overwrite_data else 'data/' + dataset_name.split('/')[-1]

    # split dataset
    script_prepare = f"{cuda_option} python prepare.py --dataset {dataset_name} " \
                     f"--tokenizer_path openai-community/gpt2 " \
                     f"--max_length {args.max_length} " \
                     f"--member_space_size {args.member_space_size} " \
                     f"--non_member_space_size {args.non_member_space_size} " \
                     f"--max_number {args.max_number} " \
                     f"--gamma {args.gamma} " \
                     f"--save_packed_data {data_path}/packed_data " \
                     f"--evaluated_data {data_path}/evaluated_data " \
                     f"--paraphrased_data {data_path}/paraphrased_data " \
                     f"--paraphrase_model {args.paraphrase_model} " \
                     f"--neighbour_number {args.neighbour_number} " \
                     f"--packing_data " \
                     f"--generating_neighbours "
    os.system(script_prepare)

    for model_name in models_dict.keys():
        hyperparams = models_dict[model_name]
        batch_size = hyperparams['batch_size']
        learning_rate = hyperparams['learning_rate']
        num_train_epochs = hyperparams['num_train_epochs']
        print('#' * 100)
        print(f'Evaluating on {dataset_name}, {model_name}')
        model_path = 'model' if args.overwrite_models else 'model/' + dataset_name.split('/')[-1] +\
                                                           '-' + model_name.split('/')[-1]

        # finetune target model
        script_ft_target = f"{cuda_option} python finetune/run_clm.py " \
                           f"--model_name_or_path {model_name} " \
                           f"--train_file {data_path}/packed_data/member.json " \
                           f"--validation_file {data_path}/packed_data/validation.json " \
                           f"--save_strategy {args.save_strategy} " \
                           f"--num_train_epochs {num_train_epochs} " \
                           f"--per_device_train_batch_size {batch_size} " \
                           f"--per_device_eval_batch_size {batch_size} " \
                           f"--do_train --do_eval " \
                           f"--output_dir {model_path}/target_model " \
                           f"--overwrite_output_dir " \
                           f"--learning_rate {learning_rate} " \
                           f"--seed 12345 "
        os.system(script_ft_target)


        # finetune reference model-align
        script_ft_reference = f"{cuda_option} python finetune/run_clm.py " \
                              f"--model_name_or_path {model_name} " \
                              f"--train_file {data_path}/evaluated_data/text.json " \
                              f"--validation_file {data_path}/packed_data/validation.json " \
                              f"--save_strategy {args.save_strategy} " \
                              f"--num_train_epochs {num_train_epochs} " \
                              f"--per_device_train_batch_size {batch_size} " \
                              f"--per_device_eval_batch_size {batch_size} " \
                              f"--do_train --do_eval " \
                              f"--output_dir {model_path}/reference_model_align " \
                              f"--overwrite_output_dir " \
                              f"--learning_rate {learning_rate} " \
                              f"--seed 12345 "
        os.system(script_ft_reference)


        result_path = 'result/' + dataset_name.split('/')[-1] + '-' + model_name.split('/')[-1]
        # evaluate membership inference attacks
        script_evaluation = f"{cuda_option} python evaluate.py --target_model {model_path}/target_model " \
                            f"--reference_model_base {model_name} " \
                            f"--reference_model_align {model_path}/reference_model_align " \
                            f"--paraphrase_model {args.paraphrase_model} " \
                            f"--tokenizer_path {model_path}/target_model " \
                            f"--packed_data {data_path}/packed_data " \
                            f"--paraphrased_data {data_path}/paraphrased_data " \
                            f"--evaluated_data {data_path}/evaluated_data " \
                            f"--save_result {result_path} " \
                            f"--overwrite_result " \
                            f"--no_showing_result "
        os.system(script_evaluation)









