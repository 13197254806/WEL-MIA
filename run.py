import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='iohadrubin/wikitext-103-raw-v1', type=str, required=False, help='the dataset name in demo')
parser.add_argument('--model_name', default='openai-community/gpt2', type=str, required=False, help='the model name in demo')
parser.add_argument('--data_path', default='data', type=str, required=False, help='the directory to save data')
parser.add_argument('--model_path', default='model', type=str, required=False, help='the directory to save models')
parser.add_argument('--result_path', default='result', type=str, required=False, help='the directory to save results')

parser.add_argument('--batch_size', default=2, type=int, required=False, help='training batch size of target/reference model')
parser.add_argument('--learning_rate', default=2e-5, type=float, required=False, help='learning rate of target/reference model')
parser.add_argument('--num_train_epochs', default=2, type=int, required=False, help='training epochs of target/reference model')

parser.add_argument('--max_length', default=128, type=int, required=False, help='the packed data length')
parser.add_argument('--member_space_size', default=10000, type=int, required=False, help='the number of all packed members')
parser.add_argument('--non_member_space_size', default=10000, type=int, required=False, help='the number of all packed non-members')
parser.add_argument('--max_number', default=1000, type=int, required=False, help='the number of evaluated non-member samples')
parser.add_argument('--gamma', default=1.0, type=float, required=False, help='the ratio of members to non-members')
parser.add_argument('--neighbour_number', default=10, type=int, required=False, help='the number of neighbours in Neighbour Attack')
parser.add_argument('--paraphrase_model', default='Vamsi/T5_Paraphrase_Paws', type=str, required=False, help='path of the align paraphrase model')

parser.add_argument('--overwrite_models', action='store_true', help='whether to store models')
parser.add_argument('--overwrite_data', action='store_true', help='whether to store data')
parser.add_argument('--save_strategy', default='no', type=str, help='save strategy of checkpoints')

args = parser.parse_args()
print('args:\n' + args.__repr__())

if 'CUDA_VISIBLE_DEVICES' in os.environ.keys():
    cuda_option = 'CUDA_VISIBLE_DEVICES=' + os.environ['CUDA_VISIBLE_DEVICES'] + ' '
else:
    cuda_option = ''

# split dataset
script_prepare = f"{cuda_option} python prepare.py --dataset {args.dataset_name} " \
                 f"--tokenizer_path {args.model_name} " \
                 f"--max_length {args.max_length} " \
                 f"--member_space_size {args.member_space_size} " \
                 f"--non_member_space_size {args.non_member_space_size} " \
                 f"--max_number {args.max_number} " \
                 f"--gamma {args.gamma} " \
                 f"--save_packed_data {args.data_path}/packed_data " \
                 f"--evaluated_data {args.data_path}/evaluated_data " \
                 f"--paraphrased_data {args.data_path}/paraphrased_data " \
                 f"--paraphrase_model {args.paraphrase_model} " \
                 f"--neighbour_number {args.neighbour_number} " \
                 f"--packing_data " \
                 f"--generating_neighbours "
os.system(script_prepare)


# finetune target model
script_ft_target = f"{cuda_option} python finetune/run_clm.py " \
                   f"--model_name_or_path {args.model_name} " \
                   f"--train_file {args.data_path}/packed_data/member.json " \
                   f"--validation_file {args.data_path}/packed_data/validation.json " \
                   f"--save_strategy {args.save_strategy} " \
                   f"--num_train_epochs {args.num_train_epochs} " \
                   f"--per_device_train_batch_size {args.batch_size} " \
                   f"--per_device_eval_batch_size {args.batch_size} " \
                   f"--do_train --do_eval " \
                   f"--output_dir {args.model_path}/target_model " \
                   f"--overwrite_output_dir " \
                   f"--learning_rate {args.learning_rate} " \
                   f"--seed 12345 "
os.system(script_ft_target)


# finetune reference model-align
script_ft_reference = f"{cuda_option} python finetune/run_clm.py " \
                      f"--model_name_or_path {args.model_name} " \
                      f"--train_file {args.data_path}/evaluated_data/text.json " \
                      f"--validation_file {args.data_path}/packed_data/validation.json " \
                      f"--save_strategy {args.save_strategy} " \
                      f"--num_train_epochs {args.num_train_epochs} " \
                      f"--per_device_train_batch_size {args.batch_size} " \
                      f"--per_device_eval_batch_size {args.batch_size} " \
                      f"--do_train --do_eval " \
                      f"--output_dir {args.model_path}/reference_model_align " \
                      f"--overwrite_output_dir " \
                      f"--learning_rate {args.learning_rate} " \
                      f"--seed 12345 "
os.system(script_ft_reference)


# evaluate membership inference attacks
script_evaluation = f"{cuda_option} python evaluate.py --target_model {args.model_path}/target_model " \
                    f"--reference_model_base {args.model_name} " \
                    f"--reference_model_align {args.model_path}/reference_model_align " \
                    f"--paraphrase_model {args.paraphrase_model} " \
                    f"--tokenizer_path {args.model_path}/target_model " \
                    f"--packed_data {args.data_path}/packed_data " \
                    f"--paraphrased_data {args.data_path}/paraphrased_data " \
                    f"--evaluated_data {args.data_path}/evaluated_data " \
                    f"--save_result {args.result_path} " \
                    f"--overwrite_result " \
                    f"--no_showing_result "
os.system(script_evaluation)









