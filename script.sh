# split dataset
python prepare.py --dataset knkarthick/xsum --tokenizer_path openai-community/gpt2-medium --paraphrase_model Vamsi/T5_Paraphrase_Paws --max_length 128 --member_space_size 100 --non_member_space_size 100 --max_number 100 --gamma 1 --save_packed_data data/packed_data --packing_data --generating_neighbours

# finetune target model
python ./finetune/run_clm.py --model_name_or_path openai-community/gpt2-medium --train_file data/packed_data/member.json --validation_file data/packed_data/validation.json --save_strategy no --num_train_epochs 2 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --do_train --do_eval --output_dir model/target_model --overwrite_output_dir --learning_rate 2e-5

# finetune reference model-align
python ./finetune/run_clm.py --model_name_or_path openai-community/gpt2-medium --train_file data/evaluated_data/text.json --validation_file data/packed_data/validation.json --save_strategy no --num_train_epochs 2 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --do_train --do_eval --output_dir model/reference_model_align --overwrite_output_dir --learning_rate 2e-5

# evaluate membership inference attacks
python evaluate.py --target_model model/target_model --reference_model_base openai-community/gpt2-medium --reference_model_align model/reference_model_align --paraphrase_model Vamsi/T5_Paraphrase_Paws --overwrite_result

