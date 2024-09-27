## Not All Tokens Are Equal: Membership Inference Attacks Against Fine-tuned Language Models

The implementation of the paper "Not All Tokens Are Equal: Membership Inference Attacks Against Fine-tuned Language Models".

### Dependencies and Environment
#### Dependencies

```
datasets>=2.21.0
evaluate>=0.4.1
transformers>=4.28.0
scikit-learn>=1.3.2
numpy>=1.26.3
tqdm>=4.66.5
scipy>=1.11.4
matplotlib>=3.8.2
torch>=2.1.2
accelerate>=0.21.0
sentencepiece>=0.2.0
pluggy>=0.12.0
```

Dependencies can be installed with the command:

```
pip install -r requirements.txt
```
#### System environment
- OS: `Ubuntu 22.04.2 LTS`
- CUDA Version: `12.4`
- Python Version: `3.9.19`


### WEL-MIA

#### Prepare data

Run prepare.py for preparing data. `--packing_data` option means to pack raw data into fixed length, and `max_number` is the size of sampled non-member records, `gamma` is the ratio of members to non-members in the target dataset. Use `--generating_neighbours` option to generate and save neighbours of the target dataset for reducing time consumption of Neighbour Attack.

```sh
python prepare.py --dataset iohadrubin/wikitext-103-raw-v1 --tokenizer_path openai-community/gpt2 --paraphrase_model Vamsi/T5_Paraphrase_Paws --max_length 128 --member_space_size 10000 --non_member_space_size 10000 --max_number 1000 --gamma 1 --save_packed_data data/packed_data --packing_data --generating_neighbours
```

#### Finetune LMs

The default model is GPT-2-Base. Finetune the target model:

```sh
python ./finetune/run_clm.py --model_name_or_path openai-community/gpt2 --train_file data/packed_data/member.json --validation_file data/packed_data/validation.json --save_strategy no --num_train_epochs 2 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --do_train --do_eval --output_dir model/target_model --overwrite_output_dir --learning_rate 2e-5
```

In our Align version attack, we use the  target dataset for finetuning the reference model:

```sh
python ./finetune/run_clm.py --model_name_or_path openai-community/gpt2 --train_file data/evaluated_data/text.json --validation_file data/packed_data/validation.json --save_strategy no --num_train_epochs 2 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --do_train --do_eval --output_dir model/reference_model_align --overwrite_output_dir --learning_rate 2e-5
```
The recommended learning rate on`['openai-community/gpt2', 'openai-community/gpt2-medium', 'EleutherAI/gpt-neo-125m']` is 2e-5, on `['openai-community/gpt2-large', 'EleutherAI/pythia-160m']` is 5e-6.  All training is done for 2 epochs. Larger models are better suited to smaller learning rates to reduce overfitting.

#### Evaluate MIAs

```sh
python evaluate.py --target_model model/target_model --reference_model_base openai-community/gpt2 --reference_model_align model/reference_model_align --paraphrase_model Vamsi/T5_Paraphrase_Paws --overwrite_result 
```

All datasets and pretrained models can be downloaded in hugging face (https://huggingface.co/).

### Run
Run MIAs on a single model and dataset from scratch. Use `--dataset_name` option to specify the dataset and `--model_name` option to specify the model.
```sh
python run.py
```

Systematically run MIAs on multiple models and datasets. The datasets are `['knkarthick/xsum', 'SetFit/ag_news', 'iohadrubin/wikitext-103-raw-v1']`. The models are `['openai-community/gpt2', 'openai-community/gpt2-medium', 'openai-community/gpt2-large', 'EleutherAI/pythia-160m', 'EleutherAI/gpt-neo-125m']`. There are 15 experiments in total, and at least 40 GB of storage space is required to save all models and data. If you don't want to save models, use the `--overwrite_models` option. The results of each set of experiments are saved in `result/dataset_name-model_name`.

```sh
python run_multiple.py --member_space_size 10000 --non_member_space_size 10000 --max_number 1000 
```

To run on specific gpus, you can set CUDA_VISIBLE_DEVICES for the python scripts above.
For example:
```sh
CUDA_VISIBLE_DEVICES=1 python run.py
```


### Resources

Running `script.sh` on two RTX 4090, the resource consumption of models of different sizes is as follows:

|                | gpt2-base | gpt2-medium | gpt2-large |
| :------------: | :-------: | :---------: | :--------: |
|   VRAM (GB)    |   12.4    |    23.2     |    30.3    |
|  storage (GB)  |    1.0    |     2.8     |    6.2     |
| time (minutes) |   22.1    |    31.5     |    54.0    |

To save storage, use `--save_strategy no` to ignore saving checkpoints.

