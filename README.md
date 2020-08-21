# L2R2

PyTorch implementation of L2R2: Leveraging Ranking for Abductive Reasoning.

## Usage

### Set up environment

L2R2 is tested on Python 3.6 and PyTorch 1.0.1.

```shell script
$ pip install -r requirements.txt
```

### Prepare data

[αNLI](https://leaderboard.allenai.org/anli/submissions/get-started)
```shell script
$ wget https://storage.googleapis.com/ai2-mosaic/public/alphanli/alphanli-train-dev.zip
$ unzip -d alphanli alphanli-train-dev.zip
```

### Training

We train the L2R2 models on 4 K80 GPUs. The appropriate batch size on each K80 is 1, so the batch size in our experiment is 4.

The available `criterion` for optimization could selected in:
- list_net: list-wise *KLD* loss used in ListNet
- list_mle: list-wise *Likelihood* loss used in ListMLE
- approx_ndcg: list-wise *ApproxNDCG* loss used in ApproxNDCG
- rank_net: pair-wise *Logistic* loss used in RankNet
- hinge: pair-wise *Hinge* loss used in Ranking SVM
- lambda: pair-wise *LambdaRank* loss used in LambdaRank

Note that in our experiment, we manually reduce the learning rate instead of using any automatic learning rate scheduler.

For example, we first fine-tune the pre-trained RoBERTa-large model for up to 10 epochs with a learning rate of 5e-6 and save the model checkpoint which performs best on the dev set.
```shell script
$ python run.py \
  --data_dir=alphanli/ \
  --output_dir=ckpts/ \
  --model_type='roberta' \
  --model_name_or_path='roberta-large' \
  --linear_dropout_prob=0.6 \
  --max_hyp_num=22 \
  --tt_max_hyp_num=22 \
  --max_seq_len=72 \
  --do_train \
  --do_eval \
  --criterion='list_net' \
  --per_gpu_train_batch_size=1 \
  --per_gpu_eval_batch_size=1 \
  --learning_rate=5e-6 \
  --weight_decay=0.0 \
  --num_train_epochs=10 \
  --seed=42 \
  --log_period=50 \
  --eval_period=100 \
  --overwrite_output_dir
```

Then, we continue to fine-tune the just saved model for up to 3 epochs with a smaller learning rate, such as 3e-6, 1e-6 and 5e-7, until the performance on the dev set is no longer improved.
```shell script
python run.py \
  --data_dir=alphanli/ \
  --output_dir=ckpts/ \
  --model_type='roberta' \
  --model_name_or_path=ckpts/H22_L72_E3_B4_LR5e-06_WD0.0_MMddhhmmss/checkpoint-best_acc/ \
  --linear_dropout_prob=0.6 \
  --max_hyp_num=22 \
  --tt_max_hyp_num=22 \
  --max_seq_len=72 \
  --do_train \
  --do_eval \
  --criterion='list_net' \
  --per_gpu_train_batch_size=1 \
  --per_gpu_eval_batch_size=1 \
  --learning_rate=1e-6 \
  --weight_decay=0.0 \
  --num_train_epochs=3 \
  --seed=43 \
  --log_period=50 \
  --eval_period=100 \
  --overwrite_output_dir
```
Note: change the seed to reshuffle training samples.

### Evaluation

Evaluate the performance on the dev set.
```shell script
$ export MODEL_DIR="ckpts/H22_L72_E3_B4_LR5e-07_WD0.0_MMddhhmmss/checkpoint-best_acc/"
$ python run.py \
  --data_dir=alphanli/ \
  --output_dir=$MODEL_DIR \
  --model_type='roberta' \
  --model_name_or_path=$MODEL_DIR \
  --max_hyp_num=2 \
  --max_seq_len=72 \
  --do_eval \
  --per_gpu_eval_batch_size=1
```

### Inference
```shell script
$ ./run_model.sh
```

## Citation
```
@inproceedings{10.1145/3397271.3401332,
  author = {Zhu, Yunchang and Pang, Liang and Lan, Yanyan and Cheng, Xueqi},
  title = {L2R²: Leveraging Ranking for Abductive Reasoning},
  year = {2020},
  url = {https://doi.org/10.1145/3397271.3401332},
  doi = {10.1145/3397271.3401332},
  booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series = {SIGIR '20}
}
```

