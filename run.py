# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import copy
from datetime import datetime
import json
import logging
import os
import random
import shutil
import glob
from tqdm import tqdm, trange

import numpy as np
import torch
from torch import optim
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Subset)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import AdamW  # , WarmupLinearSchedule
from transformers import WEIGHTS_NAME

from transformers import (BertConfig, BertTokenizer,
                          RobertaConfig, RobertaTokenizer)

from model import BertForListRank, RobertaForListRank
from losses import list_mle, list_net, approx_ndcg_loss, rank_net, pairwise_hinge, lambda_loss
from eval import eval_file
from data_process import AlphaNliProcessor, StoryFeatures, AlphaNliDataset
from utils import static_vars, cal_losses, RawResult, infer_labels, write_results

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s %(levelname)s %(name)s:%(lineno)s] %(message)s',
                              datefmt='%m/%d %H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)),
    ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForListRank, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForListRank, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    exp_dir = 'H%d_L%d_E%d_B%d_LR%s_WD%s_%s' % (args.max_hyp_num, args.max_seq_len, args.num_train_epochs,
                                                args.train_batch_size, args.learning_rate, args.weight_decay,
                                                datetime.now().strftime('%m%d%H%M'))
    args.output_dir = os.path.join(args.output_dir, exp_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        arg_dict = copy.deepcopy(args.__dict__)
        arg_dict['device'] = str(args.device)
        json.dump(arg_dict, f, indent=2)

    os.mkdir(os.path.join(args.output_dir, 'src'))
    for src_file in ['model.py', 'losses.py', 'run.py']:
        dst_file = os.path.join(args.output_dir, 'src', os.path.basename(src_file))
        shutil.copyfile(src_file, dst_file)

    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    train_dataset = load_dataset(args, tokenizer, mode='train')
    train_sampler = RandomSampler(train_dataset)
    data_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=16)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(data_loader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(data_loader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = optim.SGD(optimizer_grouped_parameters, lr=args.learning_rate)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if not args.no_cuda and args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num stories = %d", len(train_dataset))
    logger.info("  Num epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Criterion = %s", args.criterion)
    logger.info("  Learning rate = %s", args.learning_rate)

    tb_writer = SummaryWriter(os.path.join('runs/', exp_dir))

    global_step = 0
    best_acc, best_step = 0, 0
    keys = ['list_mle', 'list_net', 'approx_ndcg', 'rank_net', 'hinge', 'lambda']
    losses = dict.fromkeys(keys, 0.0)
    last_losses = losses.copy()
    model.zero_grad()
    epoch_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for epoch in epoch_iterator:
        batch_iterator = tqdm(data_loader, desc="Iteration")
        for step, batch in enumerate(batch_iterator):
            model.train()
            batch = tuple(t.to(args.device) if torch.is_tensor(t) else t for t in batch)
            x = {'input_ids': batch[0], 'token_type_ids': batch[1], 'attention_mask': batch[2]}
            # (batch_size, list_len)
            logits = model(**x)
            labels = batch[3]

            _losses = dict()
            _losses['list_mle'] = list_mle(logits, labels)
            _losses['list_net'] = list_net(logits, labels)
            _losses['approx_ndcg'] = approx_ndcg_loss(logits, labels)
            _losses['rank_net'] = rank_net(logits, labels)
            _losses['hinge'] = pairwise_hinge(logits, labels)
            _losses['lambda'] = lambda_loss(logits, labels)

            if args.n_gpu > 1:
                # mean() to average on multi-gpu parallel (not distributed) training
                for k, v in _losses.items():
                    _losses[k] = v.mean()
            if args.gradient_accumulation_steps > 1:
                for k in _losses.keys():
                    _losses[k] /= args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(_losses[args.criterion], optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                _losses[args.criterion].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            for k in losses.keys():
                losses[k] += _losses[k].item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log losses
                if args.log_period > 0 and global_step % args.log_period == 0:
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    for k in losses:
                        tb_writer.add_scalar('loss/' + k, (losses[k] - last_losses[k]) / args.log_period, global_step)
                    last_losses = losses.copy()

                # Log metrics
                if args.eval_period > 0 and global_step % args.eval_period == 0:
                    metrics, dev_losses = evaluate(args, model, tokenizer,
                                                   prefix='%d-%d' % (epoch, global_step), partition=1)
                    for k, v in metrics.items():
                        tb_writer.add_scalar('metrics_dev/' + k, v, global_step)
                    for k, v in dev_losses.items():
                        tb_writer.add_scalar('loss_dev/' + k, v, global_step)
                    if metrics['accuracy'] > best_acc:
                        best_acc = metrics['accuracy']
                        best_step = global_step
                        logger.info("  Achieve best accuracy: %.2f", best_acc * 100)
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best_acc')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        write_results('step: %d' % best_step,
                                      metrics, dev_losses, os.path.join(output_dir, "dev-eval.txt"))
                        shutil.copyfile(os.path.join(args.output_dir, 'raw_dev.pkl'),
                                        os.path.join(output_dir, 'raw_dev.pkl'))
                        shutil.copyfile(os.path.join(args.output_dir, 'dev-pred.lst'),
                                        os.path.join(output_dir, 'dev-pred.lst'))

                # Save model checkpoint
                if args.save_period > 0 and global_step % args.save_period == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    if global_step % args.eval_period == 0:
                        write_results('step: %d' % global_step,
                                      metrics, dev_losses, os.path.join(output_dir, "dev-eval.txt"))

            batch_iterator.set_description('Iteration(loss=%.4f)' % _losses[args.criterion].item())
            if 0 < args.max_steps < global_step:  # stop_train or
                batch_iterator.close()
                break
        if 0 < args.max_steps < global_step:  # stop_train or
            epoch_iterator.close()
            break

    tb_writer.close()

    logger.info(" global_step = %s, average loss = %s", global_step, losses[args.criterion] / global_step)
    logger.info("achieve best accuracy: %.2f at step %s", best_acc * 100, best_step)

    if args.save_period > 0:
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of parallel training
        model_to_save.save_pretrained(os.path.join(args.output_dir, 'checkpoint-final'))
        tokenizer.save_pretrained(os.path.join(args.output_dir, 'checkpoint-final'))

    # logger.removeHandler(file_handler)

    return global_step, losses[args.criterion] / global_step


@static_vars(all_gold_samples=None)
def evaluate(args, model, tokenizer, prefix="", partition=None):
    if evaluate.all_gold_samples is None:
        evaluate.all_gold_samples = AlphaNliProcessor.get_samples(os.path.join(args.data_dir, 'dev.jsonl'),
                                                                  os.path.join(args.data_dir, 'dev-labels.lst'))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'eval_log.txt'))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logging.getLogger("utils").setLevel(logging.DEBUG)
    logging.getLogger("utils").addHandler(file_handler)

    raw_file = os.path.join(args.output_dir, "raw_dev.pkl")
    dataset, id2example, id2feature = load_dataset(args, tokenizer, mode='dev', partition=partition)
    if os.path.exists(raw_file) and partition is None:
        logger.info('Loading raw results')
        raw_results = torch.load(raw_file)
    else:
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=16)

        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num features = %d", len(dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        raw_results = []
        for batch in tqdm(data_loader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) if torch.is_tensor(t) else t for t in batch)
            with torch.no_grad():
                x = {'input_ids': batch[0], 'token_type_ids': batch[1], 'attention_mask': batch[2]}
                # (batch_size, list_len)
                logits = model(**x)

            feature_ids = batch[-1]
            for i, f_id in enumerate(feature_ids):
                raw_result = RawResult(id=f_id, logits=logits[i].detach().cpu())
                raw_results.append(raw_result)
        if partition in (None, 1):
            torch.save(raw_results, raw_file)

    # Compute losses
    losses = cal_losses(raw_results, id2feature)

    # Compute predictions
    pred_file = os.path.join(args.output_dir, "dev-pred.lst")
    score_file = os.path.join(args.output_dir, "dev-score.csv")
    labels = infer_labels(evaluate.all_gold_samples, raw_results, id2example, pred_file, score_file)

    metrics = eval_file(pred_file, os.path.join(args.data_dir, 'dev-labels.lst'))
    logger.info('  Accuracy: %.2f', metrics['accuracy'] * 100)

    logger.removeHandler(file_handler)
    logging.getLogger("utils").removeHandler(file_handler)

    return metrics, losses


@static_vars(cached_id2example=dict(), cached_dataset=dict())
def load_dataset(args, tokenizer, mode='train', partition=None):
    if mode in load_dataset.cached_dataset:
        id2example = load_dataset.cached_id2example[mode]
        dataset = load_dataset.cached_dataset[mode]
    else:
        processor = AlphaNliProcessor(args.data_dir, tokenizer)
        cache_filename = '{}_{}_{}_{}.features'.format(mode, args.model_type,
                                                       args.max_hyp_num if mode == 'train' else 2, args.max_seq_len)
        cache_features_file = os.path.join(args.data_dir, cache_filename)
        if os.path.exists(cache_features_file) and not args.overwrite_cache:
            id2example = dict([(e.id, e) for e in processor.get_examples(mode)])
            logger.info("Loading features from cache file: %s", cache_features_file)
            features = torch.load(cache_features_file)
        else:
            logger.info("Creating %s from examples", cache_filename)
            examples = processor.get_examples(mode)
            id2example = dict([(e.id, e) for e in examples])
            features = StoryFeatures.convert_from_examples(examples, tokenizer,
                                                           args.max_hyp_num if mode == 'train' else 2,
                                                           args.max_seq_len)
            torch.save(features, cache_features_file)
            logger.info("Saving features into cache file: %s", cache_features_file)
        dataset = AlphaNliDataset(features, args.tt_max_hyp_num)
        load_dataset.cached_id2example[mode] = id2example
        load_dataset.cached_dataset[mode] = dataset
        logger.info("All features loaded into memory")

    if partition is not None and 0 < partition < 1:
        indices = [idx for idx in range(len(dataset)) if random.random() < partition]
        ret_dataset = Subset(dataset, indices)
    elif partition is not None and 1 < partition < len(dataset):
        indices = list(range(partition))
        ret_dataset = Subset(dataset, indices)
    else:
        ret_dataset = dataset

    if mode == 'train':
        return ret_dataset

    id2feature = dict([(f.id, f) for f in dataset.features])

    return ret_dataset, id2example, id2feature


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default='dataset/alphanli/', type=str, required=True,
                        help="The input data dir.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pretrained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--linear_dropout_prob', type=float, default=0.6)

    parser.add_argument("--max_hyp_num", default=22, type=int,
                        help="The maximum number of hypotheses for a story.")
    parser.add_argument("--tt_max_hyp_num", default=22, type=int,
                        help="The maximum number of hypotheses for a story at training time.")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--criterion", default="list_mle", type=str,
                        help="Criterion for optimization selected in "
                             "[list_mle, list_net, approx_ndcg, rank_net, hinge, lambda]")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,  # 0.01
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--log_period', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--eval_period', type=int, default=1000,
                        help="Evaluate every X updates steps.")
    parser.add_argument('--save_period', type=int, default=-1,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name "
                             "and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument('--comment', default=None, type=str, help='The comment to the experiment')
    args = parser.parse_args()

    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and
            args.do_train and not args.overwrite_output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty. "
                         "Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count() if not args.no_cuda else 0
    args.device = device

    logger.info("Device: %s, n_gpu: %s, 16-bits training: %s", device, args.n_gpu, args.fp16)

    # Set seed
    set_seed(args)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    logger.info("Training/evaluation parameters: %s", args)

    # Before do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations.
    # Note that running `--fp16_opt_level="O2"` will remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        if not hasattr(config, 'linear_dropout_prob'):
            config.linear_dropout_prob = args.linear_dropout_prob
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        model = model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        logger.info(str(model))

        model.to(args.device)

        train(args, model, tokenizer)

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(os.path.join(args.output_dir, 'checkpoint-best_acc'))
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    if args.do_eval:
        results = {}
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c)
                               for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        else:
            checkpoints = [os.path.join(args.output_dir, 'checkpoint-best_acc')]

        logging.getLogger("utils").setLevel(logging.INFO)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split('-')[-1] if 'checkpoint' in checkpoint else ""
            tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case,
                                                        cache_dir=args.cache_dir if args.cache_dir else None)
            model = model_class.from_pretrained(checkpoint, cache_dir=args.cache_dir if args.cache_dir else None)
            model.to(args.device)
            if not args.no_cuda and args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Evaluate
            args.output_dir = checkpoint
            metrics, losses = evaluate(args, model, tokenizer, prefix=global_step, partition=None)
            write_results(args.comment, metrics, losses, os.path.join(args.output_dir, "dev-eval.txt"))
            metrics = dict((k + ('_{}'.format(global_step) if global_step else ''), v) for k, v in metrics.items())
            results.update(metrics)
        logger.info("Results: {}".format(results))


if __name__ == "__main__":
    main()
