import argparse
import json
import logging

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, SequentialSampler

from transformers import RobertaTokenizer

from model import RobertaForListRank
from data_process import AlphaNliProcessor, StoryFeatures, AlphaNliDataset
from utils import RawResult, infer_labels

MODEL_DIR = './checkpoint-best_acc/'

logging.basicConfig(format='[%(asctime)s %(levelname)s %(name)s:%(lineno)s] %(message)s',
                    datefmt='%m/%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def infer(args, model, tokenizer):
    samples = AlphaNliProcessor.read_jsonl(args.input_file)
    dataset, id2example, id2feature = load_dataset(samples, tokenizer, args.max_seq_len)

    batch_size = 1 * max(1, args.n_gpu)
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=4)

    logger.info("***** Running inference *****")
    logger.info("  Num features = %d", len(dataset))
    logger.info("  Batch size = %d", batch_size)
    raw_results = []
    for batch in tqdm(data_loader, desc="Inferring"):
        model.eval()
        batch = tuple(t.to(args.device) if torch.is_tensor(t) else t for t in batch)
        with torch.no_grad():
            x = {'input_ids': batch[0], 'token_type_ids': batch[1], 'attention_mask': batch[2]}
            # (batch_size, list_len)
            logits = model(**x)

        feature_ids = batch[-1]
        for i, f_id in enumerate(feature_ids):
            raw_results.append(RawResult(id=f_id, logits=logits[i].detach().cpu()))

    infer_labels(samples, raw_results, id2example, args.output_file)


def load_dataset(samples, tokenizer, max_seq_len):
    examples = AlphaNliProcessor.get_test_examples(samples)
    features = StoryFeatures.convert_from_examples(examples, tokenizer, 2, max_seq_len)
    dataset = AlphaNliDataset(features)
    id2example = dict([(e.id, e) for e in examples])
    id2feature = dict([(f.id, f) for f in dataset.features])

    return dataset, id2example, id2feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Location of test records', default=None)
    parser.add_argument('--output_file', type=str, required=True, help='Location of predictions', default=None)
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--cuda", action='store_true', help="Whether to use CUDA when available")

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")

    # Setup CUDA, GPU
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.n_gpu = torch.cuda.device_count() if args.cuda else 0
    logger.info("Device: %s, n_gpu: %s", args.device, args.n_gpu)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
    model = RobertaForListRank.from_pretrained(MODEL_DIR)
    model.to(args.device)
    if args.cuda and args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Infer
    infer(args, model, tokenizer)


if __name__ == '__main__':
    main()
