import sys

import argparse

from utils import seed_everything
seed_everything(42)
parser = argparse.ArgumentParser(description ='build_tokenizer')
parser.add_argument('--lang', type=str)
parser.add_argument('--tokenizer_type', type=str, help='word, subword or char')
parser.add_argument('--vocab_size', type=int, help='number of vocab for subword tokenization', default=2000)
args = parser.parse_args()
print(args)

root_path='asrlm/models'


train_transcript_path=f'{root_path}/{args.lang}/ngram/train.transcript.txt'
model_prefix=f'{root_path}/{args.lang}/spm/'
import utils
utils.check_and_create_folder(model_prefix)

import constants
bos_id=constants.bos_id
eos_id=constants.eos_id
pad_id=constants.pad_id
unk_id=constants.unk_id


### Build tokenizer with sentencepiece
import sentencepiece as spm
if args.tokenizer_type == 'word':
    spm.SentencePieceTrainer.Train(input=train_transcript_path, model_prefix=model_prefix+f'{args.tokenizer_type}', use_all_vocab=True, bos_id=bos_id, eos_id=eos_id, unk_id=unk_id, pad_id=pad_id, model_type="word")
elif args.tokenizer_type == 'subword':
    spm.SentencePieceTrainer.Train(input=train_transcript_path, model_prefix=model_prefix+f'{args.tokenizer_type}{args.vocab_size}', vocab_size=args.vocab_size, bos_id=bos_id, eos_id=eos_id, unk_id=unk_id, pad_id=pad_id, model_type="bpe")
elif args.tokenizer_type == 'char':
    spm.SentencePieceTrainer.Train(input=train_transcript_path, model_prefix=model_prefix+f'{args.tokenizer_type}', use_all_vocab=True, bos_id=bos_id, eos_id=eos_id, unk_id=unk_id, pad_id=pad_id, model_type="char")
else:
    raise NotImplementedError
