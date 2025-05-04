import sys

import utils 
utils.seed_everything(42)

print('generating subword and char')
import sentencepiece as spm

splits = ['train', 'valid', 'test']

for split in splits:
    txt=f'/asrlm/models/{sys.argv[1]}/ngram/{split}.transcript.txt'
    for tkz in ['subword2000', 'char']:
        spm_model_path=f'/asrlm/models/{sys.argv[1]}/spm/{tkz}.model'
        sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        with open(txt, 'r') as file_read, open(txt.replace('.txt', f'.{tkz}.txt'), 'w') as file_write:
            for line in file_read:
                new_line=' '.join(sp.encode_as_pieces(line.strip())) + '\n'
                file_write.write(new_line)
        print(f'finish {split} {tkz}')
