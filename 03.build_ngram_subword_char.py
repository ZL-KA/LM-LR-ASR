# build ngram
import os
KENLM_PATH = '/lm_data/kenlm_exp/build_kenlm/kenlm/build/bin/lmplz'


for lang in ['khinalug']:
    save_path = f'/asrlm/models/{lang}/ngram'
    # for tkz in ['subword2000', 'char']:
    for tkz in ['char']:
        TRAIN_SOURCE = save_path+f'/train.transcript.{tkz}.txt'
        # for gram_num in [2,3,5,10]:
        for gram_num in [5,10]:
            print(f'Building {gram_num}-gram model')
            SAVE_PATH=save_path+f'/{gram_num}gram.{tkz}.arpa'
            os.system(f'{KENLM_PATH} -o {gram_num} <{TRAIN_SOURCE}> {SAVE_PATH} --discount_fallback')
            # os.system(f'{KENLM_PATH} -o {gram_num} <{TRAIN_SOURCE}> {SAVE_PATH}')
            os.system(f'python correct_ngram_by_adding_eos.py {SAVE_PATH}')
