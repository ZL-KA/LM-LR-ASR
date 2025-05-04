import re
from typing import List, Dict, Tuple
from datasets import DatasetDict
from datasets import Audio
import constants
# def raw_normalize_text(text):
#     # Reference: https://github.com/kevinduh/iwslt22-dialect/blob/3f8e9e83c0c0f77b904d24df94fb366afb07ad4e/1_prepare_stm.py#L30
#     arabic_filter = re.compile(r'[OUM]+/*|\u061F|\?|\!|\.')
#     english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\.|\,|\"|\:')
#     result = re.subn(english_filter, '', text)[0].lower()

#     return result


from importlib import import_module
import json
import random
import os
import numpy as np
import torch
from typing import Callable
import shutil


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from datasets import load_dataset

import soundfile as sf
def get_duration_from_audio(example):
    import pdb
    pdb.set_trace()
    try:
        audio, samplerate = sf.read(example['audio']['path'])
        example['duration'] = round(audio.shape[0]/samplerate, 2)
    except:
        print(f'fail to get duration for {example}')
        example['duration']=0
    return example



def get_duration_from_array(example):
    try:
        example['duration'] = round(example['audio']['array'].shape[0]/example['audio']['sampling_rate'], 2)
    except:
        print(f'fail to get duration for {example}')
        example['duration']=0.0
    return example


def remove_long_short_text(dataset, column_names: List, max_text: int, min_text: int):
    for col in column_names:
        dataset = dataset.filter(lambda example: min_text < len(example[col]) < max_text)
    return dataset


def do_normalize_text(example, lang, column_names: List):
    for col in column_names:
        if lang=='japhug':
            example[col] = normalize_japhug(example[col])
        else:
            example[col] = normalize_standard(example[col])
    return example

def preprocess(dataset, lang, filter_audio=False, normalize_text=False, filter_text=False, min_audio=1, max_audio=15, min_text=3, max_text=999):
    if filter_audio:
        dataset = dataset.map(get_duration_from_array)
        dataset = dataset.filter(lambda example: min_audio<example['duration']<max_audio)
        dataset = dataset.remove_columns('duration')

    if normalize_text:
        # remove abnormal chars and lowercase
        dataset = dataset.map(do_normalize_text, fn_kwargs={'lang':lang, 'column_names': ['transcript']})
    if filter_text:
        dataset = remove_long_short_text(dataset, column_names=['transcript'], max_text=max_text, min_text=min_text)
    return dataset


# Japhug normalizer
import re
def normalize_japhug(text):
    # These symbols are from the paper's script https://github.com/ CNRS-LACITO/xlsr_for_pangloss.
    MISC_SYMBOLS = { '~', '=', '¨', '↑', 'ː', '#', '$', 'X', '*', "+", "-", "_", '[', ']', '\ufeff'}
    PUNC_SYMBOLS = {',', '!', '.', ';', '?', "'", '"', '“', '”', '…', '«', '»', ':', '«', '»', "ʔ", ' ̩'}

    for symb in MISC_SYMBOLS:
        text=text.replace(symb, '') 
    for symb in PUNC_SYMBOLS:
        text=text.replace(symb, '')
    # The previous normalizer skips the Chinese chars and lead to a vocab of ~250. The paper uses a vocab of 44 and gets better scores 14CERand 22WER.
    # But the scores of the large vocab with chinese char leads to 18 CER and 35WER
    # so we start remove chinese chars.
    charas_to_ignore_regex_chinese_for_japhug = '[\u4e00-\u9fff]'
    text = re.sub(charas_to_ignore_regex_chinese_for_japhug, '', text)# Checked

    # With the chinese char removal, the vocab size is 82. The vocab still includes punctuations so do the following

    charas_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\[\]\(\)\+\/\<\=\>\«\»\'̥\…\↑\'̩\¨\~\#]' # checked for yongning na
    text = re.sub(charas_to_ignore_regex, '', text)# Checked

    # The previous end up with vocab size 79.
    import string
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    #The previous end up with vocab size 76. Some chiense character needs to be removed
    # Also lowercase it!
    chinese_punctuation = "，。！？；：“”‘’（）【】、《》—～"
    translator = str.maketrans('', '', chinese_punctuation)
    text = text.translate(translator)
    text = text.lower()
    # this one leads to vocab size of 70. Looks like they are all meaningful. This includes numbers 0-9 therefore is more than the vocab of paper.
    return text

# Other language normalizer
def normalize_standard(text):
    import re
    english_filter = re.compile(r'\(|\)|\#|\+|\=|\?|\!|\;|\.|\,|\"|\:\[\]')
    text = re.subn(english_filter, '', text)[0].lower()
    return text




def set_tokenizer_ids_tokens(tokenizer):
    import pdb
    pdb.set_trace()
    # print(tokenizer)
    import constants
    tokenizer.bos_token = constants.special_token_map_hf['bos_token']
    tokenizer.eos_token = constants.special_token_map_hf['eos_token']
    tokenizer.unk_token = constants.special_token_map_hf['unk_token']
    tokenizer.pad_token = constants.special_token_map_hf['pad_token']

    tokenizer.bos_token_id = constants.bos_id
    tokenizer.eos_token_id = constants.eos_id
    tokenizer.unk_token_id = constants.unk_id
    tokenizer.pad_token_id = constants.pad_id
    # print('After setting tokenizer ids and tokens: ')
    # print(tokenizer)
    return tokenizer

def check_and_create_folder(path, delete_old=False) -> Callable:
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if delete_old:
            print(f'**{path}** exists. With delete_old, delete the old one and create a new one')
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            pass

def build_llama_tkz(spm_model_path):
    '''
    Some special setting needed for using it with a new tokenizer; Need bos and eos, pad and unk; testing with 4.41.2
    Checked for subword
    '''
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer(vocab_file = spm_model_path, add_eos_token=True, pad_token='<pad>')
    # import pdb
    # pdb.set_trace()
    return tokenizer

def plot_training_history_min_max_checkpoints(parent_folder_path):
    '''
    parent_folder_path to the /PATH/checkpoints
    '''
    import os
    import re
    directory = parent_folder_path
    # Regular expression to match the subfolder names
    pattern = re.compile(r'checkpoint-(\d+)')
    
    min_checkpoint = None
    max_checkpoint = None
    min_folder = None
    max_folder = None

    # Iterate through the items in the given directory
    for folder_name in os.listdir(directory):
        match = pattern.match(folder_name)
        if match:
            checkpoint = int(match.group(1))
            if min_checkpoint is None or checkpoint < min_checkpoint:
                min_checkpoint = checkpoint
                min_folder = folder_name
            if max_checkpoint is None or checkpoint > max_checkpoint:
                max_checkpoint = checkpoint
                max_folder = folder_name

    min_folder = directory+'/'+min_folder
    max_folder = directory+'/'+max_folder



    os.system(f'python /asrlm/scripts/plot_training_history.py {min_folder}')
    os.system(f'python /asrlm/scripts/plot_training_history.py {max_folder}')




def measure_char_per_second(dataset):
    import torch
    torch.set_num_threads(1)

    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
    (get_speech_timestamps, _, read_audio, _, _) = utils
    for split in ['train', 'valid', 'test']:
        total_char_per_second=[]
        char_length_for_empty_vad_audio=[]
        ds=dataset[split]
        import tqdm
        for idx in tqdm.tqdm(range(len(ds))):
            audio = ds[idx]['audio']['array']
            sr = ds[idx]['audio']['sampling_rate']
            speech_timestamps = get_speech_timestamps(audio = ds[idx]['audio']['array'], model=model, sampling_rate=sr)
            total_length=0
            for item in speech_timestamps:
                total_length += round((item['end']-item['start'])/1000,2)
            if total_length==0:
                char_length_for_empty_vad_audio.append(len(ds[idx]['transcript']))
            else:
                total_char_per_second.append(round(len(ds[idx]['transcript'])/total_length, 2))
        avg_cps = round(sum(total_char_per_second)/len(total_char_per_second), 2)
        print(f'split {split} with avg_cps {avg_cps}')
        print('char_length_for_empty_vad_audio: ', char_length_for_empty_vad_audio)
