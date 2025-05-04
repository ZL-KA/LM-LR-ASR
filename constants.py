
bos_id=1
eos_id=2
pad_id=3
unk_id=0


special_token_map_hf={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}

AUDIO_SR=16000

cven_valid_test_idx=1000
cven_training_idx=10000 # Was 10000 & 1000 or 50000 & 5000 before pushing to huggingface

# Generation
beam_size=5


# Wav2vec2 preprocess settings

wav2vec2_preprocessing = {'khinalug': {'min_audio':1, 'max_audio': 25, 'min_text':3, 'max_text':999},
                          'kichwa':{'min_audio':1, 'max_audio': 15, 'min_text':3, 'max_text':999},
                          'mboshi': {'min_audio':1, 'max_audio': 15, 'min_text':3, 'max_text':999},
                          'japhug':{'min_audio':1, 'max_audio': 15, 'min_text':3, 'max_text':999},
                          'cven':{'min_audio':1, 'max_audio': 25, 'min_text':3, 'max_text':999}
                          }

dict_from_tkztype_to_spmmodel_name={'nothing':'word', 'subword2000':'subword2000','char':'char'}

