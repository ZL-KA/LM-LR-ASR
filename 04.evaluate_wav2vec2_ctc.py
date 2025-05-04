import utils
import argparse
import torch
import os
import numpy as np
# from datasets import Dataset, load_dataset, Audio, load_metric
from datasets import Dataset, load_dataset, Audio
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Trainer, EarlyStoppingCallback, TrainingArguments
from evaluate import load
import constants
import wandb
import re
import utils_wav2vec2


AUDIO_SR = 16000

SEED = 42  # same as the default in seed_everything()


def prepare_dataset(batch, augmenter):
    "process the data to the format expected by Wav2Vec2ForCTC"
    audio = batch["audio"]
    # batched output is "un-batched" to ensure mapping is correct
    
    batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcript"]).input_ids
    return batch



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str)
    parser.add_argument('--pretrained_model_path', type=str)
    parser.add_argument('--lm_model_path', type=str)
    parser.add_argument('--tokenizer_path', type=str, default=None, help='path to the directory containing the model, such as /asrlm/models/khinalug/spm')
    parser.add_argument('--figsaveprefix', type=str, default='default')
    parser.add_argument('--tkz_transcript_type', type=str, default='nothing', help='char or subword2000, this would tokenize the transcript into pieces for later pyctcdecode, but maybe decrease ASR performance. Check the ASR scores!')
    parser.add_argument('--beam_width', type=int, default=100, help='beam widith(size), default is 100 in pyctcdecode->constants.py')
    parser.add_argument('--device',default='cuda', type=str)
    parser.add_argument('--split', type=str, help='valid OR test')
    parser.add_argument('--alpha', type=float, help='float value, necessary for test split')
    parser.add_argument('--beta', type=float, help='float value, necessary for test split')

    args = parser.parse_args()
    
    print(args)
    utils.seed_everything(42)
    dataset = load_dataset(f'None')
    

    if args.lang=='cven':
        import constants
        dataset['train']=dataset['train'].select(range(constants.cven_training_idx))
        dataset['valid']=dataset['valid'].select(range(constants.cven_valid_test_idx)).select(range(500))
        dataset['test']=dataset['test'].select(range(constants.cven_valid_test_idx))
        print('dataset for cven should be sliced as train:valid:test=10,000:1,000(500 here!):1,000')


    # Preprocessing
    dev_dataset = dataset[args.split]

    if args.pretrained_model_path:
        processor = Wav2Vec2Processor.from_pretrained(os.path.dirname(os.path.dirname(args.pretrained_model_path)))
        model = Wav2Vec2ForCTC.from_pretrained(args.pretrained_model_path)
    else:
        raise NotImplementedError
    model.to(args.device)

    model.eval()
    if args.tkz_transcript_type=='nothing':
        print('No CTC training tokenization for transcript')
        pass
    else:
        spm_model_path=f'/asrlm/models/{args.lang}/spm/{constants.dict_from_tkztype_to_spmmodel_name[args.tkz_transcript_type]}.model'
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=spm_model_path)
    dev_dataset = dev_dataset.map(prepare_dataset, fn_kwargs={'augmenter': None}, batch_size=1)

    wer_metric = load("wer")
    cer_metric = load("cer")

    if args.lm_model_path:
        from pyctcdecode import build_ctcdecoder, build_ctcdecoder_transformerLM
        if args.tokenizer_path:
            # Transformer LM
            assert 'ngramlm' not in args.lm_model_path # Should use transformerLM
            if args.tkz_transcript_type == 'nothing':
                tokenizer_type='word'
            elif args.tkz_transcript_type == 'subword2000':
                tokenizer_type='subword2000'
            elif args.tkz_transcript_type=='char':
                tokenizer_type='char'
            else:
                raise NotImplementedError
            decoder = build_ctcdecoder_transformerLM(
                labels=list({k.lower(): v for k, v in sorted(processor.tokenizer.get_vocab().items(), key=lambda item: item[1])}),
                transformerLM_path=args.lm_model_path, decoder_tkz_type=tokenizer_type,
                tokenizer_path=args.tokenizer_path, args_for_cven=args)
        else:
            # Ngram LM
            decoder = build_ctcdecoder(labels=list(
                    {k.lower(): v for k, v in sorted(processor.tokenizer.get_vocab().items(), key=lambda item: item[1])}),
                    kenlm_model_path=args.lm_model_path)
        from transformers import Wav2Vec2ProcessorWithLM
        processor = Wav2Vec2ProcessorWithLM(feature_extractor=processor.feature_extractor,
                                                tokenizer=processor.tokenizer,
                                                decoder=decoder)


        def map_to_result_wav2vec2_ctc_with_lm(batch, model, processor, device, alpha, beta):
            inputs = torch.tensor(
                batch["input_values"], device=device).unsqueeze(0)
            with torch.no_grad():
                logits = model(inputs).logits
            batch['pred_str'] = processor.decode(logits[0].cpu().numpy(
            ), alpha=alpha, beta=beta, output_word_offsets=True, beam_width=args.beam_width, beam_prune_logp = -20.0)['text']
            if batch['pred_str']=='':
                batch['pred_str']==' '
            return batch

        if args.split == 'valid':
            ALPHA=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
            if args.tkz_transcript_type=='char':
                BETA=[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
            elif args.tkz_transcript_type=='subword2000':
                BETA=[-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0]
            elif args.tkz_transcript_type=='nothing':
                BETA=[-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
            else:
                import pdb
                pdb.set_trace() # No valid one
        elif args.split=='test':
            ALPHA=[args.alpha]
            BETA=[args.beta]
        else:
            raise NotImplementedError

        scores_cer = np.zeros((len(ALPHA), len(BETA)))
        scores_wer = np.zeros((len(ALPHA), len(BETA)))
        for alpha in ALPHA:
            for beta in BETA:
                alpha_idx = ALPHA.index(alpha)
                beta_idx = BETA.index(beta)
                import copy
                dd=copy.deepcopy(dev_dataset)
                results = dd.map(map_to_result_wav2vec2_ctc_with_lm, fn_kwargs={
                                            'model': model, 'processor': processor, 'device': args.device, 'alpha': alpha, 'beta': beta}, load_from_cache_file=False)
                
                if args.tkz_transcript_type=='nothing':
                    pass
                else:
                    sp = spm.SentencePieceProcessor(model_file=spm_model_path) #spm_model_path should
                    results = results.map(utils_wav2vec2.CTC_tokenize_groundtruth_backtokenizer_predstr, fn_kwargs={'spmodel': sp}, load_from_cache_file=False)

                cer = cer_metric.compute(predictions=results["pred_str"], references=results["transcript"])
                wer = wer_metric.compute(predictions=results["pred_str"], references=results["transcript"])
                cer = round(cer*100, 2) 
                wer = round(wer*100, 2)
                print('Alpha, Beta: ', alpha, beta)
                print('Dev CER: {}'.format(cer))
                print('Dev WER: {}'.format(wer))

                scores_cer[alpha_idx, beta_idx]=cer
                scores_wer[alpha_idx, beta_idx]=wer

                if args.split == 'test':
                    print('Save test results to wav2vec2 folder')
                    if args.tokenizer_path:
                        results_save_path=args.pretrained_model_path+f'/split_{args.split}_transformerLM_results_CER{cer}_WER{wer}_alpha{args.alpha}_beta{args.beta}tkztranscript_{args.tkz_transcript_type}_{args.beam_width}_{args.figsaveprefix}.csv'
                    else:
                        ngram_prefix=args.lm_model_path.split('/')[-1].split('.')[0]
                        results_save_path=args.pretrained_model_path+f'/split_{args.split}_ngramLM_{ngram_prefix}_results_CER{cer}_WER{wer}_alpha{args.alpha}_beta{args.beta}_tkztranscript_{args.tkz_transcript_type}_{args.beam_width}_{args.figsaveprefix}.csv'
                    remove_columns = ['audio', 'input_values', 'labels']
                    results = results.remove_columns(remove_columns)
                    results.to_csv(results_save_path, index=False, sep='|')
                else:
                    print('Save valid results')
                    import pickle
                    if args.tokenizer_path:
                        file_path = args.pretrained_model_path+f'/split_valid_transformerLM_beam_{args.beam_width}_tkztranscript_{args.tkz_transcript_type}_{args.figsaveprefix}_results.pkl'
                    else:
                        ngram_prefix=args.lm_model_path.split('/')[-1].split('.')[0]
                        file_path = args.pretrained_model_path+f'/split_valid_ngramLM_{ngram_prefix}_beam_{args.beam_width}_tkztranscript_{args.tkz_transcript_type}_{args.figsaveprefix}_results.pkl'
                    # Open the file in binary write mode
                    with open(file_path, 'wb') as f:
                        # Pickle the data and write it to the file
                        pickle.dump(ALPHA, f)
                        pickle.dump(BETA, f)
                        pickle.dump(scores_cer, f)
                        pickle.dump(scores_wer, f)
                    # print("Data saved successfully.")

        if args.split == 'valid':
            print('ploting the results')
            import os
            os.system(f'python /asrlm/scripts/05.visualize_evaluate_results_alpha_beta.py {file_path}')


    else: # No lm_model_path
        def map_to_result_wav2vec2_ctc(batch, model, processor, device):
            with torch.no_grad():
                input_values = torch.tensor(
                    batch["input_values"], device=device).unsqueeze(0)
                logits = model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            batch["pred_str"] = processor.batch_decode(pred_ids)[0]
            if batch['pred_str']=='':
                batch['pred_str']==' '
            return batch
        
        import copy
        dd=copy.deepcopy(dev_dataset)
        print('start mapping with pyctcdecode')
        results = dd.map(map_to_result_wav2vec2_ctc, 
            fn_kwargs={'model': model, 'processor': processor, 'device': args.device}, 
            load_from_cache_file=False)
        
        if args.tkz_transcript_type=='nothing':
            pass
        else:
            sp = spm.SentencePieceProcessor(model_file=spm_model_path)

            results = results.map(utils_wav2vec2.CTC_tokenize_groundtruth_backtokenizer_predstr, fn_kwargs={'spmodel': sp}, load_from_cache_file=False)


        cer = cer_metric.compute(predictions=results["pred_str"], references=results["transcript"])
        wer = wer_metric.compute(predictions=results["pred_str"], references=results["transcript"])
        cer = round(cer*100, 2)
        wer = round(wer*100, 2)
        print('Dev CER: {}'.format(cer))
        print('Dev WER: {}'.format(wer))

        results_save_path=args.pretrained_model_path+f'/split_{args.split}_noLM_results_CER{cer}_WER{wer}_{args.figsaveprefix}.csv'
        remove_columns = ['audio', 'input_values', 'labels']
        results = results.remove_columns(remove_columns)
        results.to_csv(results_save_path, index=False, sep='|')


