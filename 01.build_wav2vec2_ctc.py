import utils
import utils_wav2vec2
import constants
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='khinalug')
    parser.add_argument('--config_yaml', type=str, default='khinalug_wav2vec2_ctc.yaml')
    # parser.add_argument('--resume_from_checkpoint', action='store_true')
    parser.add_argument('--tkz_transcript_type', type=str, default='nothing', help='char or subword2000, this would tokenize the transcript into pieces for later pyctcdecode, but maybe decrease ASR performance. Check the ASR scores!')
    parser.add_argument('--model_saving_prefix', type=str, default='')
    parser.add_argument('--use_adapter', action='store_true')
    parser.add_argument('--model_load_ckp', type=str, default=None)
    args = parser.parse_args()
    print('args: ', args)


    utils.seed_everything(42)
    if args.tkz_transcript_type=='nothing':
        output_path=f'/asrlm/models/{args.lang}/wav2vec2{args.model_saving_prefix}'
    else:
        output_path = f'/asrlm/models/{args.lang}/wav2vec2_CTCTKZ{args.tkz_transcript_type}{args.model_saving_prefix}'

    utils.check_and_create_folder(output_path)

    # Load config
    from omegaconf import OmegaConf
    config = OmegaConf.load(args.config_yaml)
    print('config: ', config)

    
    from datasets import load_dataset
    dataset = load_dataset(f'ZL92/{args.lang}_ASR_processed')
    if args.lang=='cven':
        import constants
        dataset['train']=dataset['train'].select(range(constants.cven_training_idx))
        dataset['valid']=dataset['valid'].select(range(constants.cven_valid_test_idx))
        dataset['test']=dataset['test'].select(range(constants.cven_valid_test_idx))
        print('dataset for cven should be sliced as train:valid:test=10,000:1,000:1,000')

    # Tokenize the transcript for CTC training for later integrating LM in pyctcdecode
    if args.tkz_transcript_type=='nothing':
        print('No CTC training tokenization for transcript')
        pass
    else:
        if args.lang=='cven':
            import pdb
            pdb.set_trace()
            spm_model_path = f'/asrlm/cven_10k/tokenizer/{args.tkz_transcript_type}.model'
        else:
            spm_model_path=f'/asrlm/models/{args.lang}/spm/{constants.dict_from_tkztype_to_spmmodel_name[args.tkz_transcript_type]}.model'
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        print('before CTC TKZ: ', dataset['train'][0]['transcript'])
        dataset = dataset.map(utils_wav2vec2.CTC_tokenize, fn_kwargs={'spmodel': sp, 'column_name':'transcript'})
        print('after CTC TKZ: ', dataset['train'][0]['transcript'])


    # Build and save the tokenizer and the feature extractor
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=constants.AUDIO_SR, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    tokenizer = utils_wav2vec2.build_tokenizer_wav2vec2_ctc(training_data=dataset['train'], output_path=output_path)
    processor=Wav2Vec2Processor(feature_extractor, tokenizer)
    processor.save_pretrained(output_path)

    model = Wav2Vec2ForCTC.from_pretrained('facebook/mms-300m',vocab_size=len(processor.tokenizer),
                                            ctc_loss_reduction="mean",
                                            pad_token_id=processor.tokenizer.pad_token_id,
                                            ignore_mismatched_sizes=True)

    
    model.to('cuda')

    # Prepare dataset for training
    data_collator = utils_wav2vec2.DataCollatorCTCWithPadding(processor=processor, padding=True)
    
    # The following prepare_dataset() is correct for both word, char and subword, because the label should be CTC-based
    dataset = dataset.map(utils_wav2vec2.prepare_dataset, fn_kwargs={'processor':processor}, load_from_cache_file=False)

    # Load evaluation metric for monitoring training
    from evaluate import load
    wer_metric = load("wer")
    cer_metric = load("cer")

    # Training
    from transformers import Trainer, EarlyStoppingCallback, TrainingArguments
    training_args = TrainingArguments(
                output_dir=output_path+'/checkpoints',
                group_by_length=True,
                learning_rate = config.trainer.learning_rate,
                evaluation_strategy='steps',
                logging_strategy='steps',
                save_strategy='steps',
                per_device_train_batch_size = config.trainer.per_device_train_batch_size,
                per_device_eval_batch_size=1,
                eval_steps=config.trainer.eval_steps,
                save_steps=config.trainer.save_steps,
                logging_steps=config.trainer.logging_steps,
                num_train_epochs=config.trainer.num_train_epochs,
                gradient_accumulation_steps=1,
                seed = config.trainer.seed,
                data_seed=config.trainer.seed,
                save_total_limit=5,
                fp16=True,
                gradient_checkpointing=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                load_best_model_at_end = True)

    from transformers import AdamW
    optimizer=AdamW(
        params=model.parameters(),
        lr=config.trainer.learning_rate, 
        weight_decay=config.trainer.weight_decay)


    if config.trainer.lr_scheduler_type=='linear':
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler=get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config.trainer.num_warmup_steps,
            num_training_steps=int(config.trainer.num_train_epochs * dataset['train'].__len__()/config.trainer.per_device_train_batch_size))
    elif config.trainer.lr_scheduler_type=='inverse_sqrt':
        from transformers import get_inverse_sqrt_schedule
        lr_scheduler=get_inverse_sqrt_schedule(
            optimizer=optimizer,
            num_warmup_steps=config.trainer.num_warmup_steps,
            timescale=int(config.trainer.num_train_epochs * dataset['train'].__len__()/config.trainer.per_device_train_batch_size)
        )
    trainer=Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
        tokenizer=feature_extractor,
        callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=config.trainer.early_stopping_patience)]
        )
    

    model.config.ctc_zero_infinity=True

    trainer.train()


