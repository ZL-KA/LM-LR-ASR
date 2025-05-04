import argparse
from datasets import Dataset, concatenate_datasets, DatasetDict
import constants

def load_dataset_from_text(args):
    # Load only the text and support word, subword and char toeknizations
    datasets=[]
    for split in ['train', 'valid', 'test']:
        input_txt = f'/asrlm/models/{args.lang}/ngram/{split}.transcript.txt'
        with open(input_txt, 'r') as f:
            data=f.readlines()
        transcript = [d.strip() for d in data]
        dataset = Dataset.from_dict({'transcript':transcript})
        datasets.append(dataset)
    final_dataset = DatasetDict({
        'train': datasets[0],
        'valid': datasets[1],
        'test': datasets[2]
    })
    return final_dataset

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='khinalug')
    parser.add_argument('--model_type', type=str, help='gpt2, mbart', default='gpt2')
    parser.add_argument('--config_yaml', type=str, default='/asrlm/configs/transformerLM/khinalug/gpt2_word_config1.yaml')
    parser.add_argument('--tkz_type', type=str, help='word, subword2000, char', default='word')

    args = parser.parse_args()
    print(args)

    import utils
    utils.seed_everything()

    
    # Build sentencepiece tokenizer
    spm_path=f'/asrlm/models/{args.lang}/spm'
    utils.check_and_create_folder(spm_path)
    
    train_transcript_path=f'/asrlm/models/{args.lang}/ngram/train.transcript.txt'
    spm_model_prefix = spm_path + f'/{args.tkz_type}'
    
    dataset = load_dataset_from_text(args)

    from transformers import LlamaTokenizer as hftkz
    tokenizer = utils.build_llama_tkz(spm_model_path=f'{spm_model_prefix}.model') #Yes the special tokens of HF and spm are matched

    def preprocess_function_bos_eos(example):
        #tokenizer.add_eos_token=True, tokenizer.add_bos_token=True
        outputs = tokenizer(example['transcript'])
        return outputs

    
    dataset = dataset.map(preprocess_function_bos_eos, remove_columns=dataset['train'].column_names) # CHecked correct for word, subword and char

    from omegaconf import OmegaConf
    training_config = OmegaConf.load(args.config_yaml)
    print('training_config: ', training_config)

    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, EarlyStoppingCallback
    if args.model_type=='gpt2':
        from transformers import GPT2Config, GPT2LMHeadModel
        config = GPT2Config.from_pretrained('distilgpt2')
        config.n_layer=training_config.modelconfig.decoder_layers
        config.n_embd=training_config.modelconfig.decoder_sz
        config.vocab_size=tokenizer.vocab_size
        config.bos_token_id=tokenizer.bos_token_id
        config.eos_token_id=tokenizer.eos_token_id
        config.unk_token_id = tokenizer.unk_token_id
        config.pad_token_id = tokenizer.pad_token_id
        model = GPT2LMHeadModel(config)
    else:
        raise NotImplementedError
    
    print(model)

    config_prefix=args.config_yaml.split('/')[-1].replace('.yaml', '')
    model_save_path=f'/asrlm/models/{args.lang}/transformerLM'
    utils.check_and_create_folder(model_save_path)


    parent_folder_path_to_checkpoints=f'{model_save_path}/config{config_prefix}.tkz{args.tkz_type}'
    # Training
    training_args = TrainingArguments(
        output_dir=parent_folder_path_to_checkpoints,
        evaluation_strategy="steps",
        logging_strategy = 'steps',
        save_strategy='steps',
        per_device_train_batch_size = training_config.trainer.per_device_train_batch_size,
        per_device_eval_batch_size = training_config.trainer.per_device_eval_batch_size,
        eval_steps=training_config.trainer.eval_steps,
        save_steps=training_config.trainer.save_steps,
        logging_steps=training_config.trainer.logging_steps,
        num_train_epochs=training_config.trainer.num_train_epochs,
        seed = training_config.trainer.seed,
        save_total_limit=2,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_smoothing_factor = training_config.trainer.label_smoothing_factor,
        load_best_model_at_end = True)
    
    from transformers import AdamW
    optimizer=AdamW(
        params=model.parameters(),
        lr=training_config.trainer.learning_rate, 
        weight_decay=training_config.trainer.weight_decay)

    if training_config.trainer.lr_scheduler_type=='linear':
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler=get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_config.trainer.num_warmup_steps,
            num_training_steps=training_config.trainer.num_training_steps)
    elif training_config.trainer.lr_scheduler_type=='inverse_sqrt':
        from transformers import get_inverse_sqrt_schedule
        lr_scheduler=get_inverse_sqrt_schedule(
            optimizer=optimizer,
            num_warmup_steps=training_config.trainer.num_warmup_steps,
            timescale=training_config.trainer.timescale
        )
    else:
        raise NotImplementedError
    trainer=Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[EarlyStoppingCallback(
                    early_stopping_patience=training_config.trainer.early_stopping_patience)]
        )
    trainer.train()



