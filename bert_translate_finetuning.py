from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertForSequenceClassification, BertForNextSentencePrediction, AdamW
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import torch

def fine_tune_korean_to_english_model():
    # Load pretrained BERT model
    model = BertForSequenceClassification.from_pretrained("msarmi9/korean-english-multitarget-ted-talks-task")

    # Modify the model for Korean to English translation
    model.config.update({"is_decoder": True, "add_cross_attention": True, "vocab_size": model.config.vocab_size})

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("msarmi9/korean-english-multitarget-ted-talks-task")

    # Load Korean-English parallel corpus for fine-tuning
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="korean_english_parallel_corpus.txt",
        block_size=128,
    )

    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="./korean_to_english_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=500,
        save_steps=1000,
        eval_steps=1000,
        learning_rate=5e-5,
        warmup_steps=500,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Fine-tune the model
    trainer.train()

def fine_tune_english_to_korean_model():
    # Load pretrained BERT model
    model = BertForSequenceClassification.from_pretrained("msarmi9/korean-english-multitarget-ted-talks-task")

    # Modify the model for English to Korean translation
    model.config.update({"is_decoder": True, "add_cross_attention": True, "vocab_size": model.config.vocab_size})

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("msarmi9/korean-english-multitarget-ted-talks-task")

    # Load English-Korean parallel corpus for fine-tuning
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="english_korean_parallel_corpus.txt",
        block_size=128,
    )

    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="./english_to_korean_model",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_steps=500,
        save_steps=1000,
        eval_steps=1000,
        learning_rate=5e-5,
        warmup_steps=500,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Fine-tune the model
    trainer.train()

# Fine-tune the Korean to English translation model
fine_tune_korean_to_english_model()

# Fine-tune the English to Korean translation model
fine_tune_english_to_korean_model()
