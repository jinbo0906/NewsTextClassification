from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, BertTokenizer, Trainer, \
    TrainingArguments, LineByLineTextDataset
import tokenizers

# define model
config = BertConfig(
    vocab_size=6824,
    hidden_size=768,
    num_hidden_layers=6,
    num_attention_heads=12,
    max_position_embeddings=512
)

model = BertForMaskedLM(config)
# print('No of parameters: ', model.num_parameters())
vocab_file_dir = 'bert-mini-my/vocab.txt'
# vocab_file_dir = 'bert-base-cased/vocab.txt'
tokenizer = BertTokenizer.from_pretrained(vocab_file_dir)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='../dataset/train_my_0.txt',
    block_size=512  # maximum sequence length
)

# define train parameter
training_args = TrainingArguments(
    output_dir='bert-mini-my/',
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    save_steps=10000,
    save_total_limit=2,
    use_cpu=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()
trainer.save_model('bert-mini-my/')
