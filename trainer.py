from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, \
    DataCollatorForLanguageModeling

from Doc_processor import preprocess

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


def create_dataset(docs, tokenizer):
    texts = "\n\n".join(docs)
    with open("temp_texts.txt", "w", encoding="utf-8") as f:
        f.write(texts)
    dataset = TextDataset(tokenizer=tokenizer, file_path="temp_texts.txt", block_size=128)
    return dataset


prepocessed_docs = preprocess()
dataset = create_dataset(prepocessed_docs, tokenizer)

training_args = TrainingArguments(output_dir='./gpt2-finetuned', overwrite_output_dir=True, num_train_epochs=3,
                                  per_device_train_batch_size=2, save_steps=10_1000, save_total_limit=2, )
data_collector = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(model=model, args=training_args, data_collator=data_collector, train_dataset=dataset)

trainer.train()

model.save_pretrained('./gpt2-finetuned')
tokenizer.save_pretrained('./gpt2-finetuned')
