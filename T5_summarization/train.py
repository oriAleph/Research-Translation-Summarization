from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import create_optimizer, AdamWeightDecay

tokenizer = 0

def preprocess_function(examples):
	inputs = ["summarize: " + doc for doc in examples["article"]]
	model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

	labels = tokenizer(text_target=examples["abstract"], max_length=128, truncation=True)
	model_inputs["labels"] = labels["input_ids"]
	return model_inputs


def my_tokenize(model_checkpoint, dataset, subset):
	global tokenizer

	sum = load_dataset(dataset, subset)

	tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
	tokenized_sum = sum.map(preprocess_function, batched=True)

#	return (tokenizer, tokenized_sum)


def get_model(model_checkpoint, tokenizer):
	model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
	data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
	return (model, data_collator)


def get_my_hyper_params(my_epochs, floating_point):
	training_args = Seq2SeqTrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                weight_decay=0.01, 
                save_total_limit=3,
                num_train_epochs=my_epochs,
                fp16=floating_point,
        )

	return training_args

def get_trainer(model, tokenizer, tokenized_sum, data_collator, training_args):
	trainer = Seq2SeqTrainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_bill["train"],
		eval_dataset=tokenized_bill["test"],
		tokenizer=tokenizer,
		data_collator=data_collator,
	)

	return trainer


def my_train():
	model_name = "t5-small"
	dataset = "scientific_papers"
	subset = "arxiv"
	epochs = 1
	floating_point = False

	token_tuple = my_tokenize(model_name, dataset, subset)

	model_tuple = get_model(model_name, token_tuple[0])

	params = get_my_hyper_params(epochs, floating_point)

	trainer = get_trainer(model_tuple[0], token_tuple[0], token_tuple[1], model_tuple[1], params)

	trainer.train()


my_train()
