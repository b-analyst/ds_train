from transformers import AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq, AutoTokenizer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk


# huggingface hub model id
# model_id = "philschmid/flan-t5-xxl-sharded-fp16"
model_id = "google/flan-t5-large"

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")

# Define LoRA Config
lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)
# prepare int-8 model for training
model = prepare_model_for_int8_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# load dataset
train = load_from_disk("data/train")
test = load_from_disk("data/test")

output_dir="lora-flan-t5-large"

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
	  # auto_find_batch_size=True,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=2,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=300,
    weight_decay=0.01,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="tensorboard",
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=tokenizer,
    compute_metrics=None,
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# train model
trainer.train()
# Save our LoRA model & tokenizer results
peft_model_id="lora-flan-t5-large-uspc"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)
# if you want to save the base model to call

 