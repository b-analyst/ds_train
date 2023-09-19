from flan_t5.training.preprocessing import *
from datasets import train_test_split   
import os

prompt_template = 'Generate USPC labels for the following text: '
model_name = 'google/flan-t5-small'
max_source_length = 517
max_target_length = 128
save_dataset_path = os.path.join(os.getcwd(), 'data')

dataset = load_hf_dataset(file_name='grouped_labels.csv')
train, test = train_test_split(dataset, test_size=0.15, shuffle=True, seed=42)
preprocessor=Preprocessor(prompt_template, model_name, max_source_length, max_target_length)
train_tokenized_dataset = train.map(preprocessor.preprocess, batched=True, remove_columns=list(train.features))
test_tokenized_dataset = test.map(preprocessor.preprocess, batched=True, remove_columns=list(test.features))
train_tokenized_dataset.save_to_disk(os.path.join(save_dataset_path,"train"))
test_tokenized_dataset.save_to_disk(os.path.join(save_dataset_path,"test"))

