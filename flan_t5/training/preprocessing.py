import boto3
import botocore
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Optional

# Replace these values with your own
access_key_id = os.environ['S3_ACCESS_KEY_ID']
secret_access_key = os.environ['S3_SECRET_ACCESS_KEY']
bucket_name = os.environ['S3_BUCKET_NAME']
target = os.environ['DATA_PATH']  # Replace with the actual object key

def s3_download(destination: str='data'):
# Create an S3 client
    s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    try:
        # Download the dataset
        s3.download_file(bucket_name, target, destination)
        print(f"Dataset downloaded to './{destination}'")
    except botocore.exceptions.NoCredentialsError:
        print("AWS credentials not found or invalid.")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
def construct_prompt(prompt_template: str=None, context: str=None):
    return f'{prompt_template}\n\nContext: \n{context}\n\nOutput: '

def load_hf_dataset(data_path: str='data', download: Optional[bool]=True, file_name: str='data.csv'):
    path = os.path.join(os.getcwd(), data_path)
    if download:
        s3_download(destination=data_path)
        return Dataset.from_pandas(pd.read_csv(os.path.join(path, file_name)))
    else:
        return Dataset.from_pandas(pd.read_csv(os.path.join(path, file_name)))

class Preprocessor:
    def __init__(self, prompt_template: str='', model_name: str='google/flan-t5-small', max_source_length: int=517, max_target_length: int=128):
        self.model_name=model_name
        self.max_source_length=max_source_length
        self.max_target_length=max_target_length
        self.prompt_template=prompt_template

    def preprocess(self, sample, padding='max_length'):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        inputs = [construct_prompt(self.prompt_template, item) for item in sample['inputs']]

        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=self.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=sample['targets'], max_length=self.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs






