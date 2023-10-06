import math
import os
import argparse
from time import time
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    set_seed,
)
import datasets
import transformers
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
import evaluate
import nltk
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, get_linear_schedule_with_warmup
from pathlib import Path
from accelerate import Accelerator
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory
import logging
from accelerate.logging import get_logger
from tqdm.auto import tqdm

nltk_path = os.path.join(str(Path.home()), 'nltk_data')
if not os.path.exists(nltk_path):
    nltk.download("punkt", quiet=True)

logger = get_logger(__name__)

# Metric
metric = evaluate.load("rouge")
# evaluation generation args
gen_kwargs = {
    "early_stopping": True,
    "length_penalty": 2.0,
    "max_new_tokens": 50,
    "min_length": 30,
    "no_repeat_ngram_size": 3,
    "num_beams": 4,
}


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--model_id", type=str, default="google/flan-t5-xl", help="Model id to use for training.")
    parser.add_argument("--dataset_path", type=str, default="data", help="Path to the already processed dataset.")
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size to use for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size to use for testing.")
    parser.add_argument("--generation_max_length", type=int, default=140, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate to use for training.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_warmup_steps", type=int, default=500, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Max train steps. Will calculate for you if None.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Path to deepspeed config file."),
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="log every n steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="Token to use for uploading models to Hugging Face Hub.",
    )
    args = parser.parse_known_args()
    return args


def training_function(args):
    # set seed
    set_seed(args.seed)

    accelerator = Accelerator()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    # load model from the hub
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_id,
        # not supported for Trainer
        # load_in_8bit=True,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
    )
    # # Define LoRA Config
    # lora_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=["q", "v"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type=TaskType.SEQ_2_SEQ_LM
    # )
    # # prepare int-8 model for training
    # model = prepare_model_for_int8_training(model)

    # # add LoRA adaptor
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()

    # max_memory = get_balanced_memory(
    #     model,
    #     max_memory=None,
    #     no_split_module_classes=["T5LayerSelfAttention", "T5LayerFF", "T5LayerCrossAttention"],
    #     dtype='float32',
    #     low_zero=False,
    # )

    # device_map = infer_auto_device_map(
    #     model,
    #     max_memory=max_memory,
    #     no_split_module_classes=["T5LayerSelfAttention", "T5LayerFF", "T5LayerCrossAttention"],
    #     dtype='float32'
    # )

    # model = dispatch_model(model, device_map=device_map)

    # we want to ignore tokenizer pad token in the loss
    label_pad_token_id = -100
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8
    )

    # load dataset from disk and tokenizer
    train_dataset = load_from_disk(os.path.join(args.dataset_path, "train"))
    eval_dataset = load_from_disk(os.path.join(args.dataset_path, "test"))

    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=data_collator, 
        batch_size=args.per_device_train_batch_size
    )

    eval_dataloader = DataLoader(
        eval_dataset, 
        collate_fn=data_collator, 
        batch_size=args.per_device_eval_batch_size
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    # New Code #
    # Creates Dummy Optimizer if `optimizer` was spcified in the config file else creates Adam Optimizer
    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.lr)

     # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch


    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

     # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
     # Figure out how many steps we should save the Accelerator states
    output_dir = os.path.join(os.getcwd(), args.model_id.split("/")[-1])

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 1

    for epoch in range(starting_epoch, args.epochs):
        start_time = time()
        model.train()
        total_loss = 0
        for _, batch in enumerate(train_dataloader):
            model.zero_grad()
            optimizer.zero_grad()
            inputs = {
                'input_ids':      batch['input_ids'].to(accelerator.device),
                'attention_mask': batch['attention_mask'].to(accelerator.device),
                'labels':         batch['labels'].to(accelerator.device),
                }       

            outputs = model(**inputs)
        
            loss = outputs.loss
            total_loss += loss.detach().float()
            accelerator.backward(loss)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            completed_steps += 1

            progress_bar.update(1)
            if isinstance(args.logging_steps, int):
                if completed_steps % args.logging_steps == 0:
                    steps_this_epoch = completed_steps % len(train_dataloader)
                    train_loss = total_loss.item() / steps_this_epoch
                    train_perplexity = math.exp(train_loss)
                    accelerator.log(
                        {
                            "train_loss": train_loss,
                            "train_perplexity": train_perplexity,
                            "epoch": epoch,
                            "step": completed_steps,
                            "steps_this_epoch": steps_this_epoch,
                        },
                        step=completed_steps,
                    )
                    logger.info(
                        f"Epoch: {epoch}, Step: {completed_steps}, Loss: {train_loss}, Perplexity: {train_perplexity}"
                    )
            
        end_time = time()
        logger.info(f"Epoch {epoch} training took {end_time-start_time} seconds")

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        accelerator.save(unwrapped_model.state_dict(), f'{output_dir}/epoch-{epoch}.pt')

    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # New Code #
    # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
    # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
    # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
    # For Zero Stages 1 and 2, models are saved as usual in the output directory.
    # The model name saved is `pytorch_model.bin`
    unwrapped_model.save_pretrained(
        output_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=accelerator.get_state_dict(model),
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

    # Define compute metrics function
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
