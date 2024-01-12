import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import wandb
import copy

import datasets
import evaluate
import numpy as np
from mydataset import get_dataset, tokenizer_datasets

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    # AutoAdapterModel,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Seq2SeqTrainer,
    # AdapterTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType, LoraConfig, PrefixTuningConfig
import adapters
from adapters import AdapterConfigBase
from RecAdam import RecAdam
import torch.optim as optim

import nltk
# nltk.download("punkt")
from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use. It can be arranged like "
                                        "'res_sup, lap_sup, acl_sup', and will transform into list."}
    )
    max_input_length: int = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: int = field(
        default=50,
        metadata={
            "help": (
                "The maximum total output sequence length."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default = "google/mt5-small", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    peft_name: Optional[str] =field(
        default=None ,metadata={"help": "Which peft to do, lora or prefix."}
    )
    optimizer_name: Optional[str] =field(
        default=None ,metadata={"help": "We can also choose RecAdam as optimizer"}
    )
    ignore_mismatched_sizes: bool = field(
        default=True,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():


    '''
        Initialize logging, seed, argparse. Basically use the code proveded by the example.
    '''
    Seq2SeqTrainingArguments.report_to="wandb"
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    # Since we do not do many experiments, we do not use json as our config, just use bash to train.
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    pn = model_args.peft_name if model_args.peft_name is not None else "None"
    if model_args.optimizer_name =="RecAdam":
        pn = "RecAdam"
    modelname=model_args.model_name_or_path.lower().replace(" ", "").split("/")[-1]
    wandb.init(project="huggingface"+"_"+modelname+"_"+data_args.dataset_name+"_"+pn, entity="ellucas2000017445")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    '''
        Load datasets. We can use our dataHelper to load dataset easily.
    '''
    raw_dataset = get_dataset(data_args.dataset_name)
    tokenized_dataset = tokenizer_datasets(raw_dataset)

    '''
        Load models.
    '''
    # Show the training loss with every epoch
    logging_steps = len(tokenized_dataset["train"]) // training_args.per_device_train_batch_size
    training_args.logging_steps=logging_steps

    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    
    if model_args.peft_name == "lora":
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=4, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    if model_args.peft_name == "prefix":
        config = PrefixTuningConfig(task_type = TaskType.SEQ_2_SEQ_LM, num_virtual_tokens = 12, prefix_projection = False)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    # print(model)


    '''
        build up datacollator.
    '''
    tokenized_datasets = tokenized_dataset.remove_columns(
        raw_dataset["train"].column_names
    )
    print(tokenized_datasets)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    '''
        build up metric.
    '''
    rouge_score = evaluate.load("./metrics/rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    if model_args.optimizer_name =="RecAdam":
        class CustomTrainer(Seq2SeqTrainer):
            def create_optimizer_and_scheduler(self, num_training_steps: int):
                no_decay = ["bias", "LayerNorm"]
                model_type = 'MT5'
                recadam_anneal_w = 1.0
                recadam_anneal_fun = 'sigmoid'
                recadam_anneal_k = 0.5
                recadam_anneal_t0 = 250
                recadam_pretrain_cof = 5000.0
                new_model = self.model
                self.pretrained_model = copy.deepcopy(self.model)
                for par in self.pretrained_model.parameters():
                    par.requires_grad = False
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in new_model.named_parameters() if
                                not any(nd in n for nd in no_decay) and model_type in n],
                        "weight_decay": 0.01,
                        "anneal_w": recadam_anneal_w,
                        "pretrain_params": [p_p for p_n, p_p in self.pretrained_model.named_parameters() if
                                            not any(nd in p_n for nd in no_decay) and model_type in p_n]
                    },
                    {
                        "params": [p for n, p in new_model.named_parameters() if
                                not any(nd in n for nd in no_decay) and model_type not in n],
                        "weight_decay": 0.01,
                        "anneal_w": 0.0,
                        "pretrain_params": [p_p for p_n, p_p in self.pretrained_model.named_parameters() if
                                            not any(nd in p_n for nd in no_decay) and model_type not in p_n]
                    },
                    {
                        "params": [p for n, p in new_model.named_parameters() if
                                any(nd in n for nd in no_decay) and model_type in n],
                        "weight_decay": 0.0,
                        "anneal_w": recadam_anneal_w,
                        "pretrain_params": [p_p for p_n, p_p in self.pretrained_model.named_parameters() if
                                            any(nd in p_n for nd in no_decay) and model_type in p_n]
                    },
                    {
                        "params": [p for n, p in new_model.named_parameters() if
                                any(nd in n for nd in no_decay) and model_type not in n],
                        "weight_decay": 0.0,
                        "anneal_w": 0.0,
                        "pretrain_params": [p_p for p_n, p_p in self.pretrained_model.named_parameters() if
                                            any(nd in p_n for nd in no_decay) and model_type not in p_n]
                    }
                ]
                optimizer = RecAdam(optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon,
                                    anneal_fun=recadam_anneal_fun, anneal_k=recadam_anneal_k,
                                    anneal_t0=recadam_anneal_t0, pretrain_cof=recadam_pretrain_cof)
                print("Optimizer created:", optimizer is not None)

                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=training_args.warmup_steps,
                    num_training_steps=num_training_steps
                )
                print("Scheduler created:", scheduler is not None)
                self.optimizer = optimizer
                self.lr_scheduler = scheduler

                return optimizer, scheduler

        # 创建 Trainer 实例
        trainer = CustomTrainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Seq2SeqTrainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        train_dataset=tokenized_datasets["train"]
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [None]
        eval_datasets = [eval_dataset]

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [None]
        predict_dataset = tokenized_datasets["test"]
        predict_datasets = [predict_dataset]

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()