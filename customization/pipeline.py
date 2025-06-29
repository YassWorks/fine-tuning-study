from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from helpers.utils import TextGenerator, DataSet, random_hash
import torch
from datasets import Dataset as HFDataset, load_dataset
from typing import Any
import logging
import time
import os


OUTPUT_DIR = "./.cache/output"
LOGS_DIR = OUTPUT_DIR + "/logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(LOGS_DIR + f"/log_{time.strftime('%Y%m%d_%H%M%S')}.log", mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

handler.setFormatter(formatter)
logger.addHandler(handler)


class DAFTTrainer:
    """
    Fine-tunes a given model using the Domain-Adaptive Fine-Tuning (DAFT) approach on a given dataset.
    Outputs a new fine-tuned model that can be saved to a specified output directory.
    """


    def __init__(
        self,
        text_gen: TextGenerator = None,
        model_name: str = None,
        dataset: DataSet | HFDataset | Any = None,
        dataset_name: str = None,
        dataset_split_name: str = "train",
    ):
        """
        Initializes the DAFTTrainer with either a TextGenerator instance or a model name, 
        and either a DataSet instance or a dataset name.
        Args:
            text_gen (TextGenerator, optional): An instance of TextGenerator. Defaults to None.
            model_name (str, optional): The name of the model to use. Defaults to None.
            dataset (DataSet, optional): An instance of DataSet. Defaults to None.
            dataset_name (str, optional): The name of the dataset to use. Defaults to None.
            dataset_split (str, optional): The split of the dataset to use. Defaults to "train".
        """
        # Configuring the TextGenerator (model + tokenizer)
        if model_name:
            self.text_gen = TextGenerator(model_name)
            
        elif text_gen:
            if not isinstance(text_gen, TextGenerator):
                err_msg = "text_gen must be an instance of TextGenerator"
                logger.exception(err_msg)
                raise ValueError(err_msg)
            self.text_gen = text_gen
            
        else:
            err_msg = "Either model_name or text_gen must be provided"
            logger.exception(err_msg)
            raise ValueError(err_msg)
        
        # Configuring the Dataset
        if dataset_name:
            try:
                _dataset = DataSet(dataset_name, dataset_split_name)
                self.dataset = _dataset.load()
            except Exception:
                err_msg = "Please provide a valid dataset split."
                logger.exception(err_msg)
                raise ValueError(err_msg)
        
        elif dataset:
            if isinstance(dataset, DataSet):
                try:
                    self.dataset = dataset.load()
                except Exception:
                    err_msg = "Please provide a valid dataset split."
                    logger.exception(err_msg)
                    raise ValueError(err_msg)
            elif isinstance(dataset, HFDataset):
                self.dataset = dataset # assuming it's good to go :3
        
        else:
            err_msg = "Either dataset_name or dataset must be provided"
            logger.exception(err_msg)
            raise ValueError(err_msg)


    def fine_tune(self, save_to_disk: bool = False, limit: int = None):
        """
        Fine-tunes the model using the provided dataset. (Domain-Adaptive Fine-Tuning **DAFT** approach)
        #### Note:
        This method will change the model's parameters **inplace**.
        Args:
            save_to_disk (bool): Whether to save checkpoints and logs to disk during training. Defaults to False.
        """
        model = self.text_gen.model
        tokenizer = self.text_gen.tokenizer
        if limit:
            logger.info(f"Limiting dataset to {limit} samples.")
            self.dataset = self.dataset.select(range(limit))
        
        tokenized_train = self.dataset.map(
            self.preprocessor, remove_columns=self.dataset.column_names
        )
        logger.info(f"Tokenized dataset.")

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        logger.info(f"Data collator created.")
        
        if save_to_disk:
            _hash = random_hash()
            output_dir = OUTPUT_DIR + f"/daft/daft_session_{_hash}"
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"save_to_disk was set to 'True'. Output directory set to {output_dir}.")
            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=4,
                save_strategy="epoch",
                num_train_epochs=3,
                learning_rate=5e-5,
                warmup_steps=100,
                weight_decay=0.01,
                fp16=torch.cuda.is_available(),
                logging_dir=f"{output_dir}/logs",
                report_to="none",
            )
        else:
            # In-memory training without disk I/O
            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,  # Required but won't be used
                per_device_train_batch_size=4,
                save_strategy="no",
                num_train_epochs=3,
                learning_rate=5e-5,
                warmup_steps=100,
                weight_decay=0.01,
                fp16=torch.cuda.is_available(),
                # logging_steps=0,
                report_to="none",
                dataloader_pin_memory=False,
                remove_unused_columns=True,
            )
        logger.info(f"Training arguments set up.")
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            data_collator=data_collator,
        )

        logger.info(f"Trainer starting...")
        trainer.train()
        logger.info(f"Training completed.")
    

    def preprocessor(self, example):
        """
        Preprocess function for the dataset.
        """
        full_text = " ".join([str(value) for value in example.values()])
        encoded = self.text_gen.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        encoded["labels"] = encoded["input_ids"].clone()
        return {k: v.squeeze() for k, v in encoded.items()}
    
    
    def save_model(self, output_dir: str = None):
        """
        Saves the model and tokenizer to the specified output directory.
        Args:
            output_dir (str): The directory where the model and tokenizer will be saved.
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR + "/final_daft"
        _hash = random_hash()
        output_dir = f"{output_dir}/{_hash}"
        os.makedirs(output_dir, exist_ok=True)
        
        self.text_gen.model.save_pretrained(output_dir)
        self.text_gen.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}.")
        

class SFTTrainer:
    """
    Fine-tunes a given model using the Domain-Adaptive Fine-Tuning (SFT) approach on a given dataset.
    Outputs a new fine-tuned model that can be saved to a specified output directory.
    """
    def __init__(
        self,
        text_gen: TextGenerator = None,
        model_name: str = None,
        train_dataset: DataSet | HFDataset | Any = None,
        train_dataset_name: str = None,
        train_dataset_split_name: str = "train",
        test_dataset: DataSet | HFDataset | Any = None,
        test_dataset_name: str = None,
        test_dataset_split_name: str = "test",
    ):
        """
        Initializes the DAFTTrainer with either a TextGenerator instance or a model name, 
        and either a DataSet instance or a dataset name.
        Args:
            text_gen (TextGenerator, optional): An instance of TextGenerator. Defaults to None.
            model_name (str, optional): The name of the model to use. Defaults to None.
            train_dataset (DataSet, optional): An instance of DataSet for training. Defaults to None.
            train_dataset_name (str, optional): The name of the training dataset to use. Defaults to None.
            train_dataset_split_name (str, optional): The split of the training dataset to use. Defaults to "train".
            test_dataset (DataSet, optional): An instance of DataSet for testing. Defaults to None.
            test_dataset_name (str, optional): The name of the testing dataset to use. Defaults to None.
            test_dataset_split_name (str, optional): The split of the testing dataset to use. Defaults to "test".
        """
        if model_name:
            self.text_gen = TextGenerator(model_name)
        elif text_gen:
            if not isinstance(text_gen, TextGenerator):
                raise ValueError("text_gen must be an instance of TextGenerator")
            self.text_gen = text_gen
        else:
            raise ValueError("Either model_name or text_gen must be provided")
        
        if train_dataset_name:
            try:
                self.train_dataset = DataSet(train_dataset_name, train_dataset_split_name)
            except Exception:
                raise ValueError(f"Please provide a valid dataset split.")
        elif train_dataset:
            if isinstance(train_dataset, DataSet):
                self.train_dataset = train_dataset
            elif isinstance(train_dataset, HFDataset):
                self.train_dataset = load_dataset(train_dataset, split=train_dataset_split_name)
        else:
            raise ValueError("Either dataset_name or dataset must be provided")
        
        if test_dataset_name:
            try:
                self.test_dataset = DataSet(test_dataset_name, test_dataset_split_name)
            except Exception:
                raise ValueError(f"Please provide a valid dataset split.")
        elif test_dataset:
            if isinstance(test_dataset, DataSet):
                self.test_dataset = test_dataset
            elif isinstance(test_dataset, HFDataset):
                self.test_dataset = load_dataset(test_dataset, split=test_dataset_split_name)
        else:
            raise ValueError("Either dataset_name or dataset must be provided")


    def fine_tune(self, save_to_disk: bool = False):
        """
        Fine-tunes the model using the provided dataset. (Supervised Fine-Tuning **SFT** approach)
        #### Note:
        This method will change the model's parameters **inplace**.
        Args:
            save_to_disk (bool): Whether to save checkpoints and logs to disk during training. Defaults to False.
        """
        model = self.text_gen.model
        tokenizer = self.text_gen.tokenizer
        _train_dataset = self.train_dataset.load()
        _test_dataset = self.test_dataset.load()

        tokenized_train = _train_dataset.map(
            self.preprocessor, remove_columns=_train_dataset.column_names
        )
        tokenized_test = _test_dataset.map(
            self.preprocessor, remove_columns=_test_dataset.column_names
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        if save_to_disk:
            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                eval_strategy="steps",
                eval_steps=500,
                logging_steps=100,
                save_strategy="epoch",
                save_total_limit=2,
                num_train_epochs=3,
                learning_rate=5e-5,
                weight_decay=0.01,
                warmup_steps=100,
                fp16=torch.cuda.is_available(),
                logging_dir=f"{OUTPUT_DIR}/logs",
                report_to="none",
                push_to_hub=False
            )
        else:
            # In-memory training without disk I/O
            training_args = TrainingArguments(
                output_dir="./tmp",  # Required but won't be used
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                eval_strategy="steps",
                eval_steps=500,
                save_strategy="no", 
                num_train_epochs=3,
                learning_rate=5e-5,
                weight_decay=0.01,
                warmup_steps=100,
                fp16=torch.cuda.is_available(),
                report_to="none",
                push_to_hub=False,
                dataloader_pin_memory=False,
                remove_unused_columns=True,
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            data_collator=data_collator
        )

        trainer.train()
        

    def preprocessor(self, example):
        """
        Preprocess function for the dataset.
        """
        full_text = " ".join([str(value) for value in example.values()])
        encoded = self.text_gen.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        encoded["labels"] = encoded["input_ids"].clone()
        return {k: v.squeeze() for k, v in encoded.items()}
    
    
    def save_model(self, output_dir: str = None):
        """
        Saves the model and tokenizer to the specified output directory.
        Args:
            output_dir (str): The directory where the model and tokenizer will be saved.
        """
        if output_dir is None:
            output_dir = OUTPUT_DIR + "/final_sft"
        _hash = random_hash()
        output_dir = f"{output_dir}/{_hash}"
        
        self.text_gen.model.save_pretrained(output_dir)
        self.text_gen.tokenizer.save_pretrained(output_dir)