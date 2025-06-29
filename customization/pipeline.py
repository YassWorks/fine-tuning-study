from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from helpers.utils import TextGenerator, DataSet
import torch
from datasets import Dataset as HFDataset, load_dataset
from typing import Any


MODEL_NAME = ""
OUTPUT_DIR = "./output"


USE_DAFT = True
USE_SFT = True
USE_LoRA = True


text_gen = TextGenerator(MODEL_NAME)
tokenizer = text_gen.tokenizer
model = text_gen.model


class DAFTTrainer:
    """
    Fine-tunes a given model using the Domain-Adaptive Fine-Tuning (DAFT) approach on a given dataset.
    Outputs a new fine-tuned model saved to a specified output directory.
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
        if model_name:
            self.text_gen = TextGenerator(model_name)
        elif text_gen:
            if not isinstance(text_gen, TextGenerator):
                raise ValueError("text_gen must be an instance of TextGenerator")
            self.text_gen = text_gen
        else:
            raise ValueError("Either model_name or text_gen must be provided")
        
        if dataset_name:
            try:
                self.dataset = DataSet(dataset_name, dataset_split_name)
            except Exception:
                raise ValueError(f"Please provide a valid dataset split.")
        elif dataset:
            if isinstance(dataset, DataSet):
                self.dataset = dataset
            elif isinstance(dataset, HFDataset):
                self.dataset = load_dataset(dataset, split=dataset_split_name)
        else:
            raise ValueError("Either dataset_name or dataset must be provided")


    def fine_tune(self):
        """
        Fine-tunes the model using the provided dataset. (Domain-Adaptive Fine-Tuning **DAFT** approach)
        It saves the trained model and tokenizer to the specified output directory.
        """

        model = self.text_gen.model
        tokenizer = self.text_gen.tokenizer
        _dataset = self.dataset.load()

        tokenized_train = _dataset.map(
            DAFTTrainer.preprocesser, remove_columns=_dataset.column_names
        )

        # Data collator (convenient for batching)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # Trainer args
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=4,
            save_strategy="epoch",
            num_train_epochs=3,
            learning_rate=5e-5,
            warmup_steps=100,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            logging_dir=f"{OUTPUT_DIR}/logs",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            data_collator=data_collator,
        )

        trainer.train()
        
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)


    @staticmethod
    def preprocesser(example):
        """
        Preprocess function for the dataset.
        """
        full_text = " ".join([str(value) for value in example.values()])
        encoded = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        encoded["labels"] = encoded["input_ids"].clone()
        return {k: v.squeeze() for k, v in encoded.items()}