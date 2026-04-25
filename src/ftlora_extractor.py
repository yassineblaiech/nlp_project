import torch
import json
from typing import Literal
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

class OpinionExtractor:

    method: Literal["NOFT", "FT"] = "FT"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model_id = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None

    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        num_devices = max(1, torch.cuda.device_count())
        per_device_batch_size = max(1, 4 // num_devices)

        def format_dataset(data: list[dict]):
            formatted = []
            for item in data:
                review = item.get('Review', '')
                price = item.get('Price', 'No Opinion')
                food = item.get('Food', 'No Opinion')
                service = item.get('Service', 'No Opinion')
                prompt = (
                    f"Review: {review}\n"
                    f"Extraction:\n"
                    f"{{\"Price\": \"{price}\", \"Food\": \"{food}\", \"Service\": \"{service}\"}}"
                )
                formatted.append({"text": prompt})
            return Dataset.from_list(formatted)

        train_ds = format_dataset(train_data)
        val_ds = format_dataset(val_data)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float16,
            device_map=None,
        )

        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        training_args = SFTConfig(
            output_dir="./lora_qwen",
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="no",
            fp16=True,
            report_to="none",
            dataset_text_field="text",
            max_length=256 
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            peft_config=lora_config,
            args=training_args
        )

        trainer.train()
        self.model = trainer.model

    def predict(self, texts: list[str]) -> list[dict]:
        if self.model is not None:
            self.model.eval()

        predictions = []
        valid_classes = ["Positive", "Negative", "Mixed", "No Opinion"]

        for text in texts:
            prompt = f"Review: {text}\nExtraction:\n"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device if self.model else "cpu")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=40, 
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=0.0,
                    do_sample=False
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            pred_dict = {"Price": "No Opinion", "Food": "No Opinion", "Service": "No Opinion"}
            
            try:
                start_idx = response.find('{')
                end_idx = response.find('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx+1]
                    parsed = json.loads(json_str)
                    for k in ["Price", "Food", "Service"]:
                        if k in parsed and parsed[k] in valid_classes:
                            pred_dict[k] = parsed[k]
            except Exception:
                pass
            
            predictions.append(pred_dict)

        return predictions