import json
import torch
import logging
from typing import Dict, List, Optional, Any

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import DataloaderParams

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert software engineer.  
For each coding problem:

1. Think through the solution step by step - place your reasoning inside <think> ... </think> tags.  
2. Provide the final code solution inside <answer> ... </answer> tags.  

Do not output anything outside these tags.  
The answer must contain only the runnable code (no extra explanation after the tags)."""

def add_special_tokens(tokenizer: AutoTokenizer, model: Optional[AutoModelForCausalLM] = None):
    specials = ["<think>", "</think>", "<answer>", "</answer>"]
    
    tokenizer.add_special_tokens({
        "additional_special_tokens": specials
    })

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def _format_prompt(prompt: str, tokenizer):
    messages = [
        {"role" : "system", "content":SYSTEM_PROMPT},
        {"role" : "user", "content":prompt},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class SFTDataset(Dataset):

    def __init__(
            self,
            data: List[Dict[str,str]],
            tokenizer: PreTrainedTokenizer,
            max_length: int = 8192,
            system_prompt: str = SYSTEM_PROMPT,
            pre_tokenize: bool = True,
    ):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt
        self.data = data

        if pre_tokenize:
            logger.info("Pre‑tokenizing %d examples (this may take a moment)...", len(data))
            self.examples = [self._tokenize_example(item) for item in data]
        else:
            self.examples = None
        
    def _tokenize_example(self, item: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Tokenizer an example"""
        question = item.get("prompt","")
        response = item.get("response","")

        formatted_prompt = _format_prompt(question, self.tokenizer)

        full_text = formatted_prompt + response + self.tokenizer.eos_token

        enc_full = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = enc_full["input_ids"]

        prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        labels = [-100] * prompt_len + input_ids[prompt_len:]

        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
            input_ids = input_ids[:self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones_like(torch.tensor(input_ids, dtype=torch.long)),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.examples is not None:
            return self.examples[idx]
        else:
            return self._tokenize_example(self.data[idx])
        
def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]

    padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    padded_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return {
        "input_ids": padded_inputs,
        "labels": padded_labels,
        "attention_mask": padded_masks,
    }

def create_sft_dataloader(
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        data_config: DataloaderParams,
        drop_last: bool = False,
        ) -> DataLoader:

    dataset = SFTDataset(data, tokenizer, data_config.max_length, SYSTEM_PROMPT, data_config.pre_tokenize)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    from functools import partial
    collate_with_pad = partial(collate_fn, pad_token_id=pad_token_id)

    return DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=data_config.shuffle,
        collate_fn=collate_with_pad,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor if data_config.num_workers > 0 else None,
        persistent_workers=data_config.num_workers > 0,
        drop_last=drop_last,
    )

if __name__ == "__main__":
    # 1. Load tokenizer (and model if needed)
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16)

    tokenizer, model = add_special_tokens(tokenizer, model)

    data = []
    with open("./datasets/processed/opencode_sft_filtered.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    # 4. Create DataLoader
    dataloader = create_sft_dataloader(
        data=data,
        tokenizer=tokenizer,
        data_config = DataloaderParams(
                        **{
                            "batch_size": 1,
                            "max_length": 2048,
                            "num_workers": 1,
                            "pre_tokenize": False
                        }   
                    )
    )

    model.to("cuda")
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("Input shape:", batch["input_ids"].shape)
        print("Mask shape:", batch["attention_mask"].shape)
        print("Mask dtype:", batch["attention_mask"].dtype)
        print("Example label (first seq):", batch["labels"][0][:50])

        batch = {k: v.to(model.device) for k, v in batch.items()}

        res = model(**batch)

        break
        
