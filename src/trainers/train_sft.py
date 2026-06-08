import torch 
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Optional


def train(
    model : Any,
    tokenizer : Any,
    configs : dict[str, Any],  
    prompt : list[str],  # type: ignore
    response : list[str],  # type: ignore # pyright: ignore[reportInvalidTypeForm]
    epoch : int,

    

)