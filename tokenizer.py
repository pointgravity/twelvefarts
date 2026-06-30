"""
Tokenizer setup using the standard Llama tokenizer (SentencePiece).
"""

from transformers import LlamaTokenizer, PreTrainedTokenizerFast
from .config import TOKENIZER_NAME


def get_tokenizer() -> LlamaTokenizer:
    """
    Load the Llama tokenizer from a pretrained model.
    Sets pad_token to eos_token and configures padding side.
    """
    tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_NAME)

    # Llama tokenizer has no pad token; we set it to eos for safety
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use left padding for generation (causal)
    tokenizer.padding_side = "left"

    # Ensure we have the correct special tokens
    # eos_token is already set (e.g., '</s>')
    # bos_token is '<s>'
    return tokenizer