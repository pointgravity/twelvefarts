"""
Inference and chat interface for Perspicacity Next.
Uses Hugging Face generation APIs with support for streaming, temperature, top-k, top-p, repetition penalty.
"""

import torch
from transformers import LlamaForCausalLM, TextStreamer
from typing import List, Optional, Union

from .config import CHECKPOINT_DIR, MAX_POSITION_EMBEDDINGS, create_llama_config
from .tokenizer import get_tokenizer


def load_model_for_inference(load_dir: str = CHECKPOINT_DIR, device: Optional[torch.device] = None) -> LlamaForCausalLM:
    """Load the model from a saved directory."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizer to get vocab size
    tokenizer = get_tokenizer()
    config = create_llama_config(tokenizer.vocab_size)
    model = LlamaForCausalLM.from_pretrained(load_dir, config=config, torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.to(device)
    model.eval()
    return model


def generate_response(
    model: LlamaForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    stop_strings: Optional[List[str]] = None,
    stream: bool = False,
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate a response given a prompt.
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_POSITION_EMBEDDINGS)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    # Generation config
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
    }

    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generate_kwargs["streamer"] = streamer

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    generated_ids = outputs.sequences[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Stop strings
    if stop_strings:
        for stop in stop_strings:
            if stop in response:
                response = response.split(stop)[0]
                break

    return response


def chat():
    """Interactive chat loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[CHAT] Loading model...")
    model = load_model_for_inference(CHECKPOINT_DIR, device)
    tokenizer = get_tokenizer()

    print(f"[CHAT] Loaded model from {CHECKPOINT_DIR} on {device}")
    print("[CHAT] Type 'quit' to exit.")
    print("-" * 60)

    # Map lowercased user inputs to exact conversational pairs for better consistency
    conv_map = {q.lower().strip(): q for q, _ in dataset.CONVERSATIONAL_PAIRS}

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.strip().lower() in ["quit", "exit"]:
            break
        if not user_input.strip():
            continue

        # Try to match normalized input to known conversation pairs for better quality
        normalized = user_input.strip().lower()
        matched = False
        for q, _ in dataset.CONVERSATIONAL_PAIRS:
            if q.lower().strip() == normalized:
                user_input = q
                matched = True
                break

        # Ensure first letter uppercase (training data style)
        if not matched and len(user_input) > 0:
            user_input = user_input[0].upper() + user_input[1:]

        prompt = f"You: {user_input}\nPerspicacity: "
        print("Perspicacity: ", end="", flush=True)
        response = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=200,
            temperature=0.7,
            top_k=50,
            repetition_penalty=1.1,
            stop_strings=["\nYou:", tokenizer.eos_token],
            stream=True,  # streaming output
            device=device,
        )
        print()  # extra newline after streaming


# Import dataset for conv_map; but to avoid circular, we use dataset.CONVERSATIONAL_PAIRS inside function.
# We'll put it inside chat() after import.
if __name__ == "__main__":
    import sys
    import dataset  # local import to avoid issues
    sys.modules[__name__].dataset = dataset  # hack to make module available
    chat()