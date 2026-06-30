"""
Training loop for Perspicacity Next.
Uses a custom loop with gradient accumulation, mixed precision, and cosine scheduler.
"""

import os
import signal
import time
import argparse
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from transformers import LlamaForCausalLM, AutoTokenizer

from .config import (
    CHECKPOINT_DIR,
    OPTIMIZER_STATE_FILE,
    MAX_ITERS,
    BATCH_SIZE,
    GRAD_ACCUM,
    EVAL_INTERVAL,
    SAVE_INTERVAL,
    LEARNING_RATE,
    MIN_LR,
    WARMUP_ITERS,
    WEIGHT_DECAY,
    USE_GRAD_CKPT,
    USE_COMPILE,
    MAX_POSITION_EMBEDDINGS,
    create_llama_config,
)
from .tokenizer import get_tokenizer
from .dataset import build_chat_samples, tokenize_samples
from .inference import generate_response  # for eval sampling


def get_lr(step: int) -> float:
    """Cosine decay with warmup."""
    if step < WARMUP_ITERS:
        return LEARNING_RATE * (step + 1) / WARMUP_ITERS
    if step >= MAX_ITERS:
        return MIN_LR
    decay_ratio = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * decay_ratio)).item())
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, output_dir: str):
    """Save model and optimizer state."""
    os.makedirs(output_dir, exist_ok=True)
    # Save model using Hugging Face's save_pretrained
    model.save_pretrained(output_dir)
    # Save optimizer and step separately
    opt_path = os.path.join(output_dir, OPTIMIZER_STATE_FILE)
    torch.save({
        'optimizer': optimizer.state_dict(),
        'step': step,
    }, opt_path)
    print(f"[CHECKPOINT] Saved checkpoint at step {step} to {output_dir}")


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, load_dir: str) -> int:
    """Load model and optimizer from checkpoint. Returns starting step."""
    # Load model using from_pretrained
    # We assume the model is already created, but we can override its state.
    # Better to load the model weights directly.
    # We'll use model.load_state_dict from the saved model's pytorch_model.bin
    # but we can also use from_pretrained. To keep it simple, we'll load the state dict.
    # However, since we have the model object, we can load its state dict.
    model_path = os.path.join(load_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        # maybe the model was saved as safetensors? We'll check.
        # We'll use from_pretrained to be safe.
        print(f"[CHECKPOINT] Loading model from {load_dir}...")
        # We'll reload the model (but we already have an instance)
        # We'll load the state dict directly.
        state_dict = torch.load(model_path, map_location="cpu")
        # Remove unexpected keys like 'lm_head.weight' if tied
        model.load_state_dict(state_dict, strict=False)
    # Load optimizer
    opt_path = os.path.join(load_dir, OPTIMIZER_STATE_FILE)
    step = 0
    if os.path.exists(opt_path):
        print(f"[CHECKPOINT] Loading optimizer from {opt_path}...")
        opt_state = torch.load(opt_path, map_location="cpu")
        optimizer.load_state_dict(opt_state['optimizer'])
        step = opt_state.get('step', 0)
    return step


def train(resume_from: Optional[str] = None):
    """
    Main training function.
    If resume_from is provided, load from that directory.
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRAIN] Using device: {device}")

    # Load tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    print(f"[TOKENIZER] Vocab size: {vocab_size}")

    # Build dataset
    samples, eval_questions = build_chat_samples()
    print("[TOKENIZE] Tokenizing samples...")
    t0 = time.time()
    all_tokens = tokenize_samples(samples, tokenizer)
    data = torch.tensor(all_tokens, dtype=torch.long)
    print(f"[TOKENIZE] {len(data):,} tokens in {time.time() - t0:.1f}s")

    # Create model config
    config = create_llama_config(vocab_size)
    model = LlamaForCausalLM(config).to(device)

    # Print param count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Enable gradient checkpointing if requested
    if USE_GRAD_CKPT:
        model.gradient_checkpointing_enable()
        print("[MODEL] Gradient checkpointing enabled")

    # Compile if requested
    if USE_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("[MODEL] torch.compile enabled")

    # Optimizer: separate weight decay for biases etc. (Llama uses AdamW)
    # We'll use a simpler approach: all parameters with weight_decay (except biases and LayerNorm).
    # We'll use the same as typical transformers: weight_decay on all parameters except bias and norm.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Learning rate scheduler (cosine with warmup)
    def lr_lambda(step: int) -> float:
        return get_lr(step) / LEARNING_RATE  # normalization

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    use_amp = (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[TRAIN] AMP: {amp_dtype if use_amp else 'off'}")

    # Resume if directory exists
    start_step = 0
    if resume_from is not None and os.path.exists(resume_from):
        start_step = load_checkpoint(model, optimizer, resume_from)
        print(f"[CHECKPOINT] Resuming from step {start_step}")
        # Scheduler state not saved; we'll just step to the right point
        for _ in range(start_step):
            scheduler.step()
    else:
        # Possibly auto-resume from default directory
        if os.path.exists(CHECKPOINT_DIR):
            start_step = load_checkpoint(model, optimizer, CHECKPOINT_DIR)
            print(f"[CHECKPOINT] Auto-resume from {CHECKPOINT_DIR} at step {start_step}")
            for _ in range(start_step):
                scheduler.step()

    # Batch sampler
    def get_batch():
        ix = torch.randint(0, len(data) - MAX_POSITION_EMBEDDINGS - 1, (BATCH_SIZE,))
        x = torch.stack([data[i:i+MAX_POSITION_EMBEDDINGS] for i in ix])
        y = torch.stack([data[i+1:i+MAX_POSITION_EMBEDDINGS+1] for i in ix])
        return x.to(device), y.to(device)

    # Training loop
    print(f"[TRAIN] Training from step {start_step} to {MAX_ITERS}")
    print(f"[TRAIN] Effective batch size: {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print("[TRAIN] Press Ctrl+C anytime to save and stop.")

    model.train()
    t_start = time.time()
    tokens_processed = 0
    current_step = start_step
    stop_requested = False

    def signal_handler(sig, frame):
        nonlocal stop_requested
        if stop_requested:
            print("\n[STOP] Force quit.")
            raise SystemExit(1)
        stop_requested = True
        print("\n[STOP] Ctrl+C — saving checkpoint after current step...")

    signal.signal(signal.SIGINT, signal_handler)

    try:
        for step in range(start_step, MAX_ITERS):
            current_step = step
            if stop_requested:
                break

            # Set learning rate
            lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Gradient accumulation
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(GRAD_ACCUM):
                xb, yb = get_batch()
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(xb, labels=yb)
                    loss = outputs.loss / GRAD_ACCUM
                scaler.scale(loss).backward()
                accum_loss += loss.item() * GRAD_ACCUM  # undo division for logging

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            tokens_processed += BATCH_SIZE * MAX_POSITION_EMBEDDINGS * GRAD_ACCUM

            # Evaluation
            if step % EVAL_INTERVAL == 0:
                elapsed = time.time() - t_start
                tok_s = tokens_processed / elapsed if elapsed > 0 else 0
                eta_h = (MAX_ITERS - step) / (step - start_step + 1) * elapsed / 3600 if step > start_step else 0

                print(f"Step {step:>6d} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | Tok/s: {tok_s:,.0f} | ETA: {eta_h:.1f}h")

                # Generate sample
                eval_query = random.choice(eval_questions)
                prompt = f"You: {eval_query}\nPerspicacity: "
                response = generate_response(
                    model, tokenizer, prompt,
                    max_new_tokens=40,
                    temperature=0.7,
                    top_k=50,
                    stop_strings=["\nYou:", tokenizer.eos_token],
                    device=device
                )
                print("-" * 60)
                print(f"You: {eval_query}")
                print(f"Perspicacity: {response.strip()}")
                print("-" * 60)
                print()
                model.train()

            # Save checkpoint
            if step > 0 and step % SAVE_INTERVAL == 0:
                save_checkpoint(model, optimizer, step, CHECKPOINT_DIR)

    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt caught.")

    # Final save
    save_checkpoint(model, optimizer, current_step, CHECKPOINT_DIR)
    total_time = time.time() - t_start
    print(f"[DONE] Stopped at step {current_step} after {total_time / 3600:.1f}h")
    print(f"[DONE] Model saved to {CHECKPOINT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, help="Directory to resume from (default: checkpoints)")
    args = parser.parse_args()
    train(resume_from=args.resume)