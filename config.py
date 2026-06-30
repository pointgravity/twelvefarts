"""
Configuration for Perspicacity Next.
All hyperparameters and model configuration are defined here.
"""

from transformers import LlamaConfig

# ---------------------------------------------------------------------
# Model architecture (Llama)
# ---------------------------------------------------------------------
# These values match the original microLLM_v3 dimensions as closely as possible.
HIDDEN_SIZE = 512
NUM_LAYERS = 8
NUM_HEADS = 8
NUM_KV_HEADS = 4          # GQA: 8 heads, 4 KV heads
INTERMEDIATE_SIZE = 2048  # 4 * hidden_size (typical for Llama)
MAX_POSITION_EMBEDDINGS = 512  # block_size
VOCAB_SIZE = 32000        # will be updated after loading tokenizer
RMS_NORM_EPS = 1e-6
ROPE_THETA = 10000.0
TIE_WORD_EMBEDDINGS = True
DROPOUT = 0.1

# ---------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------
BATCH_SIZE = 8             # per micro‑step
GRAD_ACCUM = 4             # effective batch = 8*4 = 32
MAX_ITERS = 100000
EVAL_INTERVAL = 1          # show sample every step
SAVE_INTERVAL = 2000
LEARNING_RATE = 3e-4
MIN_LR = 1e-5
WARMUP_ITERS = 500
WEIGHT_DECAY = 0.1
USE_GRAD_CKPT = True       # gradient checkpointing
USE_COMPILE = False        # torch.compile (sometimes buggy)

# ---------------------------------------------------------------------
# Checkpoint / output directories
# ---------------------------------------------------------------------
CHECKPOINT_DIR = "checkpoints"       # where to save/load model
OPTIMIZER_STATE_FILE = "optimizer.pt"  # saved inside CHECKPOINT_DIR

# ---------------------------------------------------------------------
# Tokenizer settings
# ---------------------------------------------------------------------
# Use a public Llama tokenizer (no gating)
TOKENIZER_NAME = "huggyllama/llama-7b"

# ---------------------------------------------------------------------
# LlamaConfig object (constructed after tokenizer is loaded)
# ---------------------------------------------------------------------
def create_llama_config(vocab_size: int) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        rms_norm_eps=RMS_NORM_EPS,
        rope_theta=ROPE_THETA,
        tie_word_embeddings=TIE_WORD_EMBEDDINGS,
        use_cache=True,
        hidden_dropout=DROPOUT,
        attention_dropout=DROPOUT,
        # Flash attention is enabled via PyTorch SDPA by default
    )