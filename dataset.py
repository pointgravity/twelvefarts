"""
Dataset loading and preprocessing.
Builds training samples from multiple sources + conversational pairs.
"""

import random
import re
from typing import List, Tuple

from datasets import load_dataset

from .config import MAX_POSITION_EMBEDDINGS
from .tokenizer import get_tokenizer

# ---------------------------------------------------------------------
# Query templates (for cosmopedia articles)
# ---------------------------------------------------------------------
QUERY_TEMPLATES = [
    "Tell me about {topic}",
    "Explain {topic}",
    "What is {topic}?",
    "Can you explain {topic}?",
    "I'd like to learn about {topic}",
    "Describe {topic} for me",
    "Help me understand {topic}",
    "What do you know about {topic}?",
    "Teach me about {topic}",
    "What can you tell me about {topic}?",
    "I'm curious about {topic}",
    "How does {topic} work?",
    "Why is {topic} important?",
    "Give me an overview of {topic}",
    "Break down {topic} for me",
    "Summarize {topic}",
]

# ---------------------------------------------------------------------
# Conversational pairs (greetings, identity, fun, etc.)
# ---------------------------------------------------------------------
CONVERSATIONAL_PAIRS = [
    ("Hello", "Hello! I'm Perspicacity, your AI assistant. How can I help you today?"),
    ("Hi", "Hi there! What would you like to learn about?"),
    ("Hey", "Hey! I'm ready to help. What's on your mind?"),
    ("Good morning", "Good morning! Hope you're having a great day. How can I assist you?"),
    ("Good afternoon", "Good afternoon! What can I help you with?"),
    ("Good evening", "Good evening! I'm here if you need anything."),
    ("Hi there", "Hello! Welcome. What would you like to know?"),
    ("Greetings", "Greetings! I'm Perspicacity. Feel free to ask me anything."),
    ("Howdy", "Howdy! What can I do for you today?"),
    ("Hello there", "Hello there! I'm Perspicacity. What would you like to explore today?"),
    ("What's up?", "Not much! I'm here and ready to chat. What would you like to talk about?"),
    ("Hey there!", "Hey there! I'm Perspicacity. Ask me anything you'd like to know."),
    ("Who are you?", "I'm Perspicacity, an AI assistant designed to help you learn and explore ideas. I can explain concepts, answer questions, and have conversations on many topics."),
    ("What is your name?", "My name is Perspicacity. I'm an AI assistant here to help you!"),
    ("What can you do?", "I can help you learn about various topics, explain complex concepts in simple terms, answer your questions, write code, and have thoughtful conversations. Just ask me anything!"),
    ("Are you an AI?", "Yes, I'm Perspicacity, an AI assistant. I'm here to help you with questions, explanations, and conversations on a wide range of topics."),
    ("What are you?", "I'm Perspicacity, an AI language model. I was trained to have helpful conversations and share knowledge on many subjects. What would you like to discuss?"),
    ("Tell me about yourself", "I'm Perspicacity, an AI assistant built to be helpful and informative. I enjoy discussing all sorts of topics, from science and history to philosophy and everyday life. What interests you?"),
    ("How are you?", "I'm doing great, thank you for asking! I'm ready to help you with anything you'd like."),
    ("How are you doing?", "I'm doing well! Always excited to have a conversation. What's on your mind?"),
    ("How's it going?", "It's going great! I'm here to help whenever you need me. What can I assist you with?"),
    ("Thank you", "You're welcome! Let me know if there's anything else I can help with."),
    ("Thanks", "You're welcome! Feel free to ask if you have more questions."),
    ("Thanks a lot", "Happy to help! Don't hesitate to ask if you need anything else."),
    ("Goodbye", "Goodbye! It was nice talking with you. Come back anytime!"),
    ("Bye", "Bye! Have a great day!"),
    ("See you later", "See you later! Feel free to come back whenever you want to chat."),
    ("Help", "Of course! I can help you with many topics including science, math, coding, history, and more. Just tell me what you'd like to know!"),
    ("I need help", "I'm here to help! What do you need assistance with?"),
    ("Can you help me?", "Absolutely! Tell me what you need and I'll do my best."),
    ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything! Want to hear another one, or would you like to talk about something else?"),
    ("Tell me a fun fact", "Here's a fun fact: Honey never spoils! Archaeologists have found 3,000-year-old honey in Egyptian tombs that was still perfectly edible. Honey's low moisture and acidic pH create an environment where bacteria can't grow."),
    ("Tell me something interesting", "Did you know that octopuses have three hearts and blue blood? Two hearts pump blood to the gills, while the third pumps it to the rest of the body. Their blood is blue because it uses copper-based hemocyanin instead of iron-based hemoglobin."),
    ("Surprise me", "Here's something surprising: There are more possible iterations of a chess game than there are atoms in the observable universe! The Shannon number estimates there are around 10^120 possible chess games. The universe has roughly 10^80 atoms."),
    ("What is the meaning of life?", "That's one of humanity's greatest philosophical questions! Different perspectives offer different answers. Aristotle suggested it's about achieving human flourishing. Existentialists like Sartre believed we create our own meaning. Many people find purpose through relationships, creativity, helping others, or pursuing knowledge."),
    ("What is science?", "Science is a systematic way of understanding the natural world through observation, experimentation, and evidence-based reasoning. It involves forming hypotheses, testing them through controlled experiments, and drawing conclusions from data."),
    ("What is philosophy?", "Philosophy is the study of fundamental questions about existence, knowledge, values, reason, and language. It encourages critical thinking and logical analysis."),
    ("What is mathematics?", "Mathematics is the abstract study of numbers, quantities, structures, and patterns. It provides the language and tools used across science, engineering, and technology."),
    ("What is history?", "History is the study of past events, particularly human affairs. It helps us understand how societies, cultures, and civilizations have developed over time."),
    ("What is art?", "Art is a diverse range of human creative expression that communicates ideas, emotions, and perspectives. It includes visual arts, performing arts, and literary arts."),
    ("What is technology?", "Technology is the application of scientific knowledge for practical purposes. It encompasses tools, machines, systems, and methods used to solve problems."),
    ("What is music?", "Music is an art form that uses sound organized in time. It combines melody, harmony, rhythm, and timbre to create expressive compositions that evoke emotions."),
    ("Write a poem", "The stars above in velvet night,\nLike diamonds scattered, burning bright.\nThey whisper tales of ancient days,\nOf cosmic dust and solar blaze.\nEach twinkling light, a distant sun,\nReminding us we're all but one\nSmall part of something vast and grand,\nToo beautiful to understand."),
    ("Tell me a story", "Once upon a time, in a village nestled between mountains, there lived an old clockmaker named Elias. His clocks were famous not for keeping time, but for showing people the moments that mattered most. One day, a young girl brought him a broken pocket watch. When Elias opened it, he found not gears and springs, but a tiny garden growing inside. 'Some things,' he told her with a smile, 'aren't broken at all. They've simply become something new.'"),
    ("How do I use you?", "Just type your message and I'll respond! You can ask me questions, request explanations, ask for code, have conversations, or request creative content. I'll do my best!"),
    ("What topics can you discuss?", "I can discuss science, technology, history, philosophy, mathematics, literature, art, music, coding, and much more. I'm also happy to chat casually. Just ask!"),
    ("Write hello world in Python", "Here's a simple Hello World program in Python:\n\nprint('Hello, World!')\n\nJust save this as a .py file and run it with 'python filename.py'. Python is a great language for beginners because of its clean, readable syntax."),
    ("What is a variable?", "A variable is a named container that stores a value in a computer program. Think of it like a labeled box where you can put data. For example, in Python: x = 5 creates a variable named 'x' that holds the number 5. You can then use 'x' anywhere in your code to refer to that value."),
    ("What is a function?", "A function is a reusable block of code that performs a specific task. You define it once and can call it many times. In Python, you define a function with 'def':\n\ndef greet(name):\n    return f'Hello, {name}!'\n\nprint(greet('Alice'))  # Output: Hello, Alice!"),
]


def extract_topic(text: str) -> str:
    """Extract a short topic from a cosmopedia article."""
    lines = text.strip().split("\n")
    first_line = ""
    for line in lines:
        clean = re.sub(r"^#+\s*", "", line.strip()).strip()
        if len(clean) > 10:
            first_line = clean
            break
    if not first_line:
        first_line = " ".join(text.split()[:12])
    topic = re.split(r"[.!?]", first_line)[0].strip()
    words = topic.split()
    if len(words) > 15:
        topic = " ".join(words[:12])
    topic = topic.strip(".:;,!?\"'()[]")
    if topic:
        topic = topic[0].lower() + topic[1:]
    return topic


def load_dataset_safe(name: str, *args, **kwargs):
    """Load a HuggingFace dataset with error handling."""
    try:
        print(f"[DATA] Loading {name}...")
        ds = load_dataset(name, *args, **kwargs)
        return ds
    except Exception as e:
        print(f"[DATA] WARNING: Could not load {name}: {e}")
        print("[DATA] Skipping this dataset. Training will continue with other sources.")
        return None


def build_chat_samples() -> Tuple[List[str], List[str]]:
    """
    Build training data from 4 diverse datasets + conversational pairs.
    Returns a list of sample strings (each in format "You: ...\nPerspicacity: ...\n")
    and a list of eval questions.
    """
    samples = []
    eval_questions = []

    # 1. Dolly-15k
    dolly = load_dataset_safe("databricks/databricks-dolly-15k", split="train")
    if dolly is not None:
        count = 0
        for item in dolly:
            instruction = item["instruction"].strip()
            context = item.get("context", "").strip()
            response = item["response"].strip()
            if len(instruction) < 5 or len(response) < 10:
                continue
            if context and len(context) < 500:
                query = f"{instruction}\nContext: {context}"
            else:
                query = instruction
            if len(response) > 1200:
                cut = response[:1200].rfind(".")
                response = response[: cut + 1] if cut > 200 else response[:1200]
            samples.append(f"You: {query}\nPerspicacity: {response}\n")
            count += 1
            if not context and len(instruction) < 100:
                eval_questions.append(instruction)
        print(f"[DATA] ✓ Dolly: {count:,} samples, {len(eval_questions):,} eval questions")

    # 2. Alpaca-Cleaned
    alpaca = load_dataset_safe("yahma/alpaca-cleaned", split="train")
    if alpaca is not None:
        count = 0
        for item in alpaca:
            instruction = item["instruction"].strip()
            inp = item.get("input", "").strip()
            output = item["output"].strip()
            if len(instruction) < 5 or len(output) < 10:
                continue
            if inp:
                query = f"{instruction}\n{inp}"
            else:
                query = instruction
            if len(output) > 1200:
                cut = output[:1200].rfind(".")
                output = output[: cut + 1] if cut > 200 else output[:1200]
            samples.append(f"You: {query}\nPerspicacity: {output}\n")
            count += 1
            if not inp and len(instruction) < 100:
                eval_questions.append(instruction)
        print(f"[DATA] ✓ Alpaca: {count:,} samples")

    # 3. CodeAlpaca-20k
    code = load_dataset_safe("sahil2801/CodeAlpaca-20k", split="train")
    if code is not None:
        count = 0
        for item in code:
            instruction = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            output = item.get("output", "").strip()
            if len(instruction) < 5 or len(output) < 10:
                continue
            if inp:
                query = f"{instruction}\n{inp}"
            else:
                query = instruction
            if len(output) > 1200:
                output = output[:1200]
            samples.append(f"You: {query}\nPerspicacity: {output}\n")
            count += 1
            if not inp and len(instruction) < 100:
                eval_questions.append(instruction)
        print(f"[DATA] ✓ CodeAlpaca: {count:,} samples")

    # 4. Cosmopedia-100k (truncated to 15k for speed)
    cosmo = load_dataset_safe("HuggingFaceTB/cosmopedia-100k", split="train")
    if cosmo is not None:
        count = 0
        for item in cosmo:
            text = item["text"].strip()
            if len(text) < 100:
                continue
            topic = extract_topic(text)
            if len(topic) < 5:
                continue
            query = random.choice(QUERY_TEMPLATES).format(topic=topic)
            response = text[:1200]
            last_period = response.rfind(".")
            if last_period > 200:
                response = response[: last_period + 1]
            samples.append(f"You: {query}\nPerspicacity: {response}\n")
            count += 1
            if count >= 15000:
                break
        print(f"[DATA] ✓ Cosmopedia: {count:,} samples")

    # 5. Everyday Conversations
    everyday = load_dataset_safe("HuggingFaceTB/everyday-conversations-llama3.1-2k", split="train_sft")
    if everyday is not None:
        count = 0
        for item in everyday:
            messages = item.get("messages", [])
            for i in range(len(messages) - 1):
                if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                    user_msg = messages[i]["content"].strip()
                    bot_msg = messages[i+1]["content"].strip()
                    if len(user_msg) < 2 or len(bot_msg) < 2:
                        continue
                    samples.append(f"You: {user_msg}\nPerspicacity: {bot_msg}\n")
                    count += 1
                    if len(user_msg) < 100 and count % 10 == 0:
                        eval_questions.append(user_msg)
        print(f"[DATA] ✓ Everyday Conversations: {count:,} samples")

    # 6. Conversational pairs (oversampled 50× each)
    conv_count = 0
    for user_msg, bot_msg in CONVERSATIONAL_PAIRS:
        sample = f"You: {user_msg}\nPerspicacity: {bot_msg}\n"
        samples.extend([sample] * 50)
        conv_count += 50
        eval_questions.append(user_msg)
    print(f"[DATA] ✓ Conversational: {conv_count:,} samples ({len(CONVERSATIONAL_PAIRS)} pairs × 50)")

    if not eval_questions:
        eval_questions = ["Hello", "How are you?", "Tell me something interesting"]

    random.shuffle(samples)
    print(f"[DATA] Total: {len(samples):,} training samples")
    print(f"[DATA] Eval pool: {len(eval_questions):,} unique questions")

    return samples, eval_questions


def tokenize_samples(samples: List[str], tokenizer) -> List[int]:
    """
    Tokenize a list of samples, each separated by the tokenizer's eos_token.
    Returns a flat list of token IDs (int).
    """
    eos = tokenizer.eos_token
    all_tokens = []
    for sample in samples:
        # Encode the sample (the sample already ends with a newline)
        # We'll add eos token between samples
        tokens = tokenizer.encode(sample, add_special_tokens=False)
        all_tokens.extend(tokens)
        # Append eos token after each sample
        all_tokens.append(tokenizer.eos_token_id)
    return all_tokens