#!/usr/bin/env python3
"""
Perspicacity Next - Main entry point.
Supports training, inference, and migration.
"""

import argparse
import sys

from perspicacity_next import train, inference, migrate


def main():
    parser = argparse.ArgumentParser(description="Perspicacity Next - Llama-based LLM")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

    # Train
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--resume", type=str, help="Resume from a specific checkpoint directory")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Chat with the model")

    # Migrate
    migrate_parser = subparsers.add_parser("migrate", help="Migrate old checkpoint to new architecture")
    migrate_parser.add_argument("old_checkpoint", type=str, help="Path to the old .pt checkpoint")
    migrate_parser.add_argument("--output_dir", type=str, default="migrated_model", help="Output directory for new model")

    args = parser.parse_args()

    if args.command == "train":
        train.train(resume_from=args.resume)
    elif args.command == "chat":
        inference.chat()
    elif args.command == "migrate":
        migrate.migrate(args.old_checkpoint, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()