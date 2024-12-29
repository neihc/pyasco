"""
PyAsco Telegram Bot - An AI-powered Python assistant on Telegram

This module provides a Telegram bot interface for the PyAsco AI assistant,
capable of executing Python code, managing skills, and providing intelligent responses.

Usage:
    python -m pyasco.app.telegram_bot [options]

Options:
    --config PATH          Path to YAML configuration file
    --telegram-token TEXT  Telegram bot token
    All other options from console.py are supported
"""

import argparse
import os
import logging
from typing import Optional, Dict
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from rich.console import Console
from ..config import ConfigManager
from ..agent import Agent
from ..logger_config import setup_logger

# Setup logging
logger = setup_logger(__name__)
console = Console()

class TelegramInterface:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.user_states: Dict[int, dict] = {}
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        welcome_message = (
            "Welcome to PyAsco Bot! 🤖\n\n"
            "I can help you with Python programming and execute code.\n\n"
            "Available commands:\n"
            "/reset - Start over\n"
            "/learn - Convert current conversation into a reusable skill\n"
            "/improve - Improve an existing skill\n"
            "/help - Show this help message"
        )
        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /help is issued."""
        await self.start_command(update, context)

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Reset the agent when /reset command is issued."""
        user_id = update.effective_user.id
        self.agent.reset()
        if user_id in self.user_states:
            self.user_states[user_id] = {}
        await update.message.reply_text("Chat history has been reset! 🔄")

    async def learn_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Learn a new skill from the conversation."""
        try:
            skill = self.agent.learn_that_skill()
            await update.message.reply_text(
                f"✅ Learned new skill: {skill.name}\n"
                f"Usage: {skill.usage}"
            )
        except ValueError as e:
            await update.message.reply_text(f"❌ Error learning skill: {str(e)}")

    async def improve_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Improve an existing skill."""
        try:
            skill = self.agent.improve_that_skill()
            if skill:
                await update.message.reply_text(
                    f"✅ Improved skill: {skill.name}\n"
                    f"New usage: {skill.usage}"
                )
            else:
                await update.message.reply_text("❌ No skill to improve")
        except ValueError as e:
            await update.message.reply_text(f"❌ Error improving skill: {str(e)}")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages."""
        user_id = update.effective_user.id
        if user_id not in self.user_states:
            self.user_states[user_id] = {}

        user_input = update.message.text
        try:
            # Get response from agent
            response = self.agent.get_response(user_input, stream=False)
            
            # Send the response
            await update.message.reply_text(response.content)

            # If there's code to execute, ask user
            if self.agent.should_ask_user():
                reply_markup = {
                    'inline_keyboard': [[
                        {'text': 'Yes ✅', 'callback_data': 'execute_yes'},
                        {'text': 'No ❌', 'callback_data': 'execute_no'}
                    ]]
                }
                await update.message.reply_text(
                    "Do you want to execute the code snippets?",
                    reply_markup=reply_markup
                )

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await update.message.reply_text(
                "Sorry, I encountered an error while processing your message. 😕"
            )

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="PyAsco Telegram Bot")
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--telegram-token", required=True,
                       help="Telegram bot token")
    parser.add_argument("--use-docker", action="store_true",
                       help="Run code in Docker")
    parser.add_argument("--docker-image", default="python:3.9-slim",
                       help="Docker image to use")
    parser.add_argument("--mem-limit", default="512m",
                       help="Docker memory limit (e.g., 512m, 1g)")
    parser.add_argument("--cpu-count", type=int, default=1,
                       help="Docker CPU count")
    parser.add_argument("--env-file",
                       help="Path to environment file for Docker container")
    parser.add_argument("--mount", action='append',
                       help="Mount points in format 'host_path:container_path'")
    parser.add_argument("--model", default="meta-llama/llama-3.3-70b-instruct",
                       help="LLM model to use for responses")
    parser.add_argument("--skills-path", default="skills",
                       help="Path to skills directory")
    return parser.parse_args()

async def main():
    """Main function to run the Telegram bot"""
    args = parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = ConfigManager.load_from_yaml(args.config)
    else:
        config = ConfigManager.from_args(args)
    
    # Initialize agent
    agent = Agent(config)
    interface = TelegramInterface(agent)
    
    # Initialize bot
    application = Application.builder().token(args.telegram_token).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", interface.start_command))
    application.add_handler(CommandHandler("help", interface.help_command))
    application.add_handler(CommandHandler("reset", interface.reset_command))
    application.add_handler(CommandHandler("learn", interface.learn_command))
    application.add_handler(CommandHandler("improve", interface.improve_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,
                                         interface.handle_message))
    
    try:
        # Start the bot
        await application.initialize()
        await application.start()
        await application.run_polling()
    finally:
        await application.stop()
        agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
