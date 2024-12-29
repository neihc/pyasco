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

# Setup logging will be done in main() after parsing args
logger = logging.getLogger(__name__)
console = Console()

# Add file handler for detailed logging
file_handler = logging.FileHandler('telegram_bot.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

class TelegramInterface:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.user_states: Dict[int, dict] = {}
    
    async def ping_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Simple command to test if bot is responsive"""
        logger.debug("Received ping command")
        await update.message.reply_text("Pong! 🏓")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send a message when the command /start is issued."""
        logger.debug("Received start command")
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
        if not update or not update.message:
            logger.error("Received update with no message")
            return
            
        user_id = update.effective_user.id
        logger.info(f"Received message from user {user_id}: {update.message.text}")
        
        if user_id not in self.user_states:
            logger.debug(f"Initializing state for new user {user_id}")
            self.user_states[user_id] = {}

        user_input = update.message.text
        try:
            # Get response from agent
            logger.debug("Sending request to agent")
            response = self.agent.get_response(user_input, stream=False)
            logger.debug(f"Got response from agent: {response.content}")
            
            # Send the response
            await update.message.reply_text(response.content)
            logger.debug("Sent response to user")

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
            logger.error(f"Error processing message from user {user_id}: {str(e)}", exc_info=True)
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
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Set the logging level")
    return parser.parse_args()

def main():
    """Main function to run the Telegram bot"""
    args = parse_args()
    
    # Setup logging with command line specified level
    log_level = getattr(logging, args.log_level.upper())
    logger = setup_logger(__name__, log_file='telegram_bot.log', level=log_level)
    
    if not args.telegram_token:
        logger.error("Telegram token is missing!")
        return
        
    logger.info(f"Starting bot with configuration from: {args.config if args.config else 'command line arguments'}")
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        config = ConfigManager.load_from_yaml(args.config)
    else:
        logger.info("Loading configuration from command line arguments")
        config = ConfigManager.from_args(args)
    
    # Initialize agent
    logger.info("Initializing agent...")
    agent = Agent(config)
    interface = TelegramInterface(agent)
    
    # Initialize bot
    logger.info("Setting up Telegram bot...")
    try:
        application = Application.builder().token(args.telegram_token).build()
        logger.info("Successfully created Telegram application")
    except Exception as e:
        logger.error(f"Failed to create Telegram application: {str(e)}", exc_info=True)
        return
    
    # Add handlers
    logger.info("Registering command handlers...")
    application.add_handler(CommandHandler("ping", interface.ping_command))
    application.add_handler(CommandHandler("start", interface.start_command))
    application.add_handler(CommandHandler("help", interface.help_command))
    application.add_handler(CommandHandler("reset", interface.reset_command))
    application.add_handler(CommandHandler("learn", interface.learn_command))
    application.add_handler(CommandHandler("improve", interface.improve_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,
                                         interface.handle_message))
    
    try:
        logger.info("Starting bot polling...")
        print("Bot started successfully!")
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.error(f"Failed to start bot: {str(e)}", exc_info=True)
    finally:
        logger.info("Cleaning up...")
        agent.cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped by user")
