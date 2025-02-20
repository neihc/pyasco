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
import glob
import shutil
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from io import BytesIO
import asyncio
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters, CallbackQueryHandler
from rich.console import Console
from ..config import ConfigManager
from ..agent import Agent
from ..logger_config import setup_logger
from ..services.code_to_image import CodeToImage
from ..services.code_snippet_extractor import CodeSnippetExtractor

# Maximum length for telegram messages
MAX_MESSAGE_LENGTH = 4096

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
    def __init__(self, agent: Agent, auto: bool = False):
        self.agent = agent
        self.auto = auto
        self.user_states: Dict[int, dict] = {}
        self.code_to_image = CodeToImage()
        self.code_extractor = CodeSnippetExtractor()
        self.workspace_dir = os.path.expanduser("~/.pyasco/workspace")
        self.archive_dir = os.path.join(self.workspace_dir, "archived")
        
        # Ensure directories exist
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)

    def _get_workspace_files(self) -> List[str]:
        """Get list of files in workspace directory"""
        files = []
        for file in glob.glob(os.path.join(self.workspace_dir, '*')):
            if os.path.isfile(file) and not file.startswith(self.archive_dir):
                files.append(file)
        return files

    def _is_image_file(self, filepath: str) -> bool:
        """Check if file is an image based on extension"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        return os.path.splitext(filepath)[1].lower() in image_extensions

    async def _send_workspace_files(self, message) -> List[Tuple[str, str]]:
        """Send workspace files and return list of (filename, file_id)"""
        sent_files = []
        for filepath in self._get_workspace_files():
            try:
                with open(filepath, 'rb') as f:
                    filename = os.path.basename(filepath)
                    if self._is_image_file(filepath):
                        sent_message = await message.reply_photo(
                            photo=InputFile(f, filename=filename),
                            caption=f"Workspace image: {filename}"
                        )
                        sent_files.append((filepath, sent_message.photo[-1].file_id))
                    else:
                        sent_message = await message.reply_document(
                            document=InputFile(f, filename=filename),
                            caption=f"Workspace file: {filename}"
                        )
                        sent_files.append((filepath, sent_message.document.file_id))
            except Exception as e:
                logger.error(f"Error sending file {filepath}: {str(e)}")
        return sent_files

    def _archive_files(self, files: List[str]):
        """Move files to archive directory"""
        for filepath in files:
            try:
                filename = os.path.basename(filepath)
                archive_path = os.path.join(self.archive_dir, filename)
                shutil.move(filepath, archive_path)
                logger.info(f"Archived {filename}")
            except Exception as e:
                logger.error(f"Error archiving {filepath}: {str(e)}")
    
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
            "/remember - Store current conversation in memory\n"
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

    async def remember_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Store the current conversation in memory."""
        try:
            self.agent.remember_conversation()
            await update.message.reply_text("✅ Conversation stored in memory!")
        except Exception as e:
            await update.message.reply_text(f"❌ Error storing conversation: {str(e)}")

    def _should_process_message(self, message, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """
        Check if the message should be processed based on:
        - Message is a direct message (DM)
        - Message is a reply to bot's message
        - Message mentions the bot
        """
        # Check if message is a DM (chat type is 'private')
        if message.chat.type == "private":
            return True
            
        # Check if message is a reply to bot's message
        if message.reply_to_message and message.reply_to_message.from_user.is_bot:
            return True
            
        # Check if message mentions the bot
        if message.entities:
            bot_username = context.bot.username
            for entity in message.entities:
                if entity.type == "mention":
                    # Extract mention text
                    mention = message.text[entity.offset:entity.offset + entity.length]
                    if mention.lower() == f"@{bot_username.lower()}":
                        return True
        
        return False

    async def handle_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()  # Acknowledge the button click
        
        if query.data == "execute_yes":
            # Execute the pending code
            results = self.agent.confirm()
            if results:
                output_message = "Execution Output:\n"
                for result in results:
                    output_message += f"{result}\n"
                
                # If output is too long, send as file
                if len(output_message) > MAX_MESSAGE_LENGTH:
                    # Create file-like object
                    output_file = BytesIO(output_message.encode('utf-8'))
                    await query.message.reply_document(
                        document=InputFile(output_file, filename='output.txt'),
                        caption="Execution output (sent as file due to length)"
                    )
                else:
                    await query.message.reply_text(output_message)
                
                # Send any workspace files that were generated
                sent_files = await self._send_workspace_files(query.message)
                if sent_files:
                    self._archive_files([f[0] for f in sent_files])
                
                # Handle follow-up if needed
                follow_up = self.agent.get_follow_up(results)
                if follow_up:
                    response = self.agent.get_response(follow_up, stream=False)
                    await query.message.reply_text(response.content)
                    
                    # If there's more code to execute, ask again
                    if self.agent.should_ask_user():
                        reply_markup = {
                            'inline_keyboard': [[
                                {'text': 'Yes ✅', 'callback_data': 'execute_yes'},
                                {'text': 'No ❌', 'callback_data': 'execute_no'}
                            ]]
                        }
                        await query.message.reply_text(
                            "Do you want to execute the code snippets?",
                            reply_markup=reply_markup
                        )
        
        elif query.data == "execute_no":
            # Cancel the execution
            response = self.agent.get_response("no", stream=False)
            await query.message.reply_text(response.content)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages."""
        if not update or not update.message:
            logger.error("Received update with no message")
            return
            
        # Only process messages that are replies or mentions
        if not self._should_process_message(update.message, context):
            return
            
        user_id = update.effective_user.id
        logger.info(f"Received message from user {user_id}: {update.message.text}")
        
        if user_id not in self.user_states:
            logger.debug(f"Initializing state for new user {user_id}")
            self.user_states[user_id] = {}

        user_input = update.message.text
        try:
            # Send processing message
            processing_message = await update.message.reply_text("Processing your message... 🤔")
            
            # Get response from agent
            logger.debug("Sending request to agent")
            response = await self.agent.ask(user_input, stream=False, auto=self.auto)
            logger.debug(f"Got response from agent: {response.content}")

            # Check for workspace files first
            sent_files = await self._send_workspace_files(update.message)
            if sent_files:
                self._archive_files([f[0] for f in sent_files])

            # If in auto mode, handle execution automatically
            if self.auto and self.agent.should_ask_user():
                results = self.agent.confirm()
                if results:
                    output_message = "Execution Output:\n"
                    for result in results:
                        output_message += f"{result}\n"
                    
                    # Send execution output
                    if len(output_message) > MAX_MESSAGE_LENGTH:
                        output_file = BytesIO(output_message.encode('utf-8'))
                        await update.message.reply_document(
                            document=InputFile(output_file, filename='output.txt'),
                            caption="Execution output (sent as file due to length)"
                        )
                    else:
                        await update.message.reply_text(output_message)
                    
                    # Check for any new workspace files after execution
                    sent_files = await self._send_workspace_files(update.message)
                    if sent_files:
                        self._archive_files([f[0] for f in sent_files])
            
            # Delete processing message
            await processing_message.delete()
            
            # Prepare all messages to send
            messages_to_send = []
            
            # Extract code snippets and convert to images
            snippets = self.code_extractor.extract_snippets(response.content)
            
            if snippets:
                # Get text response with code blocks removed
                text_response = self.code_extractor.omit_snippets(response.content)
                
                # Add cleaned text response if not empty
                if text_response:
                    messages_to_send.append(("text", text_response, None))
                
                # Prepare code snippets as images
                for snippet in snippets:
                    image_bytes = self.code_to_image.convert(
                        snippet.content,
                        snippet.language
                    )
                    messages_to_send.append(
                        ("photo", 
                         image_bytes, 
                         f"Code snippet ({snippet.language or 'unknown language'})")
                    )
            else:
                # No code snippets, just text response
                messages_to_send.append(("text", response.content, None))
            
            # Send all messages at once
            for msg_type, content, caption in messages_to_send:
                if msg_type == "text":
                    await update.message.reply_text(content)
                elif msg_type == "photo":
                    await update.message.reply_photo(
                        InputFile(content, filename='code.png'),
                        caption=caption
                    )
            
            logger.debug(f"Sent {len(messages_to_send)} messages to user")

            # If there's code to execute, ask user only if not in auto mode
            if self.agent.should_ask_user() and not self.auto:
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
    parser.add_argument("--auto", action="store_true",
                       help="Automatically execute code without asking user")
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
    interface = TelegramInterface(agent, auto=args.auto)
    
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
    application.add_handler(CommandHandler("remember", interface.remember_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,
                                         interface.handle_message))
    application.add_handler(CallbackQueryHandler(interface.handle_button))
    
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
