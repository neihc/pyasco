"""
Test script to verify Deepgram API connectivity
"""

import os
import asyncio
from dotenv import load_dotenv
from deepgram import DeepgramClient
from rich.console import Console

console = Console()

async def test_deepgram_connection():
    """Test if the Deepgram API key is valid and the connection works"""
    
    console.print("[bold green]Testing Deepgram connection...[/bold green]")
    
    # Load environment variables
    load_dotenv()
    
    # Get Deepgram API key
    deepgram_key = os.getenv('DEEPGRAM_API_KEY')
    if not deepgram_key:
        console.print("[bold red]Deepgram API key is missing![/bold red]")
        console.print("Please set the DEEPGRAM_API_KEY environment variable")
        return False
    
    try:
        # Initialize Deepgram client
        console.print("Initializing Deepgram client...")
        client = DeepgramClient(deepgram_key)
        
        # Test a simple API call (get available models)
        console.print("Testing API connection by fetching available models...")
        response = await client.manage.get_projects()
        
        # Check if the response is valid
        if response and hasattr(response, 'projects'):
            console.print("[bold green]Deepgram connection successful![/bold green]")
            console.print(f"Found {len(response.projects)} projects")
            return True
        else:
            console.print("[bold yellow]Received response from Deepgram, but it doesn't contain expected data.[/bold yellow]")
            console.print(f"Response: {response}")
            return False
            
    except Exception as e:
        console.print(f"[bold red]Error connecting to Deepgram: {str(e)}[/bold red]")
        return False

async def main():
    await test_deepgram_connection()

if __name__ == "__main__":
    asyncio.run(main())
