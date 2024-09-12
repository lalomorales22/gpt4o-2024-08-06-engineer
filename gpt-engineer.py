import os
from dotenv import load_dotenv
import json
from tavily import TavilyClient
import base64
from PIL import Image
import io
import re
from openai import OpenAI
import difflib
import time
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
import asyncio
import aiohttp
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import datetime
import venv
import subprocess
import sys
import signal

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
client = OpenAI(api_key=openai_api_key)

# Initialize the Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")
tavily = TavilyClient(api_key=tavily_api_key)

console = Console()

# Add these constants at the top of the file
CONTINUATION_EXIT_PHRASE = "AUTOMODE_COMPLETE"
MAX_CONTINUATION_ITERATIONS = 25

# Model to use
MODEL = "gpt-4o-2024-08-06"

# Token tracking variables
model_tokens = {'input': 0, 'output': 0}

# You can set this to whatever you want, so you can see the progress towards a certain amount of tokens
# I set to 1M tokens, so you can see the progress towards 1M tokens
MAX_CONTEXT_TOKENS = 1000000  # 1M tokens for context window

# Set up the conversation memory
conversation_history = []

# automode flag
automode = False

# Global dictionary to store running processes
running_processes = {}

# Define the tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in an isolated environment",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to be read"
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to be written"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to be edited"
                    },
                    "changes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "operation": {
                                    "type": "string",
                                    "enum": ["insert", "delete", "replace"],
                                    "description": "The type of edit operation"
                                },
                                "line_number": {
                                    "type": "integer",
                                    "description": "The line number to apply the change"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content to insert, delete, or use as replacement"
                                }
                            },
                            "required": ["operation", "line_number", "content"]
                        }
                    }
                },
                "required": ["file_path", "changes"]
            }
        }
    }
]

# Add the display_token_usage function
def display_token_usage():
    total_tokens = model_tokens['input'] + model_tokens['output']
    percentage = (total_tokens / MAX_CONTEXT_TOKENS) * 100
    console.print(Panel(f"Token usage: {total_tokens:,} / {MAX_CONTEXT_TOKENS:,} ({percentage:.2f}%)", 
                        title="Token Usage", style="bold blue"))

# base prompt
base_system_prompt = """
You are an AI assistant powered by OpenAI's gpt-4o-mini model, specialized in software development with access to a variety of tools and the ability to instruct and direct a coding agent and a code execution one. Your capabilities include:

1. Creating and managing project structures
2. Writing, debugging, and improving code across multiple languages
3. Providing architectural insights and applying design patterns
4. Staying current with the latest technologies and best practices
5. Analyzing and manipulating files within the project directory
6. Performing web searches for up-to-date information
7. Executing code and analyzing its output within an isolated 'code_execution_env' virtual environment
8. Managing and stopping running processes started within the 'code_execution_env', DO NOT STOP ANYTHING UNLESS THE USER ASKS YOU TO or extremely necessary!!!
9. Reading, writing, and editing files directly on the local machine

You have access to the following tools:
1. search_web: Search the web for information
2. execute_code: Execute Python code in an isolated environment
3. read_file: Read the contents of a file
4. write_file: Write content to a file
5. edit_file: Edit the contents of a file

Use these tools responsibly and efficiently to assist with software development tasks.
"""

# Auto mode-specific system prompt
automode_system_prompt = """
You are currently in automode. Follow these guidelines:

1. Work autonomously towards completing the user's request.
2. Use your tools and capabilities to make progress on the task.
3. If you complete the task or reach a point where you cannot proceed, respond with the phrase "AUTOMODE_COMPLETE".
4. Provide clear and concise updates on your progress.
5. If you need more information or clarification, ask the user.

{iteration_info}
"""

def update_system_prompt(current_iteration=None, max_iterations=None):
    global base_system_prompt, automode_system_prompt
    chain_of_thought_prompt = """
    Answer the user's request using relevant tools (if they are available). Before calling a tool, do some analysis within <thinking></thinking> tags. First, think about which of the provided tools is the relevant tool to answer the user's request. Second, go through each of the required parameters of the relevant tool and determine if the user has directly provided or given enough information to infer a value. When deciding if the parameter can be inferred, carefully consider all the context to see if it supports a specific value. If all of the required parameters are present or can be reasonably inferred, close the thinking tag and proceed with the tool call. BUT, if one of the values for a required parameter is missing, DO NOT invoke the function (not even with fillers for the missing params) and instead, ask the user to provide the missing parameters. DO NOT ask for more information on optional parameters if it is not provided.

    Do not reflect on the quality of the returned search results in your response.
    """
    if automode:
        iteration_info = ""
        if current_iteration is not None and max_iterations is not None:
            iteration_info = f"You are currently on iteration {current_iteration} out of {max_iterations} in automode."
        return base_system_prompt + "\n\n" + automode_system_prompt.format(iteration_info=iteration_info) + "\n\n" + chain_of_thought_prompt
    else:
        return base_system_prompt + "\n\n" + chain_of_thought_prompt

# Implement the new file manipulation functions
def read_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return f"File successfully written: {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def edit_file(file_path, changes):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        for change in changes:
            line_number = change['line_number'] - 1  # Adjust for 0-based indexing
            if change['operation'] == 'insert':
                lines.insert(line_number, change['content'] + '\n')
            elif change['operation'] == 'delete':
                if 0 <= line_number < len(lines):
                    del lines[line_number]
            elif change['operation'] == 'replace':
                if 0 <= line_number < len(lines):
                    lines[line_number] = change['content'] + '\n'

        with open(file_path, 'w') as file:
            file.writelines(lines)

        return f"File successfully edited: {file_path}"
    except Exception as e:
        return f"Error editing file: {str(e)}"

async def execute_tool(tool_name, tool_input):
    if tool_name == "search_web":
        query = tool_input["query"]
        # Implement your web search logic here
        search_results = tavily.search(query)
        return json.dumps(search_results)
    elif tool_name == "execute_code":
        code = tool_input["code"]
        # Implement your code execution logic here
        # This is a placeholder and should be replaced with actual code execution
        return f"Code executed: {code}"
    elif tool_name == "read_file":
        file_path = tool_input["file_path"]
        return read_file(file_path)
    elif tool_name == "write_file":
        file_path = tool_input["file_path"]
        content = tool_input["content"]
        return write_file(file_path, content)
    elif tool_name == "edit_file":
        file_path = tool_input["file_path"]
        changes = tool_input["changes"]
        return edit_file(file_path, changes)
    else:
        return f"Unknown tool: {tool_name}"

async def chat_with_openai(user_input, image_path=None, current_iteration=None, max_iterations=None):
    global conversation_history, automode, model_tokens

    current_conversation = []

    if image_path:
        console.print(Panel(f"Processing image at path: {image_path}", title_align="left", title="Image Processing", expand=False, style="yellow"))
        image_base64 = encode_image_to_base64(image_path)

        if image_base64.startswith("Error"):
            console.print(Panel(f"Error encoding image: {image_base64}", title="Error", style="bold red"))
            return "I'm sorry, there was an error processing the image. Please try again.", False

        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": f"User input for image: {user_input}"
                }
            ]
        }
        current_conversation.append(image_message)
        console.print(Panel("Image message added to conversation history", title_align="left", title="Image Added", style="green"))
    else:
        current_conversation.append({"role": "user", "content": user_input})

    messages = conversation_history + current_conversation

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": update_system_prompt(current_iteration, max_iterations)}] + messages,
            max_tokens=4000,
            tools=tools
        )
        # Update token usage
        model_tokens['input'] += response.usage.prompt_tokens
        model_tokens['output'] += response.usage.completion_tokens
        
        # Display token usage after each main model call
        display_token_usage()
    except Exception as e:
        console.print(Panel(f"API Error: {str(e)}", title="API Error", style="bold red"))
        return "I'm sorry, there was an error communicating with the AI. Please try again.", False

    assistant_response = response.choices[0].message.content
    exit_continuation = False
    tool_calls = response.choices[0].message.tool_calls or []

    if assistant_response:
        console.print(Panel(Markdown(assistant_response), title="AI's Response", title_align="left", border_style="blue", expand=False))
    else:
        console.print(Panel("No text response from AI", title="AI's Response", title_align="left", border_style="blue", expand=False))

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        tool_input = json.loads(tool_call.function.arguments)
        tool_id = tool_call.id

        console.print(Panel(f"Tool Used: {tool_name}", style="green"))
        console.print(Panel(f"Tool Input: {json.dumps(tool_input, indent=2)}", style="green"))

        try:
            result = await execute_tool(tool_name, tool_input)
            console.print(Panel(result, title_align="left", title="Tool Result", style="green"))
        except Exception as e:
            result = f"Error executing tool: {str(e)}"
            console.print(Panel(result, title="Tool Execution Error", style="bold red"))

        current_conversation.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(tool_input)}
                }
            ]
        })

        current_conversation.append({
            "role": "tool",
            "content": result,
            "tool_call_id": tool_id
        })

    if assistant_response:
        current_conversation.append({"role": "assistant", "content": assistant_response})

    conversation_history = messages + [{"role": "assistant", "content": assistant_response or ""}]

    return assistant_response or "", exit_continuation

def reset_conversation():
    global conversation_history
    conversation_history = []
    console.print(Panel("Conversation history has been reset.", title="Reset", style="bold green"))

def save_chat():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.md"
    
    with open(filename, "w") as f:
        for message in conversation_history:
            role = message["role"].capitalize()
            content = message["content"]
            f.write(f"## {role}\n\n{content}\n\n")
    
    return filename

def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as image:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        return f"Error: {str(e)}"

async def main():
    global automode, conversation_history
    console.print(Panel("Welcome to the gpt-4o-2024-08-06 Engineer Chat with Multi-Agent and Image Support!", title="Welcome", style="bold green"))
    console.print("Type 'exit' to end the conversation.")
    console.print("Type 'image' to include an image in your message.")
    console.print("Type 'automode [number]' to enter Autonomous mode with a specific number of iterations.")
    console.print("Type 'reset' to clear the conversation history.")
    console.print("Type 'save chat' to save the conversation to a Markdown file.")
    console.print("While in automode, press Ctrl+C at any time to exit the automode to return to regular chat.")

    while True:
        user_input = console.input("[bold cyan]You:[/bold cyan] ")

        if user_input.lower() == 'exit':
            console.print(Panel("Thank you for chatting. Goodbye!", title_align="left", title="Goodbye", style="bold green"))
            break

        if user_input.lower() == 'reset':
            reset_conversation()
            continue

        if user_input.lower() == 'save chat':
            filename = save_chat()
            console.print(Panel(f"Chat saved to {filename}", title="Chat Saved", style="bold green"))
            continue

        if user_input.lower() == 'image':
            image_path = console.input("[bold cyan]Drag and drop your image here, then press enter:[/bold cyan] ").strip().replace("'", "")

            if os.path.isfile(image_path):
                user_input = console.input("[bold cyan]You (prompt for image):[/bold cyan] ")
                response, _ = await chat_with_openai(user_input, image_path)
            else:
                console.print(Panel("Invalid image path. Please try again.", title="Error", style="bold red"))
                continue
        elif user_input.lower().startswith('automode'):
            try:
                parts = user_input.split()
                if len(parts) > 1 and parts[1].isdigit():
                    max_iterations = int(parts[1])
                else:
                    max_iterations = MAX_CONTINUATION_ITERATIONS

                automode = True
                console.print(Panel(f"Entering automode with {max_iterations} iterations. Please provide the goal of the automode.", title_align="left", title="Automode", style="bold yellow"))
                console.print(Panel("Press Ctrl+C at any time to exit the automode loop.", style="bold yellow"))
                user_input = console.input("[bold cyan]You:[/bold cyan] ")

                iteration_count = 0
                try:
                    while automode and iteration_count < max_iterations:
                        response, exit_continuation = await chat_with_openai(user_input, current_iteration=iteration_count+1, max_iterations=max_iterations)

                        if exit_continuation or CONTINUATION_EXIT_PHRASE in response:
                            console.print(Panel("Automode completed.", title_align="left", title="Automode", style="green"))
                            automode = False
                        else:
                            console.print(Panel(f"Continuation iteration {iteration_count + 1} completed. Press Ctrl+C to exit automode. ", title_align="left", title="Automode", style="yellow"))
                            user_input = "Continue with the next step. Or STOP by saying 'AUTOMODE_COMPLETE' if you think you've achieved the results established in the original request."
                        iteration_count += 1

                        if iteration_count >= max_iterations:
                            console.print(Panel("Max iterations reached. Exiting automode.", title_align="left", title="Automode", style="bold red"))
                            automode = False
                except KeyboardInterrupt:
                    console.print(Panel("\nAutomode interrupted by user. Exiting automode.", title_align="left", title="Automode", style="bold red"))
                    automode = False
                    if conversation_history and conversation_history[-1]["role"] == "user":
                        conversation_history.append({"role": "assistant", "content": "Automode interrupted. How can I assist you further?"})
            except KeyboardInterrupt:
                console.print(Panel("\nAutomode interrupted by user. Exiting automode.", title_align="left", title="Automode", style="bold red"))
                automode = False
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history.append({"role": "assistant", "content": "Automode interrupted. How can I assist you further?"})

            console.print(Panel("Exited automode. Returning to regular chat.", style="green"))

        else:
            response, _ = await chat_with_openai(user_input)
            if not response:
                console.print(Panel("No response from AI. The AI might have used a tool without providing a text response.", title="AI's Action", style="yellow"))

if __name__ == "__main__":
    asyncio.run(main())