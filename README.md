# GPT-4o-mini Engineer Chat
![Screenshot 2024-09-11 at 8 44 16 PM](https://github.com/user-attachments/assets/9adfeb13-0fda-4d0f-b32d-58bb7c307209)



This project is an AI-powered chat application using OpenAI's GPT-4o-mini model. It provides a multi-agent system with image support, code execution capabilities, and an autonomous mode for extended interactions.

## Features

- Chat with GPT-4o-mini model
- Image analysis support
- Code execution in an isolated environment
- Autonomous mode for continuous task execution
- Web search integration using Tavily API
- Rich console output for better readability

## Prerequisites

- Python 3.7 or higher
- OpenAI API key
- Tavily API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/gpt-4o-mini-engineer-chat.git
   cd gpt-4o-mini-engineer-chat
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your API keys:
   Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## Usage

Run the main script:
```
python main.py
```

Follow the on-screen instructions to interact with the AI assistant. You can:
- Chat normally by typing your messages
- Enter 'image' to include an image in your message
- Enter 'automode [number]' to start autonomous mode with a specific number of iterations
- Type 'reset' to clear the conversation history
- Type 'save chat' to save the conversation to a Markdown file
- Type 'exit' to end the conversation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
