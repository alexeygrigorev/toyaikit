# ToyAIKit

ToyAIKit is a minimalistic Python library for building AI assistants powered by Large Language Models (LLMs). It provides a simple yet powerful framework for creating agentic conversational systems with advanced capabilities including function calling, tool integration, and multi-provider support.

## Purpose

This library is designed for **educational purposes** and learning about AI agents. It builds upon concepts from multiple courses and workshops:

- ["AI Bootcamp: From RAG to Agents"](https://maven.com/alexey-grigorev/from-rag-to-agents) course
- ["Agents and MCP"** workshop](https://www.youtube.com/watch?v=W2EDdZplLcU) - Deep dive into function calling, MCP servers, and agentic flows
- ["Create Your Own Coding Agent" workshop](https://www.youtube.com/watch?v=Sue_mn0JCsY) - Building Django coding agents from scratch
- [LLM Zoomcamp Course](https://github.com/DataTalksClub/llm-zoomcamp) covering AI Agents and MCP

> **⚠️ Important**: ToyAIKit is great for learning about agents and agentic assistants, but **not suitable for production use**. For production applications, consider using frameworks like OpenAI Agents SDK, PydanticAI.

## Key Features

- Multi-Provider Support: OpenAI (both `responses` and `chat.completions` APIs), Anthropic Claude, Z.ai, and other OpenAI-compatible providers
- Framework Integration: Wrappers for OpenAI Agents SDK and PydanticAI
- Function Calling: Easy tool integration with automatic schema generation
- MCP Support: Model Context Protocol client and server utilities
- Interactive Chat: IPython-based chat interface for Jupyter notebooks
- Agentic Flow: Support for multi-step reasoning and tool orchestration
- Educational Focus: Clear, readable code designed for learning

## Quick Start

```bash
pip install toyaikit
```

### Basic Usage with OpenAI

```python
from openai import OpenAI

from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner

# Create tools
tools = Tools()

# Add a simple function as a tool
def get_weather(city: str):
    """Get weather information for a city."""
    return f"Weather in {city}: Sunny, 25°C"

tools.add_tool(get_weather)

# Create chat interface and client
chat_interface = IPythonChatInterface()
openai_client = OpenAIClient(
    model="gpt-4o-mini",
    client=OpenAI()
)

# Create and run chat assistant
runner = OpenAIResponsesRunner(
    tools=tools,
    developer_prompt="You are a helpful weather assistant.",
    chat_interface=chat_interface,
    llm_client=openai_client
)

runner.run()
```

The interface displays responses from the assistant and function calls:

<img src="./images/weather.png" width="50%" />

## Agentic Flow

ToyAIKit implements the agentic conversational flow that distinguishes agents from simple Q&A bots:

1. User Input: User types their question
2. LLM Decision: The system sends the question to the LLM provider  
3. Tool Invocation: LLM decides whether to invoke tools or answer directly
4. Function Execution: If tools are needed, the system executes them and sends results back to LLM
5. Final Response: LLM analyzes results and provides the final output

This cycle can repeat multiple times within a single conversation turn, enabling complex multi-step reasoning.


### Tools System

The tools system allows you to easily integrate Python functions with LLM function calling:

```python
from toyaikit.tools import Tools

tools = Tools()

# Add individual functions
def calculate_area(length: float, width: float):
    """Calculate the area of a rectangle."""
    return length * width

tools.add_tool(calculate_area)

# Add all methods from a class instance
class MathTools:
    def add(self, a: float, b: float):
        """Add two numbers."""
        return a + b
    
    def multiply(self, a: float, b: float):
        """Multiply two numbers."""
        return a * b

math_tools = MathTools()
tools.add_tools(math_tools)
```

### Chat Interface

The IPython-based chat interface provides an interactive way to chat with your AI assistant:

```python
from toyaikit.chat import IPythonChatInterface

chat_interface = IPythonChatInterface()

# Get user input
user_input = chat_interface.input()

# Display message
chat_interface.display("Hello!")

# Display AI response
chat_interface.display_response("AI response")

# Display function call
chat_interface.display_function_call("function_name", '{"arg1": "value1"}', "result")
```

## Complete Examples from Workshops

### 1. FAQ Search Agent (from "Agents and MCP" Workshop)

Build a course teaching assistant that can search through FAQ documents and add new entries:

```python
import requests
from minsearch import AppendableIndex
from typing import List, Dict, Any

# Load FAQ data
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

# Prepare documents
documents = []
for course in documents_raw:
    course_name = course['course']
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

# Create search index
index = AppendableIndex(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)
index.fit(documents)

# Define search tools
class SearchTools:
    def __init__(self, index):
        self.index = index

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the FAQ database for entries matching the given query.
        
        Args:
            query (str): Search query text to look up in the course FAQ.
        
        Returns:
            List[Dict[str, Any]]: A list of search result entries.
        """
        boost = {'question': 3.0, 'section': 0.5}
        results = self.index.search(
            query=query,
            filter_dict={'course': 'data-engineering-zoomcamp'},
            boost_dict=boost,
            num_results=5,
        )
        return results

    def add_entry(self, question: str, answer: str) -> None:
        """
        Add a new entry to the FAQ database.
        
        Args:
            question (str): The question to be added to the FAQ database.
            answer (str): The corresponding answer to the question.
        """
        doc = {
            'question': question,
            'text': answer,
            'section': 'user added',
            'course': 'data-engineering-zoomcamp'
        }
        self.index.append(doc)

# Create and run FAQ agent
search_tools = SearchTools(index)

tools = Tools()
tools.add_tools(search_tools)

developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.

If you want to look up the answer, explain why before making the call. Use as many 
keywords from the user question as possible when making first requests.

Make multiple searches. Try to expand your search by using new keywords based on the results you
get from the search.

At the end, make a clarifying question based on what you presented and ask if there are 
other areas that the user wants to explore.
"""

runner = OpenAIResponsesRunner(
    tools=tools,
    developer_prompt=developer_prompt,
    chat_interface=IPythonChatInterface(),
    llm_client=OpenAIClient()
)

runner.run()
```

### 2. Django Coding Agent (from "Create Your Own Coding Agent" Workshop)

Create a coding agent that can build Django applications from templates:

```python
import os
import shutil
import subprocess
from pathlib import Path

class AgentTools:
    def __init__(self, project_path: Path):
        self.project_path = project_path

    def read_file(self, file_path: str) -> str:
        """Read the contents of a file."""
        try:
            full_path = self.project_path / file_path
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file."""
        try:
            full_path = self.project_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"File {file_path} written successfully"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def execute_bash_command(self, command: str) -> str:
        """Execute a bash command in the project directory."""
        if "runserver" in command:
            return "runserver command blocked in agent mode"
        try:
            result = subprocess.run(
                command.split(), 
                cwd=self.project_path, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            return f"Exit code: {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        except Exception as e:
            return f"Error executing command: {str(e)}"

# Copy Django template
def start_project():
    project_name = input("Enter the new Django project name: ").strip()
    if not project_name:
        print("Project name cannot be empty.")
        return
    
    if os.path.exists(project_name):
        print(f"Directory '{project_name}' already exists.")
        return
    
    shutil.copytree('django_template', project_name)
    print(f"Django template copied to '{project_name}' directory.")
    return project_name

# Setup coding agent
project_name = start_project()
if project_name:
    project_path = Path(project_name)
    agent_tools = AgentTools(project_path)
    
    tools = Tools()
    tools.add_tools(agent_tools)
    
    developer_prompt = """
    You are a coding agent. Your task is to modify the provided Django project template
    according to user instructions. You don't tell the user what to do; you do it yourself 
    using the available tools.
    
    Always ensure changes are consistent with Django best practices and the project's structure.
    Use TailwindCSS for styling and make the results look beautiful.
    """
    
    runner = OpenAIResponsesRunner(
        tools=tools,
        developer_prompt=developer_prompt,
        chat_interface=IPythonChatInterface(),
        llm_client=OpenAIClient()
    )
    
    runner.run()
```

## Model Context Protocol (MCP) Integration

ToyAIKit includes utilities for working with MCP servers and clients:

### MCP Client Usage

```python
from toyaikit.mcp import MCPClient, SubprocessMCPTransport

# Connect to an MCP server
client = MCPClient(
    transport=SubprocessMCPTransport(
        server_command=["uv", "run", "python", "main.py"],
        workdir="faq-mcp"
    )
)

# Initialize connection
client.full_initialize()

# Use MCP tools with ToyAIKit
from toyaikit.mcp import MCPTools

mcp_tools = MCPTools(client)

runner = OpenAIResponsesRunner(
    tools=mcp_tools,
    developer_prompt="You are an assistant with access to MCP tools.",
    chat_interface=IPythonChatInterface(),
    llm_client=OpenAIClient()
)

runner.run()
```

### MCP Server Creation (with FastMCP)

```python
from fastmcp import FastMCP
from toyaikit.tools import wrap_instance_methods

# Create MCP server
mcp = FastMCP("FAQ Server")

# Add tools to server
search_tools = SearchTools(index)
wrap_instance_methods(mcp.tool, search_tools)

# Run server
if __name__ == "__main__":
    mcp.run()  # STDIO transport
    # or
    mcp.run(transport="sse")  # HTTP SSE transport
```


## Framework Integration Examples

### OpenAI Chat Completions API

The default runner uses the `responses` API. If you need to use the `chat.completions` API, use `OpenAIChatCompletionsRunner`:

```python
from openai import OpenAI

from toyaikit.tools import Tools
from toyaikit.llm import OpenAIChatCompletionsClient
from toyaikit.chat.runners import OpenAIChatCompletionsRunner
from toyaikit.chat import IPythonChatInterface

# Setup tools and client
agent_tools = ... # class with some functions to be called

tools = Tools()
tools.add_tools(agent_tools)

chat_interface = IPythonChatInterface()

llm_client = OpenAIChatCompletionsClient(
    model="gpt-4o-mini",
    client=OpenAI()
)

# Create and run the chat completions runner
runner = OpenAIChatCompletionsRunner(
    tools=tools,
    developer_prompt="You are a coding agent that can modify Django projects.",
    chat_interface=chat_interface,
    llm_client=llm_client
)
runner.run()
```

### Multiple LLM Providers

Most LLM providers follow the OpenAI API and can be used with the OpenAI client.

**Z.ai's GLM-4.5:**

```python
import os
from openai import OpenAI

from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIChatCompletionsRunner
from toyaikit.llm import OpenAIChatCompletionsClient

# Setup z.ai client
zai_client = OpenAI(
    api_key=os.getenv('ZAI_API_KEY'),
    base_url='https://api.z.ai/api/paas/v4/'
)

# Define the model to use
llm_client = OpenAIChatCompletionsClient(
    model='glm-4.5',
    client=zai_client
)

# Setup tools and run
agent_tools = ...

tools = Tools()
tools.add_tools(agent_tools)

runner = OpenAIChatCompletionsRunner(
    tools=tools,
    developer_prompt="You are a coding agent that can modify Django projects.",
    chat_interface=IPythonChatInterface(),
    llm_client=llm_client
)

runner.run()
```

### Anthropic (Claude)

**Using Anthropic's Messages API:**

```python
import os

from toyaikit import AnthropicClient, AnthropicMessagesRunner
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface

# Setup Anthropic client
# Uses ANTHROPIC_API_KEY from environment automatically
llm_client = AnthropicClient(
    model="claude-sonnet-4-5-20250514",
)

# Or with custom base URL (e.g., for z.ai compatibility)
llm_client = AnthropicClient(
    api_key=os.environ.get("ZAI_AUTH_TOKEN"),
    base_url="https://api.z.ai/api/anthropic",
    model="claude-sonnet-4-5-20250929",
)

# Setup tools and run
agent_tools = ... # class with some functions to be called

tools = Tools()
tools.add_tools(agent_tools)

runner = AnthropicMessagesRunner(
    tools=tools,
    developer_prompt="You are a helpful assistant.",
    chat_interface=IPythonChatInterface(),
    llm_client=llm_client
)

runner.run()
```


### OpenAI Agents SDK Integration

```python
from agents import Agent, function_tool

from toyaikit.tools import wrap_instance_methods
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIAgentsSDKRunner

# Wrap tools with function_tool decorator
agent_tools = ... # your tools class instance
coding_agent_tools_list = wrap_instance_methods(function_tool, agent_tools)

# Create the Agent
coding_agent = Agent(
    name="CodingAgent",
    instructions="You are a coding agent that can modify Django projects.",
    tools=coding_agent_tools_list,
    model='gpt-4o-mini'
)

# Setup and run with ToyAIKit
chat_interface = IPythonChatInterface()
runner = OpenAIAgentsSDKRunner(
    chat_interface=chat_interface,
    agent=coding_agent
)

# In Jupyter, run asynchronously
await runner.run()
```

### PydanticAI Integration

**With OpenAI:**

```python
from pydantic_ai import Agent

from toyaikit.tools import get_instance_methods
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import PydanticAIRunner

# Get tools from your object with functions
coding_agent_tools_list = get_instance_methods(agent_tools)

# Create Pydantic AI agent with OpenAI
coding_agent = Agent(
    'openai:gpt-4o-mini',
    instructions="You are a coding agent that can modify Django projects.",
    tools=coding_agent_tools_list
)

# Setup and run with ToyAIKit
chat_interface = IPythonChatInterface()
runner = PydanticAIRunner(
    chat_interface=chat_interface,
    agent=coding_agent
)

# Run asynchronously
await runner.run()
```

**Switching to Claude:**

```python
coding_agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    instructions="You are a coding agent that can modify Django projects.",
    tools=coding_agent_tools_list
)
```

### PydanticAI with MCP

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from toyaikit.chat.interface import StdOutputInterface
from toyaikit.chat.runners import PydanticAIRunner

# Connect to MCP server
mcp_client = MCPServerStdio(
    command="uv",
    args=["run", "python", "main.py"],
    cwd="faq-mcp"
)

# Create agent with MCP tools
agent = Agent(
    name="faq_agent",
    instructions="You're a course teaching assistant.",
    toolsets=[mcp_client],
    model='gpt-4o-mini'
)

# Run agent
runner = PydanticAIRunner(
    chat_interface=StdOutputInterface(),
    agent=agent
)

import asyncio
asyncio.run(runner.run())
```

## Structured Output

The `OpenAIResponsesRunner` and `OpenAIChatCompletionsRunner` support structured output using Pydantic models. This allows you to get responses in a specific format:

### Using OpenAIResponsesRunner

```python
from pydantic import BaseModel
from toyaikit.llm import OpenAIClient
from toyaikit.chat.runners import OpenAIResponsesRunner

# Define your response structure
class TestResponse(BaseModel):
    hello: bool
    world: str

# Create the runner
runner = OpenAIResponsesRunner(llm_client=OpenAIClient())

# Use structured output
result = runner.loop(prompt='how are you', output_format=TestResponse)
print(result.last_message)
# Output: TestResponse(hello=True, world="I'm doing well, thank you! How can I assist you today?")

# Or use without structured output for free-form text
result = runner.loop(prompt='how are you')
print(result.last_message)
# Output: "I'm just a program, but I'm here and ready to help you! How can I assist you today?"
```

### Using OpenAIChatCompletionsRunner

```python
from pydantic import BaseModel
from toyaikit.llm import OpenAIChatCompletionsClient
from toyaikit.chat.runners import OpenAIChatCompletionsRunner

# Define your response structure
class TestResponse(BaseModel):
    hello: bool
    world: str

# Create the runner
runner = OpenAIChatCompletionsRunner(llm_client=OpenAIChatCompletionsClient())

# Use structured output
result = runner.loop(prompt='how are you', output_format=TestResponse)
print(result.last_message)
# Output: TestResponse(hello=True, world="I'm doing well, thank you! How can I assist you today?")

# Or use without structured output for free-form text
result = runner.loop(prompt='how are you')
print(result.last_message)
# Output: "I'm just a program, but I'm here and ready to help you! How can I assist you today?"
```


## Use Cases & Best Practices

### When to Use ToyAIKit

✅ **Good for:**
- Learning about AI agents and agentic patterns
- Prototyping agent-based applications
- Educational projects and workshops
- Understanding function calling mechanics
- Experimenting with different LLM providers
- Building proof-of-concept agents

❌ **Not suitable for:**
- Production applications
- High-scale systems
- Applications requiring advanced agent orchestration
- Enterprise-grade reliability


## Development

### Running Tests

```bash
make test
```

### Publishing

Build the package:
```bash
uv run hatch build
```

Publish to test PyPI:
```bash
uv run hatch publish --repo test
```

Publish to PyPI:
```bash
uv run hatch publish
```

Clean up:
```bash
rm -r dist/
```

Note: For Hatch publishing, you'll need to configure your PyPI credentials in `~/.pypirc` or use environment variables.