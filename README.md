# toyaikit

ToyAIKit is a minimalistic Python library for building AI assistants powered by Large Language Models (LLMs). It provides a simple yet powerful framework for creating chatbots with advanced capabilities like:

The project builds upon concepts from multiple courses and workshops:
- ["From RAG to Agents: Build Your Own AI Assistant" Workshop](https://github.com/alexeygrigorev/rag-agents-workshop)
- [MLZoomcamp's LLM Course](https://github.com/DataTalksClub/llm-zoomcamp) covering AI Agents and MCP

It's great for learning about agents and agentic asisstants, but not suitable for production use. 

Main features:

- Support for OpenAI with both `reponses` and `chat.completions` APIs
- Support for OpenAI Agents SDK and Pydantic AI
- Tool integration for function calling
- Interactive IPython-based chat interface
- Easy to add new providers and runners

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

It displays the responses form the assistant and 
function calls

<img src="./images/weather.png" width="50%" />


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


## Examples

### OpenAI Chat Completions API

The default runner users the `responses` API. If you need to use 
the `chat.completions` API, do it with `OpenAIChatCompletionsRunner`:

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

### Extending it to other LLM providers 

Most of LLM providers follow the OpenAI API and can be used with the
OpenAI client. 

For example, this is how we can use Z.ai's GLM-4.5:

```python
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

# define the model to use
llm_client = OpenAIChatCompletionsClient(
    model='glm-4.5',
    client=zai_client
)

# Setup tools and run
agent_tools = ...

tools = Tools()
tools.add_tools(agent_tools)

chat_interface = IPythonChatInterface()

runner = OpenAIChatCompletionsRunner(
    tools=tools,
    developer_prompt="You are a coding agent that can modify Django projects.",
    chat_interface=chat_interface,
    llm_client=llm_client
)

runner.run()
```

## Wrappers

ToyAIKit can also help with running agents from OpenAI Agents SDK
and PydanticAI

### OpenAI Agents SDK


```python
from agents import Agent, Runner, SQLiteSession, function_tool

from toyaikit.tools import get_instance_methods
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIAgentsSDKRunner


# use get_instance_methods to find all the methods of an object
coding_agent_tools_list = []

for m in get_instance_methods(agent_tools):
    tool = function_tool(m)
    coding_agent_tools_list.append(tool)


# alternatively, define the list yourself:
coding_agent_tools_list = [
    function_tool(agent_tools.execute_bash_command),
    function_tool(agent_tools.read_file),
    function_tool(agent_tools.search_in_files),
    function_tool(agent_tools.see_file_tree),
    function_tool(agent_tools.write_file)
]

# create the Agent
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

# In Jypyter, run asynchronously
await runner.run()
```

### Pydantic AI with OpenAI

```python
from pydantic_ai import Agent

from toyaikit.tools import get_instance_methods
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import PydanticAIRunner

# get tools from your object with functions
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

You can easily switch to Claude:

```python
coding_agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    instructions="You are a coding agent that can modify Django projects.",
    tools=coding_agent_tools_list
)
```

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