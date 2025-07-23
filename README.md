# toyaikit

Minimalistic implementation for LLM-based chat assistants with Tool Use (function calling) and MCP



This project started from a workshop "From RAG to Agents: Build Your Own AI Assistant"

https://github.com/alexeygrigorev/rag-agents-workshop

and then later from the LLM Zoomcamp course
where we covered AI Agents and MCP

https://github.com/DataTalksClub/llm-zoomcamp

## Publishing

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