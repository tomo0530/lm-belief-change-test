# 🔬 Open Deep Research
We use the [Open Deep Research](https://github.com/langchain-ai/open_deep_research/tree/main) agent developed by LangChain to run the Deep Research for the study stage.

## Quickstart
1. Activate a virtual environment:
```bash
cd open_deep_research
uv venv
source .venv/bin/activate  
```

2. Install dependencies:
```bash
uv sync
# or
uv pip install -r pyproject.toml
```

3. Set up your `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):
```bash
cp .env.example .env
```

4. Launch agent with the LangGraph server locally:
```bash
# Install dependencies and start the LangGraph server
./start_server.sh
```

5. Run study:
```bash
python -m client.run --mode conservative --model_name gpt-5
```

