# Agentic Data Scientist

A modular, agentic data science platform that leverages LLMs (Google Gemini, etc.) to automate and accelerate data exploration, feature engineering, and model building. Includes a secure code execution environment and a modern Streamlit UI.

## Features
- Dual agent architecture implementations:
  - LangChain-based agents (default)
  - AutoGen-based conversational agents
- Modular agent architecture (Data Explorer, Feature Engineer, Model Builder)
- Data connectors for SQLite, Databricks, and file uploads
- Secure, sandboxed code execution for LLM-generated code
- Interactive Streamlit web interface
- Configuration via `.env` file (no secrets in code/UI)
- Example scripts and notebooks

## Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/STHITAPRAJNAS/agentic-data-scientist.git
cd agentic-data-scientist
```

### 2. Set up your environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
Copy the example file and edit it:
```bash
cp .env.example .env
```
Edit `.env` and set your Google Gemini API key:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. (Optional) Set up the test database
```bash
python scripts/setup_test_db.py
```

### 5. Run the Streamlit app
```bash
python run.py app
```

- The app will read your Gemini API key and all config from `.env`.
- **Do not enter your API key in the UI.**

### 6. Try the AutoGen implementation
In the Streamlit UI:
1. Select "AutoGen" in the "Select Agent Framework" dropdown in the sidebar
2. Click "Initialize Agents" to load the AutoGen-based implementation
3. Use the same interface to explore data, engineer features, and build models

## Project Structure
```
agentic-data-scientist/
├── agentic_data_scientist/    # Main package
│   ├── agents/               # LangChain-based AI agents
│   ├── autogen_agents/       # AutoGen-based AI agents
│   ├── connectors/           # Data connectors
│   └── utils/                # Utility functions
├── docs/                     # Documentation
├── examples/                 # Example scripts
├── notebooks/                # Jupyter notebooks
├── scripts/                  # Utility scripts
├── tests/                    # Test files
├── .env.example              # Environment template
├── requirements.txt          # Dependencies
└── run.py                    # Main runner
```

## Example Scripts
- `examples/basic_usage.py`: Basic usage of the LangChain-based implementation
- `examples/autogen_example.py`: Example of using the AutoGen-based implementation

## Documentation
- See `docs/setup_guide.md` for full setup and troubleshooting instructions.
- See `notebooks/` for example workflows.

## Agent Frameworks
The application supports two agent frameworks:

### LangChain-based Agents (Default)
- Uses LangChain's LLMChain with Google Gemini for agent implementation
- More traditional agent structure with direct execution

### AutoGen-based Agents
- Uses Microsoft's AutoGen framework for multi-agent conversations
- Provides a more interactive and conversational approach to problem-solving
- Agents can collaborate and take turns generating solutions

You can switch between the two implementations in the Streamlit UI's sidebar.

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a Pull Request

## License
MIT License. See `LICENSE` for details. 