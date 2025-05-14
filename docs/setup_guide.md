# Development Environment Setup Guide

## Prerequisites

- Python 3.9 or higher
- Git
- A Google Gemini API key
- (Optional) Anaconda/Miniconda

## Step 1: Clone the Repository

```bash
git clone https://github.com/STHITAPRAJNAS/agentic-data-scientist.git
cd agentic-data-scientist
```

## Step 2: Set Up Virtual Environment

### Using venv (Recommended)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Using Conda (Alternative)

```bash
# Create conda environment
conda create -n agentic-ds python=3.9
conda activate agentic-ds
```

## Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

## Step 4: Environment Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your Google Gemini API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Step 5: Verify Installation

Run the example script:
```bash
python examples/basic_usage.py
```

## Step 6: Start the Streamlit App

```bash
python run.py app
```

## Common Issues and Solutions

### 1. NumPy Compatibility Issues

If you encounter NumPy-related errors:
```bash
pip install 'numpy<2.0.0' --force-reinstall
```

### 2. Streamlit Performance

For better Streamlit performance:
```bash
# On macOS:
xcode-select --install
pip install watchdog
```

### 3. Database Setup

To set up the test database:
```bash
python scripts/setup_test_db.py
```

## Development Workflow

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and test them:
```bash
python run.py app
```

3. Commit your changes:
```bash
git add .
git commit -m "Description of your changes"
```

4. Push to GitHub:
```bash
git push origin feature/your-feature-name
```

## Project Structure

```
agentic-data-scientist/
├── agentic_data_scientist/    # Main package
│   ├── agents/               # AI agents
│   ├── connectors/           # Data connectors
│   └── utils/               # Utility functions
├── docs/                    # Documentation
├── examples/                # Example scripts
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── tests/                  # Test files
├── .env.example           # Environment template
├── requirements.txt       # Dependencies
└── run.py                # Main runner
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 