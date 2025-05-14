# Agentic Data Scientist

An AI-powered data science team that can perform data analysis, feature engineering, and machine learning tasks using natural language commands.

![Agentic Data Scientist](https://img.shields.io/badge/Agentic-Data%20Scientist-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![AutoGen](https://img.shields.io/badge/AutoGen-Powered-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![LLM](https://img.shields.io/badge/LLM-Google%20Gemini-orange)

## Overview

Agentic Data Scientist is a project inspired by the [ai-data-science-team](https://github.com/business-science/ai-data-science-team) repository. It provides an agentic system that acts as a complete data science team, capable of performing exploratory data analysis (EDA), feature engineering, model building, and more.

The system uses Google's Gemini Pro as the LLM backend and provides a Streamlit-based user interface for easy interaction.

## Features

- **Data Connectors**: Import data from SQLite databases or local files (CSV, Excel)
- **Exploratory Data Analysis**: Generate summaries, visualizations, and insights about your data
- **Feature Engineering**: Transform, select, and create features for machine learning
- **Model Building**: Train and evaluate machine learning models
- **Interactive UI**: Streamlit-based user interface for seamless interaction
- **Agentic System**: Uses AutoGen framework for agentic capabilities

## Components

- **Data Explorer Agent**: Analyzes data, generates visualizations, and provides insights
- **Feature Engineer Agent**: Creates and selects features for machine learning models
- **Model Builder Agent**: Builds, tunes, and evaluates machine learning models
- **Code Generator**: Executes LLM-generated code securely

## Requirements

- Python 3.8 or higher
- Google Gemini API key
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/agentic-data-scientist.git
   cd agentic-data-scientist
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up the test database with sample data:
   ```bash
   python scripts/setup_test_db.py
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Enter your Google Gemini API key in the sidebar

4. Load data from a file or database

5. Use the different tabs to explore data, engineer features, and build models

## Workflow Example

1. **Load Data**: Upload a CSV/Excel file or connect to the SQLite database
2. **Explore Data**: Run EDA to understand the dataset's characteristics
3. **Engineer Features**: Create new features or transform existing ones
4. **Build Models**: Train and evaluate machine learning models

## Demo Dataset

The project includes a script to generate several sample datasets:

- California Housing
- Breast Cancer
- Iris
- Wine
- Employee Attrition (simulated)
- Credit Risk (simulated)
- Sales Time Series (simulated)
- Customer Churn (simulated)

Run `python scripts/setup_test_db.py` to generate these datasets.

## Project Structure

```
agentic-data-scientist/
├── app.py                   # Main Streamlit application
├── config.py                # Configuration settings
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
├── agentic_data_scientist/
│   ├── agents/              # Agent implementations
│   │   ├── base_agent.py    # Base agent class
│   │   ├── data_explorer.py # Data exploration agent
│   │   ├── feature_engineer.py # Feature engineering agent
│   │   ├── model_builder.py # Model building agent
│   ├── connectors/          # Data connectors
│   │   ├── sqlite_connector.py # SQLite database connector
│   │   ├── file_connector.py # File connector for CSV/Excel
│   ├── utils/               # Utility functions
│   │   ├── code_generator.py # For executing LLM-generated code
│   └── data/                # Sample data and database
└── scripts/                 # Helper scripts
    └── setup_test_db.py     # Script to set up test database
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project is inspired by:
- [ai-data-science-team](https://github.com/business-science/ai-data-science-team)
- [AutoGen](https://github.com/microsoft/autogen)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 