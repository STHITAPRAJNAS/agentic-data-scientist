#!/usr/bin/env python
"""Main Streamlit application for the Agentic Data Scientist."""
import os
import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import nest_asyncio
import base64
import io
import time
import json
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Union, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TEMP_DIR = os.getenv("TEMP_DIR", "temp_code")
os.makedirs(TEMP_DIR, exist_ok=True)

# Import our modules
from config import config
from agentic_data_scientist.connectors.sqlite_connector import SQLiteConnector
from agentic_data_scientist.connectors.file_connector import FileConnector
from agentic_data_scientist.agents.data_explorer import DataExplorerAgent, DataExplorerAgentConfig
from agentic_data_scientist.agents.feature_engineer import FeatureEngineerAgent, FeatureEngineerAgentConfig
from agentic_data_scientist.agents.model_builder import ModelBuilderAgent, ModelBuilderAgentConfig

# Import the AutoGen-based agents
from agentic_data_scientist.autogen_agents.agent_wrappers import (
    AutoGenDataExplorerAgentConfig,
    AutoGenFeatureEngineerAgentConfig,
    AutoGenModelBuilderAgentConfig,
    AutoGenDataExplorerAgent,
    AutoGenFeatureEngineerAgent,
    AutoGenModelBuilderAgent
)

# Set page configuration
st.set_page_config(
    page_title="Agentic Data Scientist",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up session state
if "db_connector" not in st.session_state:
    st.session_state.db_connector = None
if "file_connector" not in st.session_state:
    st.session_state.file_connector = FileConnector("data")
if "current_data" not in st.session_state:
    st.session_state.current_data = None
if "data_source" not in st.session_state:
    st.session_state.data_source = None
if "data_source_type" not in st.session_state:
    st.session_state.data_source_type = None
if "explorer_agent" not in st.session_state:
    st.session_state.explorer_agent = None
if "feature_engineer_agent" not in st.session_state:
    st.session_state.feature_engineer_agent = None
if "model_builder_agent" not in st.session_state:
    st.session_state.model_builder_agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "engineered_data" not in st.session_state:
    st.session_state.engineered_data = None
if "model_results" not in st.session_state:
    st.session_state.model_results = {}

# Add new session state variables for data persistence
if "loaded_data_info" not in st.session_state:
    st.session_state.loaded_data_info = None
if "is_data_loaded" not in st.session_state:
    st.session_state.is_data_loaded = False

# Add a session state variable for agent type
if "agent_type" not in st.session_state:
    st.session_state.agent_type = "langchain"  # Default to LangChain agents

# Custom CSS
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
        border-bottom: 2px solid #4c84ff;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e6f0ff;
        border-left: 5px solid #4c84ff;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
        border-left: 5px solid #9ca3af;
    }
    .chat-message .message-content {
        display: flex;
        margin-top: 0.5rem;
    }
    .info-box {
        background-color: #e6f0ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_agents():
    """Initialize the agents with the API key from environment variables."""
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    
    if not api_key:
        st.error("API key is not set in environment variables. Please set GOOGLE_API_KEY in your .env file.")
        return False
    
    try:
        # Validate the API key
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Test the API key with a simple request
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Test")
        
        # Initialize agents based on the selected agent type
        if st.session_state.agent_type == "autogen":
            # Initialize the AutoGen-based data explorer agent
            explorer_config = AutoGenDataExplorerAgentConfig(
                name="DataExplorerAgent",
                description="Agent for exploring and analyzing data",
                api_key=api_key,
                model=model_name,
                code_execution=True
            )
            st.session_state.explorer_agent = AutoGenDataExplorerAgent(explorer_config)
            
            # Initialize the AutoGen-based feature engineer agent
            engineer_config = AutoGenFeatureEngineerAgentConfig(
                name="FeatureEngineerAgent",
                description="Agent for engineering features",
                api_key=api_key,
                model=model_name,
                code_execution=True
            )
            st.session_state.feature_engineer_agent = AutoGenFeatureEngineerAgent(engineer_config)
            
            # Initialize the AutoGen-based model builder agent
            model_config = AutoGenModelBuilderAgentConfig(
                name="ModelBuilderAgent",
                description="Agent for building models",
                api_key=api_key,
                model=model_name,
                code_execution=True
            )
            st.session_state.model_builder_agent = AutoGenModelBuilderAgent(model_config)
        else:
            # Initialize the original LangChain-based data explorer agent
            explorer_config = DataExplorerAgentConfig(
                name="DataExplorerAgent",
                description="Agent for exploring and analyzing data",
                api_key=api_key,
                model=model_name,
                code_execution=True
            )
            st.session_state.explorer_agent = DataExplorerAgent(explorer_config)
            
            # Initialize the original LangChain-based feature engineer agent
            engineer_config = FeatureEngineerAgentConfig(
                name="FeatureEngineerAgent",
                description="Agent for engineering features",
                api_key=api_key,
                model=model_name,
                code_execution=True
            )
            st.session_state.feature_engineer_agent = FeatureEngineerAgent(engineer_config)
            
            # Initialize the original LangChain-based model builder agent
            model_config = ModelBuilderAgentConfig(
                name="ModelBuilderAgent",
                description="Agent for building models",
                api_key=api_key,
                model=model_name,
                code_execution=True
            )
            st.session_state.model_builder_agent = ModelBuilderAgent(model_config)
        
        return True
    except Exception as e:
        st.error(f"Error initializing agents: {str(e)}")
        return False

def sanitize_dataframe(df):
    """Helper function to sanitize dataframe for display and PyArrow conversion."""
    if df is None:
        return None
    
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Force conversion of all float dtypes to basic numpy float64
        for col in df.select_dtypes(include=['float', 'Float64']).columns:
            df[col] = df[col].astype('float64')
        
        # Force conversion of all integer dtypes to basic numpy int64
        for col in df.select_dtypes(include=['integer', 'Int64']).columns:
            df[col] = df[col].astype('int64')
            
        # Force conversion of all object dtypes to string
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str)
            
        # Replace any infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Downcast to more efficient types
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
                
        # Drop any columns that still cause issues
        problematic_cols = []
        for col in df.columns:
            try:
                # Test if this column can be converted to arrow
                pd.DataFrame({col: df[col]}).to_numpy()
            except Exception:
                problematic_cols.append(col)
                
        if problematic_cols:
            print(f"Warning: Dropping problematic columns: {problematic_cols}")
            df = df.drop(columns=problematic_cols)
            
    except Exception as e:
        print(f"Warning: Could not convert data types: {e}")
        # Last resort: convert everything to strings
        for col in df.columns:
            try:
                df[col] = df[col].astype(str)
            except:
                df[col] = ["ERROR"] * len(df)
    
    return df

def persist_dataframes():
    """Store copies of the current dataframe for stability across reruns."""
    # Only persist if data is actually loaded
    if not st.session_state.is_data_loaded:
        return
        
    df = st.session_state.current_data
    if df is None or (hasattr(df, 'empty') and df.empty):
        return
    
    # First clear any existing copies to avoid memory leaks or stale data
    feature_engineering_df = None
    modeling_df = None
    exploration_df = None
    dim_reduction_df = None
        
    # Make a deep copy of the dataframe for each major function
    st.session_state.feature_engineering_df = df.copy() 
    st.session_state.modeling_df = df.copy()
    st.session_state.exploration_df = df.copy()
    
    # Log for debugging
    st.session_state.data_state_debug = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "shape": df.shape,
        "columns": list(df.columns)
    }

def display_dataframe(df: pd.DataFrame, max_rows: int = 10) -> None:
    """Display a dataframe with a download button."""
    # Convert to standard types to avoid ArrowInvalid errors
    df = sanitize_dataframe(df)
    if df is None:
        st.error("Unable to display dataframe - invalid data")
        return
        
    st.dataframe(df.head(max_rows))
    
    # Add download button
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv" class="btn">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
    
    # Show data info
    col1, col2 = st.columns(2)
    with col1:
        st.write("Data Shape:", df.shape)
    with col2:
        st.write("Data Types:")
        st.write(df.dtypes)

def display_chat_message(role: str, content: str, key: Optional[str] = None) -> None:
    """Display a chat message."""
    with st.chat_message(role, avatar="🤔" if role == "user" else "🧪"):
        st.markdown(content)
    
    # Add to chat history
    if key is None:
        st.session_state.chat_history.append({"role": role, "content": content})

def handle_file_upload() -> None:
    """Handle file upload."""
    st.subheader("Upload Data")
    
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Save the uploaded file
            file_connector = st.session_state.file_connector
            file_path = file_connector.save_upload(uploaded_file, uploaded_file.name)
            
            # Read the data based on file type
            file_type = file_connector.detect_file_type(uploaded_file.name)
            
            if file_type == "csv":
                df = pd.read_csv(file_path)
                st.session_state.current_data = df
                st.session_state.data_source = f"Uploaded file: {uploaded_file.name}"
                st.session_state.data_source_type = "Uploaded file"
                st.session_state.is_data_loaded = True
                # Create persistent copies for tab functions
                persist_dataframes()
                st.success(f"Successfully loaded data from {uploaded_file.name}")
                
                # Display the data
                display_dataframe(df)
                
            elif file_type == "excel":
                sheets = file_connector.read_excel(file_path)
                sheet_name = st.selectbox("Select a sheet", list(sheets.keys()))
                if sheet_name:
                    df = sheets[sheet_name]
                    st.session_state.current_data = df
                    st.session_state.data_source = f"Uploaded file: {uploaded_file.name} (Sheet: {sheet_name})"
                    st.session_state.data_source_type = f"Uploaded file: {uploaded_file.name} (Sheet: {sheet_name})"
                    st.session_state.is_data_loaded = True
                    # Create persistent copies for tab functions
                    persist_dataframes()
                    st.success(f"Successfully loaded data from {uploaded_file.name}, sheet {sheet_name}")
                    
                    # Display the data
                    display_dataframe(df)
            else:
                st.error(f"Unsupported file type: {file_type}")
                
            # After loading data, reinitialize all agents to ensure fresh state
            api_key = os.getenv("GOOGLE_API_KEY")
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

            st.session_state.explorer_agent = DataExplorerAgent(
                DataExplorerAgentConfig(
                    api_key=api_key,
                    model=model_name,
                    code_execution=True
                )
            )
            st.session_state.feature_engineer_agent = FeatureEngineerAgent(
                FeatureEngineerAgentConfig(
                    api_key=api_key,
                    model=model_name,
                    code_execution=True
                )
            )
            st.session_state.model_builder_agent = ModelBuilderAgent(
                ModelBuilderAgentConfig(
                    api_key=api_key,
                    model=model_name,
                    code_execution=True
                )
            )
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

def handle_database_connection() -> None:
    """Handle database connection."""
    st.subheader("Connect to Database")
    
    # Create the database connector if it doesn't exist
    if st.session_state.db_connector is None:
        try:
            st.session_state.db_connector = SQLiteConnector("data/test_database.db")
            st.success("Connected to the default SQLite database")
        except Exception as e:
            st.error(f"Error connecting to database: {str(e)}")
            return
    
    # Show available tables
    try:
        tables = st.session_state.db_connector.list_tables()
        
        if not tables:
            st.info("No tables found in the database")
            
            # Show option to set up test database
            if st.button("Set up test database with sample data"):
                with st.spinner("Setting up test database..."):
                    # Run the setup script
                    import subprocess
                    process = subprocess.Popen(["python", "scripts/setup_test_db.py"], 
                                            stdout=subprocess.PIPE, 
                                            stderr=subprocess.PIPE,
                                            universal_newlines=True)
                    stdout, stderr = process.communicate()
                    
                    if process.returncode == 0:
                        st.success("Test database set up successfully")
                        # Reconnect to the database
                        st.session_state.db_connector = SQLiteConnector("data/test_database.db")
                        tables = st.session_state.db_connector.list_tables()
                    else:
                        st.error(f"Error setting up test database: {stderr}")
        
        if tables:
            selected_table = st.selectbox("Select a table", tables)
            
            if selected_table:
                # Show preview
                preview = st.session_state.db_connector.get_table_preview(selected_table)
                st.write(f"Preview of table '{selected_table}':")
                st.dataframe(preview)
                
                # Option to load the full table
                if st.button(f"Load full table '{selected_table}'"):
                    with st.spinner(f"Loading table '{selected_table}'..."):
                        df = st.session_state.db_connector.get_table_data(selected_table)
                        st.session_state.current_data = df
                        st.session_state.data_source = f"Database table: {selected_table}"
                        st.session_state.data_source_type = "Database table"
                        st.session_state.is_data_loaded = True
                        # Create persistent copies for tab functions
                        persist_dataframes()
                        st.success(f"Successfully loaded table '{selected_table}'")
                        
                        # Display the full data
                        display_dataframe(df)
    
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")

    # After loading data, reinitialize all agents to ensure fresh state
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    st.session_state.explorer_agent = DataExplorerAgent(
        DataExplorerAgentConfig(
            api_key=api_key,
            model=model_name,
            code_execution=True
        )
    )
    st.session_state.feature_engineer_agent = FeatureEngineerAgent(
        FeatureEngineerAgentConfig(
            api_key=api_key,
            model=model_name,
            code_execution=True
        )
    )
    st.session_state.model_builder_agent = ModelBuilderAgent(
        ModelBuilderAgentConfig(
            api_key=api_key,
            model=model_name,
            code_execution=True
        )
    )

def display_agent_response(result, temp_dir=TEMP_DIR):
    """Display the agent's response, execute code blocks, show outputs/plots, and save artifacts."""
    import re
    import streamlit as st
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import pandas as pd
    import pickle
    import uuid
    import datetime

    response = result.get("response", "")
    code_blocks = []
    summary_parts = []
    in_code = False
    current_code = []
    for line in response.splitlines():
        if line.strip().startswith("```"):
            if in_code:
                code_blocks.append("\n".join(current_code))
                current_code = []
            in_code = not in_code
        elif in_code:
            current_code.append(line)
        else:
            summary_parts.append(line)
    # Show summary
    summary = "\n".join(summary_parts).strip()
    if summary:
        st.markdown(summary)
    # Execute and display code blocks
    for idx, code in enumerate(code_blocks):
        st.code(code, language="python")
        # Save code block
        code_filename = os.path.join(temp_dir, f"code_block_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.py")
        with open(code_filename, "w") as f:
            f.write(code)
        st.info(f"Code block saved to: {code_filename}")
        try:
            exec_result = result.get("locals", {})
            local_vars = dict(exec_result)
            if "df" in result.get("locals", {}):
                local_vars["df"] = result["locals"]["df"]
            import io, sys
            stdout = io.StringIO()
            stderr = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = stdout, stderr
            try:
                exec(code, local_vars)
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
            output = stdout.getvalue()
            if output:
                st.text(output)
            # Show and save any DataFrames created
            for name, value in local_vars.items():
                if isinstance(value, pd.DataFrame) and name != "df":
                    st.subheader(f"DataFrame: {name}")
                    # Sanitize before display
                    sanitized_df = sanitize_dataframe(value)
                    st.dataframe(sanitized_df)
                    # Save DataFrame as CSV
                    csv_filename = os.path.join(temp_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv")
                    value.to_csv(csv_filename, index=False)
                    st.info(f"DataFrame '{name}' saved to: {csv_filename}")
            # Show and save matplotlib plots
            if "plt" in local_vars:
                plt = local_vars["plt"]
                if plt.get_fignums():
                    st.pyplot(plt.gcf())
                    # Save plot as PNG
                    plot_filename = os.path.join(temp_dir, f"matplotlib_plot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png")
                    plt.savefig(plot_filename)
                    st.info(f"Matplotlib plot saved to: {plot_filename}")
                    plt.close('all')
            # Show and save plotly figures
            for name, value in local_vars.items():
                if isinstance(value, go.Figure):
                    st.plotly_chart(value)
                    plotly_filename = os.path.join(temp_dir, f"plotly_{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.html")
                    value.write_html(plotly_filename)
                    st.info(f"Plotly figure '{name}' saved to: {plotly_filename}")
            # Save any pickled objects
            for name, value in local_vars.items():
                if hasattr(value, "__module__") and ("sklearn" in value.__module__ or "xgboost" in value.__module__ or "lightgbm" in value.__module__):
                    pickle_filename = os.path.join(temp_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pkl")
                    with open(pickle_filename, "wb") as pf:
                        pickle.dump(value, pf)
                    st.info(f"Pickled object '{name}' saved to: {pickle_filename}")
        except Exception as e:
            st.error(f"Error executing code block: {e}")

def get_current_df():
    """Get the current dataframe from session state, with fallback to function-specific copies."""
    # First check if is_data_loaded flag is True
    if not st.session_state.is_data_loaded:
        st.error("No dataset loaded. Please load data first.")
        return None
    
    # Try to get the main dataframe
    df = st.session_state.get("current_data", None)
    
    # If the main dataframe is missing, try to get function-specific copies
    if df is None or (hasattr(df, 'empty') and df.empty):
        # Try feature engineering dataframe
        df = st.session_state.get("feature_engineering_df", None)
        if df is not None and not (hasattr(df, 'empty') and df.empty):
            st.warning("Restored data from feature engineering cache.")
        else:
            # Try modeling dataframe
            df = st.session_state.get("modeling_df", None)
            if df is not None and not (hasattr(df, 'empty') and df.empty):
                st.warning("Restored data from modeling cache.")
            else:
                # Try exploration dataframe
                df = st.session_state.get("exploration_df", None)
                if df is not None and not (hasattr(df, 'empty') and df.empty):
                    st.warning("Restored data from exploration cache.")
    
    # If we still couldn't find a valid dataframe, reset state and show error
    if df is None or (hasattr(df, 'empty') and df.empty):
        st.error("Dataset was lost. Please reload data.")
        st.session_state.is_data_loaded = False  # Reset flag since data is invalid
        return None
    
    # Sanitize the dataframe to avoid Arrow errors
    df = sanitize_dataframe(df)
    
    return df

def explore_data():
    """Explore data with the agent."""
    st.subheader("Exploratory Data Analysis")
    df = get_current_df()
    if df is None:
        return
    
    # Show data source and shape
    st.info(f"Data Source: {st.session_state.data_source}")
    st.info(f"Data Shape: {df.shape}")
    
    # Display the current data
    st.subheader("Current Dataset")
    display_dataframe(df)
    
    # Analysis options
    analysis_type = st.selectbox(
        "Select Analysis Type",
        [
            "Custom Analysis",
            "Data Summary",
            "Missing Values Analysis",
            "Distribution Analysis",
            "Correlation Analysis",
            "Outlier Analysis",
            "Full EDA Report"
        ]
    )
    
    if analysis_type == "Custom Analysis":
        query = st.text_area("Enter your analysis question or request:",
                            "Please analyze this dataset and provide key insights.")
        
        if st.button("Run Custom Analysis"):
            with st.spinner("Running analysis..."):
                try:
                    # Pass the current dataframe to the agent
                    result = st.session_state.explorer_agent.run(query, data=df)
                    display_agent_response(result)
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
    
    elif analysis_type == "Data Summary":
        if st.button("Generate Data Summary"):
            with st.spinner("Generating data summary..."):
                try:
                    # Pass the current dataframe to the agent
                    result = st.session_state.explorer_agent.run(
                        "Please provide a comprehensive summary of this dataset, including data types, missing values, and basic statistics.",
                        data=df
                    )
                    display_agent_response(result)
                except Exception as e:
                    st.error(f"Error generating data summary: {str(e)}")
    
    elif analysis_type == "Missing Values Analysis":
        if st.button("Analyze Missing Values"):
            with st.spinner("Analyzing missing values..."):
                try:
                    # Pass the current dataframe to the agent
                    result = st.session_state.explorer_agent.run(
                        "Please analyze the missing values in this dataset, including their distribution and potential impact.",
                        data=df
                    )
                    display_agent_response(result)
                except Exception as e:
                    st.error(f"Error analyzing missing values: {str(e)}")
    
    elif analysis_type == "Distribution Analysis":
        if st.button("Analyze Distributions"):
            with st.spinner("Analyzing distributions..."):
                try:
                    # Pass the current dataframe to the agent
                    result = st.session_state.explorer_agent.run(
                        "Please analyze the distributions of variables in this dataset, including histograms and summary statistics.",
                        data=df
                    )
                    display_agent_response(result)
                except Exception as e:
                    st.error(f"Error analyzing distributions: {str(e)}")
    
    elif analysis_type == "Correlation Analysis":
        if st.button("Analyze Correlations"):
            with st.spinner("Analyzing correlations..."):
                try:
                    # Pass the current dataframe to the agent
                    result = st.session_state.explorer_agent.run(
                        "Please analyze the correlations between variables in this dataset, including visualizations and key findings.",
                        data=df
                    )
                    display_agent_response(result)
                except Exception as e:
                    st.error(f"Error analyzing correlations: {str(e)}")
    
    elif analysis_type == "Outlier Analysis":
        if st.button("Analyze Outliers"):
            with st.spinner("Analyzing outliers..."):
                try:
                    # Pass the current dataframe to the agent
                    result = st.session_state.explorer_agent.run(
                        "Please analyze the outliers in this dataset, including their identification and potential impact.",
                        data=df
                    )
                    display_agent_response(result)
                except Exception as e:
                    st.error(f"Error analyzing outliers: {str(e)}")
    
    elif analysis_type == "Full EDA Report":
        if st.button("Generate Full EDA Report"):
            with st.spinner("Generating full EDA report..."):
                try:
                    # Pass the current dataframe to the agent
                    result = st.session_state.explorer_agent.run(
                        "Please provide a comprehensive exploratory data analysis report for this dataset, including all relevant visualizations and insights.",
                        data=df
                    )
                    display_agent_response(result)
                except Exception as e:
                    st.error(f"Error generating full EDA report: {str(e)}")

def engineer_features():
    """Engineer features with the agent."""
    st.subheader("Feature Engineering")
    df = get_current_df()
    if df is None:
        return
    
    # Show data source and shape
    st.info(f"Data Source: {st.session_state.data_source}")
    st.info(f"Data Shape: {df.shape}")
    
    # Display the current data
    st.subheader("Current Dataset")
    display_dataframe(df)
    
    # Analysis options
    engineering_type = st.selectbox(
        "Select Feature Engineering Type",
        [
            "Custom Feature Engineering",
            "Basic Feature Engineering",
            "Encode Categorical Features",
            "Normalize Features",
            "Feature Selection",
            "Dimensionality Reduction",
            "Time Series Features"
        ]
    )
    
    if engineering_type == "Custom Feature Engineering":
        query = st.text_area("Enter your feature engineering request:",
                            "Please engineer features for this dataset to improve model performance.")
        
        if st.button("Run Custom Feature Engineering"):
            with st.spinner("Engineering features..."):
                try:
                    # Verify data is valid before proceeding
                    if not verify_data(df, "custom feature engineering"):
                        return
                        
                    # Get a fresh copy from session state to avoid any issues
                    if st.session_state.get("feature_engineering_df") is not None:
                        df = st.session_state.feature_engineering_df.copy()
                    
                    # Make sure the dataframe is sanitized
                    df = sanitize_dataframe(df)
                    
                    # CRITICAL: Directly set the dataframe in the agent's code generator
                    st.session_state.feature_engineer_agent.code_generator.locals_dict["df"] = df
                    
                    result = st.session_state.feature_engineer_agent.custom_feature_engineering(df, query)
                    
                    # Log the error message if any
                    if not result.get("success", False):
                        st.error(f"Agent error: {result.get('response', 'Unknown error')}")
                        # Check what's in the agent's locals dict
                        agent_df = st.session_state.feature_engineer_agent.code_generator.locals_dict.get("df")
                        if agent_df is None:
                            st.error("Agent lost reference to the DataFrame")
                        else:
                            st.info(f"Agent DataFrame shape: {agent_df.shape}")
                            
                    display_agent_response(result)
                    
                    # Store engineered data if available
                    locals_dict = result.get("locals", {})
                    for name, value in locals_dict.items():
                        if isinstance(value, pd.DataFrame) and name != "df":
                            st.success(f"Engineered data available as DataFrame: {name}")
                            st.session_state.engineered_data = value
                            
                            # Display the engineered data
                            st.subheader("Engineered Data Preview")
                            display_dataframe(value)
                    
                except Exception as e:
                    st.error(f"Error engineering features: {str(e)}")
    
    elif engineering_type == "Basic Feature Engineering":
        # Get a list of potential target variables
        target_options = df.columns.tolist()
        target_options.insert(0, "None")
        target_variable = st.selectbox("Select Target Variable (optional)", target_options)
        
        if target_variable == "None":
            target_variable = None
        
        if st.button("Run Basic Feature Engineering"):
            with st.spinner("Engineering features..."):
                try:
                    # Verify data is valid before proceeding
                    if not verify_data(df, "feature engineering"):
                        return
                    
                    # Get a fresh copy from session state to avoid any issues
                    if st.session_state.get("feature_engineering_df") is not None:
                        df = st.session_state.feature_engineering_df.copy()
                    
                    # Make sure the dataframe is sanitized
                    df = sanitize_dataframe(df)
                    
                    # Add progress indicator
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    progress_bar.progress(10)
                    status_text.text("Preparing data...")
                    
                    # CRITICAL: Directly set the dataframe in the agent's code generator
                    st.session_state.feature_engineer_agent.code_generator.locals_dict["df"] = df
                    
                    # Update progress
                    progress_bar.progress(30)
                    status_text.text("Sending request to LLM...")
                    
                    # Implement a simple timeout for the LLM call
                    import concurrent.futures
                    
                    def run_with_timeout(func, *args, timeout=120, **kwargs):
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(func, *args, **kwargs)
                            try:
                                return future.result(timeout=timeout)
                            except concurrent.futures.TimeoutError:
                                return {"success": False, "response": "LLM request timed out after 120 seconds. Please try again or try with a simpler request."}
                    
                    # Run the feature engineering with timeout
                    result = run_with_timeout(
                        st.session_state.feature_engineer_agent.engineer_features,
                        df, target_variable, timeout=120
                    )
                    
                    # Update progress
                    progress_bar.progress(80)
                    status_text.text("Processing response...")
                    
                    # Log the error message if any
                    if not result.get("success", False):
                        st.error(f"Agent error: {result.get('response', 'Unknown error')}")
                        # Check what's in the agent's locals dict
                        agent_df = st.session_state.feature_engineer_agent.code_generator.locals_dict.get("df")
                        if agent_df is None:
                            st.error("Agent lost reference to the DataFrame")
                        else:
                            st.info(f"Agent DataFrame shape: {agent_df.shape}")
                            
                        # Direct fallback to basic feature engineering implementation if the agent fails
                        st.warning("Attempting fallback to basic feature engineering implementation...")
                        
                        try:
                            engineered_df = df.copy()
                            
                            # Generate basic numeric features
                            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                            if len(numeric_cols) > 0:
                                # Log transformations
                                for col in numeric_cols:
                                    if (df[col] > 0).all():
                                        engineered_df[f'{col}_log'] = np.log(df[col])
                                
                                # Square and square root transformations
                                for col in numeric_cols:
                                    engineered_df[f'{col}_squared'] = df[col] ** 2
                                    if (df[col] >= 0).all():
                                        engineered_df[f'{col}_sqrt'] = np.sqrt(df[col])
                                
                                # Interactions between numeric features (up to 5 combinations to avoid explosion)
                                if len(numeric_cols) >= 2:
                                    import itertools
                                    for col1, col2 in list(itertools.combinations(numeric_cols, 2))[:5]:
                                        engineered_df[f'{col1}_times_{col2}'] = df[col1] * df[col2]
                                        engineered_df[f'{col1}_by_{col2}'] = df[col1] / df[col2].replace(0, np.nan)
                            
                            # One-hot encode categorical features
                            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                            if len(categorical_cols) > 0:
                                for col in categorical_cols:
                                    if df[col].nunique() < 10:  # Only one-hot encode if fewer than 10 categories
                                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                                        engineered_df = pd.concat([engineered_df, dummies], axis=1)
                            
                            # Drop columns with too many NaN values
                            engineered_df = engineered_df.dropna(axis=1, thresh=len(engineered_df) * 0.8)
                            
                            # Display results
                            st.success("Fallback feature engineering completed successfully")
                            st.subheader("Feature Engineering Result")
                            display_dataframe(engineered_df)
                            
                            # Show feature counts
                            st.subheader("Feature Summary")
                            st.write(f"Original features: {df.shape[1]}")
                            st.write(f"New features: {engineered_df.shape[1]}")
                            st.write(f"Added features: {engineered_df.shape[1] - df.shape[1]}")
                            
                            # Save the result
                            st.session_state.engineered_data = engineered_df
                            
                        except Exception as e:
                            st.error(f"Fallback feature engineering failed: {str(e)}")
                    else:
                        # If successful, store engineered data if available
                        locals_dict = result.get("locals", {})
                        for name, value in locals_dict.items():
                            if isinstance(value, pd.DataFrame) and name != "df":
                                st.success(f"Engineered data available as DataFrame: {name}")
                                st.session_state.engineered_data = value
                                
                                # Display the engineered data
                                st.subheader("Engineered Data Preview")
                                display_dataframe(value)
                    
                    # Complete progress
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    # Display the agent's response
                    display_agent_response(result)
                    
                except Exception as e:
                    st.error(f"Error engineering features: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    elif engineering_type == "Encode Categorical Features":
        # Allow selection of categorical columns
        categorical_columns = st.multiselect(
            "Select categorical columns (leave empty for auto-detection)",
            df.columns.tolist()
        )
        
        if not categorical_columns:
            categorical_columns = None
        
        if st.button("Encode Categorical Features"):
            with st.spinner("Encoding categorical features..."):
                try:
                    result = st.session_state.feature_engineer_agent.encode_categorical_features(df, categorical_columns)
                    display_agent_response(result)
                    
                    # Store engineered data if available
                    locals_dict = result.get("locals", {})
                    for name, value in locals_dict.items():
                        if isinstance(value, pd.DataFrame) and name != "df":
                            st.success(f"Encoded data available as DataFrame: {name}")
                            st.session_state.engineered_data = value
                            
                            # Display the engineered data
                            st.subheader("Encoded Data Preview")
                            display_dataframe(value)
                    
                except Exception as e:
                    st.error(f"Error encoding categorical features: {str(e)}")
    
    elif engineering_type == "Normalize Features":
        # Allow selection of numeric columns
        numeric_columns = st.multiselect(
            "Select numeric columns (leave empty for auto-detection)",
            df.select_dtypes(include=np.number).columns.tolist()
        )
        
        if not numeric_columns:
            numeric_columns = None
        
        if st.button("Normalize Features"):
            with st.spinner("Normalizing features..."):
                try:
                    result = st.session_state.feature_engineer_agent.normalize_features(df, numeric_columns)
                    display_agent_response(result)
                    
                    # Store engineered data if available
                    locals_dict = result.get("locals", {})
                    for name, value in locals_dict.items():
                        if isinstance(value, pd.DataFrame) and name != "df":
                            st.success(f"Normalized data available as DataFrame: {name}")
                            st.session_state.engineered_data = value
                            
                            # Display the engineered data
                            st.subheader("Normalized Data Preview")
                            display_dataframe(value)
                    
                except Exception as e:
                    st.error(f"Error normalizing features: {str(e)}")
    
    elif engineering_type == "Feature Selection":
        # Get target variable for feature selection
        target_variable = st.selectbox("Select Target Variable", df.columns.tolist())
        
        # Number of features to select
        n_features = st.slider("Number of features to select", 1, len(df.columns) - 1, min(5, len(df.columns) - 1))
        
        if st.button("Select Features"):
            with st.spinner("Selecting features..."):
                try:
                    result = st.session_state.feature_engineer_agent.select_features(df, target_variable, n_features)
                    display_agent_response(result)
                    
                    # Store selected features data if available
                    locals_dict = result.get("locals", {})
                    for name, value in locals_dict.items():
                        if isinstance(value, pd.DataFrame) and name != "df":
                            st.success(f"Selected features data available as DataFrame: {name}")
                            st.session_state.engineered_data = value
                            
                            # Display the selected features data
                            st.subheader("Selected Features Data Preview")
                            display_dataframe(value)
                    
                except Exception as e:
                    st.error(f"Error selecting features: {str(e)}")
    
    elif engineering_type == "Dimensionality Reduction":
        # Number of components for dimensionality reduction
        n_components = st.slider("Number of components", 2, min(10, len(df.columns) - 1), 2)
        
        if st.button("Reduce Dimensionality"):
            with st.spinner("Reducing dimensionality..."):
                try:
                    # Verify data is valid before proceeding
                    if not verify_data(df, "dimensionality reduction"):
                        return
                    
                    # Get a fresh copy from session state to avoid any issues
                    if st.session_state.get("feature_engineering_df") is not None:
                        df = st.session_state.feature_engineering_df.copy()
                    
                    # Make sure the dataframe is sanitized
                    df = sanitize_dataframe(df)
                    
                    # Log info for debugging
                    st.info(f"Running dimensionality reduction with {n_components} components on data shape {df.shape}")
                    
                    # Add progress indicator
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Update progress
                    progress_bar.progress(10)
                    status_text.text("Preparing data...")
                    
                    # CRITICAL: Directly set the dataframe in the agent's code generator
                    st.session_state.feature_engineer_agent.code_generator.locals_dict["df"] = df
                    
                    # Update progress
                    progress_bar.progress(30)
                    status_text.text("Sending request to LLM...")
                    
                    # Implement a simple timeout for the LLM call
                    import concurrent.futures
                    import time
                    
                    def run_with_timeout(func, *args, timeout=120, **kwargs):
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(func, *args, **kwargs)
                            try:
                                return future.result(timeout=timeout)
                            except concurrent.futures.TimeoutError:
                                return {"success": False, "response": "LLM request timed out after 120 seconds. Please try again or try with a simpler request."}
                    
                    # Run the dimensionality reduction with timeout
                    result = run_with_timeout(
                        st.session_state.feature_engineer_agent.reduce_dimensionality,
                        df, n_components, timeout=120
                    )
                    
                    # Update progress
                    progress_bar.progress(80)
                    status_text.text("Processing response...")
                    
                    # Log the error message if any
                    if not result.get("success", False):
                        st.error(f"Agent error: {result.get('response', 'Unknown error')}")
                        # Check what's in the agent's locals dict
                        agent_df = st.session_state.feature_engineer_agent.code_generator.locals_dict.get("df")
                        if agent_df is None:
                            st.error("Agent lost reference to the DataFrame")
                        else:
                            st.info(f"Agent DataFrame shape: {agent_df.shape}")
                            
                        # Direct fallback to basic PCA implementation if the agent fails
                        st.warning("Attempting fallback to basic PCA implementation...")
                        
                        try:
                            from sklearn.decomposition import PCA
                            from sklearn.preprocessing import StandardScaler
                            
                            # Select only numeric columns
                            numeric_df = df.select_dtypes(include=['int64', 'float64'])
                            if numeric_df.shape[1] < 2:
                                st.error("Not enough numeric columns for PCA")
                                return
                                
                            # Scale the data
                            scaler = StandardScaler()
                            scaled_data = scaler.fit_transform(numeric_df)
                            
                            # Perform PCA
                            pca = PCA(n_components=min(n_components, numeric_df.shape[1]))
                            pca_result = pca.fit_transform(scaled_data)
                            
                            # Create DataFrame with results
                            pca_df = pd.DataFrame(
                                data=pca_result,
                                columns=[f'PC{i+1}' for i in range(pca_result.shape[1])]
                            )
                            
                            # Add original index
                            pca_df.index = df.index
                            
                            # Display results
                            st.success("Fallback PCA completed successfully")
                            st.subheader("PCA Result")
                            display_dataframe(pca_df)
                            
                            # Show variance explained
                            st.subheader("Variance Explained")
                            explained_variance = pca.explained_variance_ratio_
                            st.bar_chart({f'PC{i+1}': var for i, var in enumerate(explained_variance)})
                            
                            # Save the result
                            st.session_state.engineered_data = pca_df
                            
                        except Exception as e:
                            st.error(f"Fallback PCA failed: {str(e)}")
                    
                    # Complete progress
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    # Display the agent's response
                    display_agent_response(result)
                    
                except Exception as e:
                    st.error(f"Error reducing dimensionality: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    elif engineering_type == "Time Series Features":
        # Get time column
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        potential_time_cols = datetime_cols + object_cols
        
        if not potential_time_cols:
            st.warning("No potential date/time columns found in the dataset")
            return
        
        time_column = st.selectbox("Select Time Column", potential_time_cols)
        
        if st.button("Create Time Series Features"):
            with st.spinner("Creating time series features..."):
                try:
                    result = st.session_state.feature_engineer_agent.handle_time_series_features(df, time_column)
                    display_agent_response(result)
                    
                    # Store time series data if available
                    locals_dict = result.get("locals", {})
                    for name, value in locals_dict.items():
                        if isinstance(value, pd.DataFrame) and name != "df":
                            st.success(f"Time series data available as DataFrame: {name}")
                            st.session_state.engineered_data = value
                            
                            # Display the time series data
                            st.subheader("Time Series Data Preview")
                            display_dataframe(value)
                    
                except Exception as e:
                    st.error(f"Error creating time series features: {str(e)}")
    
    # Option to use engineered data
    if st.session_state.engineered_data is not None:
        if st.button("Use Engineered Data for Modeling"):
            # Sanitize the engineered data before storing it
            sanitized_data = sanitize_dataframe(st.session_state.engineered_data)
            if sanitized_data is not None:
                st.session_state.current_data = sanitized_data
                st.session_state.data_source = f"{st.session_state.data_source} (Engineered)"
                st.success("Now using engineered data for modeling")
            else:
                st.error("Could not use engineered data due to data type issues")

def build_model():
    """Build model with the agent."""
    st.subheader("Model Building")
    df = get_current_df()
    if df is None:
        return
    
    # Show data source and shape
    st.info(f"Data Source: {st.session_state.data_source}")
    st.info(f"Data Shape: {df.shape}")
    
    # Display the current data
    st.subheader("Current Dataset")
    display_dataframe(df)
    
    # Add a separator
    st.markdown("---")
    
    # Modeling options
    st.subheader("Select Model Options")
    modeling_type = st.selectbox(
        "Select Modeling Type",
        [
            "Custom Modeling",
            "Build Single Model",
            "Compare Models",
            "Tune Hyperparameters",
            "Cross-Validation"
        ]
    )
    
    if modeling_type == "Custom Modeling":
        query = st.text_area("Enter your modeling request:",
                            "Please build a machine learning model for this dataset.")
        
        if st.button("Run Custom Modeling"):
            with st.spinner("Building model..."):
                try:
                    # Verify data is valid before proceeding
                    if not verify_data(df, "custom modeling"):
                        return
                        
                    # Get a fresh copy from session state to avoid any issues
                    if st.session_state.get("modeling_df") is not None:
                        df = st.session_state.modeling_df.copy()
                        
                    # Make sure data is sanitized
                    df = sanitize_dataframe(df)
                    
                    # Log info for debugging
                    st.info(f"Running custom modeling on data shape {df.shape}")
                    
                    result = st.session_state.model_builder_agent.custom_modeling(df, query)
                    display_agent_response(result)
                    st.session_state.model_results = {
                        "type": "custom",
                        "result": result
                    }
                except Exception as e:
                    st.error(f"Error building model: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    elif modeling_type == "Build Single Model":
        # Get target variable
        target_variable = st.selectbox("Select Target Variable", df.columns.tolist())
        
        # Get model type
        model_type = st.selectbox(
            "Select Model Type",
            ["Auto-detect", "Classification", "Regression"]
        )
        
        model_type = None if model_type == "Auto-detect" else model_type.lower()
        
        # Test size
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        if st.button("Build Model"):
            with st.spinner("Building model..."):
                try:
                    # CRITICAL: Directly set the dataframe in the agent's code generator
                    st.session_state.model_builder_agent.code_generator.locals_dict["df"] = df
                    
                    result = st.session_state.model_builder_agent.build_model(
                        df, target_variable, model_type, test_size
                    )
                    
                    # Log the error message if any
                    if not result.get("success", False):
                        st.error(f"Agent error: {result.get('response', 'Unknown error')}")
                        # Check what's in the agent's locals dict
                        agent_df = st.session_state.model_builder_agent.code_generator.locals_dict.get("df")
                        if agent_df is None:
                            st.error("Agent lost reference to the DataFrame")
                        else:
                            st.info(f"Agent DataFrame shape: {agent_df.shape}")
                    
                    display_agent_response(result)
                    st.session_state.model_results = {
                        "type": "single_model",
                        "target": target_variable,
                        "model_type": model_type,
                        "result": result
                    }
                except Exception as e:
                    st.error(f"Error building model: {str(e)}")
    
    elif modeling_type == "Compare Models":
        # Get target variable
        target_variable = st.selectbox("Select Target Variable", df.columns.tolist())
        
        # Get model type
        model_type = st.selectbox(
            "Select Model Type",
            ["Auto-detect", "Classification", "Regression"]
        )
        
        model_type = None if model_type == "Auto-detect" else model_type.lower()
        
        # Test size
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        if st.button("Compare Models"):
            with st.spinner("Comparing models..."):
                try:
                    result = st.session_state.model_builder_agent.compare_models(
                        df, target_variable, model_type, test_size
                    )
                    display_agent_response(result)
                    st.session_state.model_results = {
                        "type": "compare_models",
                        "target": target_variable,
                        "model_type": model_type,
                        "result": result
                    }
                except Exception as e:
                    st.error(f"Error comparing models: {str(e)}")
    
    elif modeling_type == "Tune Hyperparameters":
        # Get target variable
        target_variable = st.selectbox("Select Target Variable", df.columns.tolist())
        
        # Get model type
        model_type = st.selectbox(
            "Select Model Type",
            ["RandomForest", "XGBoost", "LightGBM", "LinearRegression", "LogisticRegression"]
        )
        
        # Test size
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
        
        if st.button("Tune Hyperparameters"):
            with st.spinner("Tuning hyperparameters..."):
                try:
                    result = st.session_state.model_builder_agent.tune_hyperparameters(
                        df, target_variable, model_type, test_size
                    )
                    display_agent_response(result)
                    st.session_state.model_results = {
                        "type": "tune_hyperparameters",
                        "target": target_variable,
                        "model_type": model_type,
                        "result": result
                    }
                except Exception as e:
                    st.error(f"Error tuning hyperparameters: {str(e)}")
    
    elif modeling_type == "Cross-Validation":
        # Get target variable
        target_variable = st.selectbox("Select Target Variable", df.columns.tolist())
        
        # Get model type
        model_type = st.selectbox(
            "Select Model Type",
            ["RandomForest", "XGBoost", "LightGBM", "LinearRegression", "LogisticRegression"]
        )
        
        # Number of folds
        n_splits = st.slider("Number of Folds", 3, 10, 5)
        
        if st.button("Run Cross-Validation"):
            with st.spinner("Running cross-validation..."):
                try:
                    result = st.session_state.model_builder_agent.cross_validation(
                        df, target_variable, model_type, n_splits
                    )
                    display_agent_response(result)
                    st.session_state.model_results = {
                        "type": "cross_validation",
                        "target": target_variable,
                        "model_type": model_type,
                        "result": result
                    }
                except Exception as e:
                    st.error(f"Error running cross-validation: {str(e)}")

def verify_data(df, action_name="this action"):
    """Verify that data exists and is valid before running agents."""
    if df is None:
        st.error(f"No valid dataset found for {action_name}. Please reload your data.")
        return False
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Quick check for common data issues
    try:
        # Check for at least some rows and columns
        if df_copy.shape[0] == 0 or df_copy.shape[1] == 0:
            st.error(f"Dataset is empty (shape: {df_copy.shape}). Please reload your data.")
            return False
            
        # Check for all NaN values
        if df_copy.isna().all().all():
            st.error("Dataset contains only missing values. Please check your data.")
            return False
            
        # Check for infinite values and replace with NaN to avoid errors
        df_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        return True
    except Exception as e:
        st.error(f"Error verifying data: {str(e)}")
        return False

def main():
    """Main function to render the Streamlit app."""
    st.title("🧪 Agentic Data Scientist")
    
    # Add a sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Add a selectbox for agent type
        agent_type = st.selectbox(
            "Select Agent Framework",
            ["LangChain (default)", "AutoGen"],
            index=0,
            help="Choose between LangChain and AutoGen agent implementations."
        )
        
        # Update the agent type in session state
        st.session_state.agent_type = "langchain" if agent_type == "LangChain (default)" else "autogen"
        
        # Add API key input
        api_key = st.text_input("Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        
        # Initialize agents button
        if st.button("Initialize Agents"):
            if initialize_agents():
                st.success(f"Successfully initialized {agent_type} agents!")
            else:
                st.error("Failed to initialize agents. Check your API key and try again.")
        
        # Show the current agent framework
        st.info(f"Current Agent Framework: {agent_type}")
        
        # Reset session
        if st.button("Reset Session"):
            # Clear session state except for agent_type
            current_agent_type = st.session_state.agent_type
            for key in list(st.session_state.keys()):
                if key != "agent_type":
                    del st.session_state[key]
            st.session_state.agent_type = current_agent_type
            st.experimental_rerun()
    
    # Check if agents are initialized
    agents_initialized = all([
        "explorer_agent" in st.session_state,
        "feature_engineer_agent" in st.session_state,
        "model_builder_agent" in st.session_state
    ])
    
    if not agents_initialized:
        st.warning("Agents not initialized. Please enter your Google Gemini API key and click Initialize Agents.")
        if not os.getenv("GOOGLE_API_KEY"):
            st.info("You can get a Google Gemini API key from https://makersuite.google.com/")
        return

    # Main content
    tabs = st.tabs(["Load Data", "Explore Data", "Engineer Features", "Build Models"])

    # Check if data is already loaded
    if st.session_state.is_data_loaded:
        with tabs[0]:
            st.header("Load Data")
            st.info(f"Data already loaded from: {st.session_state.data_source}")
            display_dataframe(st.session_state.current_data) # Display the loaded data
            
            # Create copies of the data for each tab function to prevent state loss
            persist_dataframes()

        # Proceed to other tabs as data is available
        with tabs[1]:
            explore_data()

        with tabs[2]:
            engineer_features()

        with tabs[3]:
            build_model()

    else:
        # If no data loaded, only show the Load Data tab initially
        with tabs[0]:
            st.header("Load Data")

            # Choose data source
            data_source = st.radio("Choose a data source", ["Upload File", "Connect to Database"])

            if data_source == "Upload File":
                handle_file_upload()
            else:
                handle_database_connection()

        # Inform the user to load data first for other tabs
        with tabs[1]:
            st.info("Please load data in the 'Load Data' tab to proceed with data exploration.")
        with tabs[2]:
            st.info("Please load data in the 'Load Data' tab to proceed with feature engineering.")
        with tabs[3]:
            st.info("Please load data in the 'Load Data' tab to proceed with model building.")

if __name__ == "__main__":
    main() 