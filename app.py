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

# Import our modules
from config import config
from agentic_data_scientist.connectors.sqlite_connector import SQLiteConnector
from agentic_data_scientist.connectors.file_connector import FileConnector
from agentic_data_scientist.agents.data_explorer import DataExplorerAgent, DataExplorerAgentConfig
from agentic_data_scientist.agents.feature_engineer import FeatureEngineerAgent, FeatureEngineerAgentConfig
from agentic_data_scientist.agents.model_builder import ModelBuilderAgent, ModelBuilderAgentConfig

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
        
        # Initialize the data explorer agent
        explorer_config = DataExplorerAgentConfig(
            name="DataExplorerAgent",
            description="Agent for exploring and analyzing data",
            api_key=api_key,
            model=model_name,
            code_execution=True
        )
        st.session_state.explorer_agent = DataExplorerAgent(explorer_config)
        
        # Initialize the feature engineer agent
        engineer_config = FeatureEngineerAgentConfig(
            name="FeatureEngineerAgent",
            description="Agent for engineering features",
            api_key=api_key,
            model=model_name,
            code_execution=True
        )
        st.session_state.feature_engineer_agent = FeatureEngineerAgent(engineer_config)
        
        # Initialize the model builder agent
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

def display_dataframe(df: pd.DataFrame, max_rows: int = 10) -> None:
    """Display a dataframe with a download button."""
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
                    st.success(f"Successfully loaded data from {uploaded_file.name}, sheet {sheet_name}")
                    
                    # Display the data
                    display_dataframe(df)
            else:
                st.error(f"Unsupported file type: {file_type}")
                
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
                        st.success(f"Successfully loaded table '{selected_table}'")
                        
                        # Display the full data
                        display_dataframe(df)
    
    except Exception as e:
        st.error(f"Error accessing database: {str(e)}")

def explore_data(df: pd.DataFrame) -> None:
    """Explore data with the agent."""
    st.subheader("Exploratory Data Analysis")
    
    if df is None:
        st.warning("Please load data first")
        return
    
    # Initialize agents if needed
    if st.session_state.explorer_agent is None:
        if not initialize_agents():
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
                    # Display figures if any
                    figures = []
                    locals_dict = result.get("locals", {})
                    for name, value in locals_dict.items():
                        if isinstance(value, pd.DataFrame) and len(value) < 1000:
                            st.subheader(f"DataFrame: {name}")
                            st.dataframe(value)
                    
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
                except Exception as e:
                    st.error(f"Error generating full EDA report: {str(e)}")

def engineer_features(df: pd.DataFrame) -> None:
    """Engineer features with the agent."""
    st.subheader("Feature Engineering")
    
    if df is None:
        st.warning("Please load data first")
        return
    
    # Initialize agents if needed
    if st.session_state.feature_engineer_agent is None:
        if not initialize_agents():
            return
    
    # Show data source and shape
    st.info(f"Data Source: {st.session_state.data_source}")
    st.info(f"Data Shape: {df.shape}")
    
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
                    result = st.session_state.feature_engineer_agent.custom_feature_engineering(df, query)
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
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
                    result = st.session_state.feature_engineer_agent.engineer_features(df, target_variable)
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
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
                    result = st.session_state.feature_engineer_agent.reduce_dimensionality(df, n_components)
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
                    # Store reduced data if available
                    locals_dict = result.get("locals", {})
                    for name, value in locals_dict.items():
                        if isinstance(value, pd.DataFrame) and name != "df":
                            st.success(f"Reduced data available as DataFrame: {name}")
                            st.session_state.engineered_data = value
                            
                            # Display the reduced data
                            st.subheader("Reduced Data Preview")
                            display_dataframe(value)
                    
                except Exception as e:
                    st.error(f"Error reducing dimensionality: {str(e)}")
    
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
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
            st.session_state.current_data = st.session_state.engineered_data
            st.session_state.data_source = f"{st.session_state.data_source} (Engineered)"
            st.success("Now using engineered data for modeling")

def build_model(df: pd.DataFrame) -> None:
    """Build model with the agent."""
    st.subheader("Model Building")
    
    if df is None:
        st.warning("Please load data first")
        return
    
    # Initialize agents if needed
    if st.session_state.model_builder_agent is None:
        if not initialize_agents():
            return
    
    # Show data source and shape
    st.info(f"Data Source: {st.session_state.data_source}")
    st.info(f"Data Shape: {df.shape}")
    
    # Modeling options
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
                    result = st.session_state.model_builder_agent.custom_modeling(df, query)
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
                    # Store model results
                    st.session_state.model_results = {
                        "type": "custom",
                        "result": result
                    }
                    
                except Exception as e:
                    st.error(f"Error building model: {str(e)}")
    
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
                    result = st.session_state.model_builder_agent.build_model(
                        df, target_variable, model_type, test_size
                    )
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
                    # Store model results
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
                    # Store model results
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
                    # Store model results
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
                    
                    # Display the chat history
                    for message in result.get("chat_history", []):
                        if message["role"] == "user":
                            display_chat_message("user", message["content"])
                        else:
                            display_chat_message("assistant", message["content"])
                    
                    # Store model results
                    st.session_state.model_results = {
                        "type": "cross_validation",
                        "target": target_variable,
                        "model_type": model_type,
                        "result": result
                    }
                    
                except Exception as e:
                    st.error(f"Error running cross-validation: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    # Apply nest_asyncio to handle nested event loops
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)
    
    st.title("Agentic Data Scientist 🧪")
    st.markdown("""
    This application provides an AI-powered data science team to help you perform various data science tasks.
    Load your data, explore it, engineer features, and build machine learning models - all with the help of AI agents.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This application uses AI agents to help you perform data science tasks.
        It's powered by Google Gemini Pro LLM and runs locally.
        """
    )
    
    # Main content
    tabs = st.tabs(["Load Data", "Explore Data", "Engineer Features", "Build Models"])
    
    with tabs[0]:
        st.header("Load Data")
        
        # Choose data source
        data_source = st.radio("Choose a data source", ["Upload File", "Connect to Database"])
        
        if data_source == "Upload File":
            handle_file_upload()
        else:
            handle_database_connection()
    
    with tabs[1]:
        explore_data(st.session_state.current_data)
    
    with tabs[2]:
        engineer_features(st.session_state.current_data)
    
    with tabs[3]:
        build_model(st.session_state.current_data)

if __name__ == "__main__":
    main() 