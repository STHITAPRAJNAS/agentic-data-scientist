#!/usr/bin/env python
"""Example script showing basic usage of the Agentic Data Scientist agents."""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_data_scientist.agents.data_explorer import DataExplorerAgent, DataExplorerAgentConfig
from agentic_data_scientist.agents.feature_engineer import FeatureEngineerAgent, FeatureEngineerAgentConfig
from agentic_data_scientist.agents.model_builder import ModelBuilderAgent, ModelBuilderAgentConfig
from agentic_data_scientist.connectors.sqlite_connector import SQLiteConnector
from agentic_data_scientist.connectors.file_connector import FileConnector

def main():
    """Main function to demonstrate the usage of agents."""
    # Get API key from environment variable or prompt user
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        api_key = input("Please enter your Google Gemini API Key: ")
        os.environ["GOOGLE_API_KEY"] = api_key
    
    print("Setting up database connector...")
    db_path = "../data/test_database.db"
    
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        alt_path = "data/test_database.db"
        if os.path.exists(alt_path):
            db_path = alt_path
            print(f"Using alternative path: {db_path}")
        else:
            print("Please run scripts/setup_test_db.py first to create the test database")
            return
    
    # Set up database connector
    connector = SQLiteConnector(db_path)
    
    # Show available tables
    tables = connector.list_tables()
    print(f"Available tables: {tables}")
    
    if not tables:
        print("No tables found in the database. Please run scripts/setup_test_db.py first.")
        return
    
    # Load a sample dataset (customer_churn)
    table_name = "customer_churn" if "customer_churn" in tables else tables[0]
    print(f"Loading data from table: {table_name}")
    
    df = connector.get_table_data(table_name)
    print(f"Data loaded: {df.shape}")
    
    # Initialize the Data Explorer Agent
    print("\nInitializing Data Explorer Agent...")
    explorer_config = DataExplorerAgentConfig(
        api_key=api_key,
        code_execution=True
    )
    explorer_agent = DataExplorerAgent(explorer_config)
    
    # Run a simple data exploration
    print("\nRunning data exploration...")
    result = explorer_agent.summarize_data(df)
    
    # Print the last message from the agent
    if result.get("chat_history"):
        last_message = next((m for m in reversed(result["chat_history"]) 
                            if m["role"] == "assistant"), None)
        if last_message:
            print("\nData Explorer Agent Summary:")
            print(last_message["content"][:500] + "...\n")
    
    # Initialize the Feature Engineer Agent
    print("\nInitializing Feature Engineer Agent...")
    engineer_config = FeatureEngineerAgentConfig(
        api_key=api_key,
        code_execution=True
    )
    feature_agent = FeatureEngineerAgent(engineer_config)
    
    # Get target variable (assuming it's a binary target if available)
    binary_columns = [col for col in df.columns if df[col].nunique() == 2]
    target_variable = binary_columns[0] if binary_columns else df.columns[-1]
    print(f"Using target variable: {target_variable}")
    
    # Run feature engineering
    print("\nRunning feature engineering...")
    result = feature_agent.engineer_features(df, target_variable)
    
    # Get the engineered dataframe if available
    engineered_df = None
    for name, value in result.get("locals", {}).items():
        if isinstance(value, pd.DataFrame) and name != "df":
            engineered_df = value
            print(f"\nEngineered data available: {name} with shape {value.shape}")
            break
    
    # Use original data if no engineered data is available
    if engineered_df is None:
        print("No engineered data found, using original data for modeling")
        engineered_df = df
    
    # Initialize the Model Builder Agent
    print("\nInitializing Model Builder Agent...")
    model_config = ModelBuilderAgentConfig(
        api_key=api_key,
        code_execution=True
    )
    model_agent = ModelBuilderAgent(model_config)
    
    # Run model building
    print("\nBuilding a simple model...")
    result = model_agent.build_model(engineered_df, target_variable)
    
    # Print the last message from the agent
    if result.get("chat_history"):
        last_message = next((m for m in reversed(result["chat_history"]) 
                            if m["role"] == "assistant"), None)
        if last_message:
            print("\nModel Builder Agent Results:")
            print(last_message["content"][:500] + "...\n")
    
    print("\nDemonstration complete. The agents have successfully:")
    print("1. Explored the dataset")
    print("2. Engineered features for the target variable")
    print("3. Built a machine learning model")

if __name__ == "__main__":
    main() 