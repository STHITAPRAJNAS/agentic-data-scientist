#!/usr/bin/env python
"""Example script to demonstrate the AutoGen-based Agentic Data Scientist."""
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from sklearn.datasets import load_breast_cancer

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from agentic_data_scientist.autogen_agents.agent_wrappers import (
    AutoGenDataExplorerAgentConfig,
    AutoGenFeatureEngineerAgentConfig,
    AutoGenModelBuilderAgentConfig,
    AutoGenDataExplorerAgent,
    AutoGenFeatureEngineerAgent,
    AutoGenModelBuilderAgent
)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set it in your .env file or as an environment variable.")
    sys.exit(1)

def load_sample_data():
    """Load a sample dataset for demonstration."""
    # Load the breast cancer dataset
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['target'] = cancer.target
    
    print(f"Loaded breast cancer dataset with shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns[:5])}...")
    
    return df

def run_explorer_agent(df):
    """Run the data explorer agent."""
    print("\n=== Data Explorer Agent ===")
    
    # Initialize the agent
    explorer_config = AutoGenDataExplorerAgentConfig(
        name="DataExplorerAgent",
        description="Agent for exploring and analyzing data",
        api_key=GOOGLE_API_KEY,
        model="gemini-2.0-flash",
        code_execution=True
    )
    
    explorer_agent = AutoGenDataExplorerAgent(explorer_config)
    
    # Perform a simple data summary
    print("\nGenerating data summary...")
    result = explorer_agent.summarize_data(df)
    
    print(f"Summary result: {result['success']}")
    print(f"Response (first 200 chars): {result['response'][:200]}...")
    
    return explorer_agent

def run_feature_engineer_agent(df):
    """Run the feature engineer agent."""
    print("\n=== Feature Engineer Agent ===")
    
    # Initialize the agent
    engineer_config = AutoGenFeatureEngineerAgentConfig(
        name="FeatureEngineerAgent",
        description="Agent for feature engineering",
        api_key=GOOGLE_API_KEY,
        model="gemini-2.0-flash",
        code_execution=True
    )
    
    engineer_agent = AutoGenFeatureEngineerAgent(engineer_config)
    
    # Perform feature engineering
    print("\nEngineering features...")
    result = engineer_agent.engineer_features(df, "target")
    
    print(f"Feature engineering result: {result['success']}")
    print(f"Response (first 200 chars): {result['response'][:200]}...")
    
    # Get engineered data if available
    engineered_data = result.get("engineered_data")
    if engineered_data is not None:
        print(f"Engineered data shape: {engineered_data.shape}")
    
    return engineer_agent

def run_model_builder_agent(df):
    """Run the model builder agent."""
    print("\n=== Model Builder Agent ===")
    
    # Initialize the agent
    model_config = AutoGenModelBuilderAgentConfig(
        name="ModelBuilderAgent",
        description="Agent for building models",
        api_key=GOOGLE_API_KEY,
        model="gemini-2.0-flash",
        code_execution=True
    )
    
    model_agent = AutoGenModelBuilderAgent(model_config)
    
    # Build a model
    print("\nBuilding a model...")
    result = model_agent.build_model(df, "target")
    
    print(f"Model building result: {result['success']}")
    print(f"Response (first 200 chars): {result['response'][:200]}...")
    
    # Get model results if available
    model_results = result.get("model_results", {})
    if model_results:
        print(f"Model results: {list(model_results.keys())}")
    
    return model_agent

def main():
    """Main function to run the example."""
    print("=== AutoGen-based Agentic Data Scientist Example ===")
    
    # Load sample data
    df = load_sample_data()
    
    # Run the agents
    explorer_agent = run_explorer_agent(df)
    engineer_agent = run_feature_engineer_agent(df)
    model_agent = run_model_builder_agent(df)
    
    print("\n=== Example Complete ===")

if __name__ == "__main__":
    main() 