"""AutoGen implementation of the Agentic Data Scientist."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Dict, List, Optional, Union, Callable
import autogen
from autogen.coding.base import CodeBlock, CodeExecutor, CodeExecutorRegistry
from pydantic import BaseModel, Field
from agentic_data_scientist.utils.code_generator import CodeGenerator

class SafeCodeExecutor(CodeExecutor):
    """Custom executor for safely executing code."""
    
    def __init__(self, code_generator: CodeGenerator):
        """Initialize the safe code executor.
        
        Args:
            code_generator: CodeGenerator instance for safe code execution.
        """
        self.code_generator = code_generator
    
    def execute_code(self, code: str, code_execution_config=None) -> str:
        """Execute code using the code generator.
        
        Args:
            code: The code to execute.
            code_execution_config: Ignored configuration.
            
        Returns:
            The output of the code execution.
        """
        # Add necessary imports if they're not present
        if "import" not in code:
            code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
""" + code
        
        result = self.code_generator.execute_code(code)
        
        if not result["success"]:
            return f"Error executing code: {result['error']}"
        
        output = result["output"] if result["output"] else "Code executed successfully with no output."
        
        # Check if there are matplotlib figures
        if "plt" in self.code_generator.locals_dict:
            plt = self.code_generator.locals_dict["plt"]
            if plt.get_fignums():
                # Save the figure to a file
                fig_path = "temp_code/figure.png"
                os.makedirs("temp_code", exist_ok=True)
                plt.savefig(fig_path)
                plt.close('all')
                output += f"\n[Figure saved to {fig_path}]"
        
        # Check if there are plotly figures
        for name, value in self.code_generator.locals_dict.items():
            if isinstance(value, (go.Figure, px.Figure)):
                # Save the figure to a file
                os.makedirs("temp_code", exist_ok=True)
                fig_path = f"temp_code/{name}.html"
                value.write_html(fig_path)
                output += f"\n[Plotly figure saved to {fig_path}]"
        
        return output

class AutoGenAgentConfig(BaseModel):
    """Configuration for the AutoGen-based agent."""
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    api_key: str = Field(..., description="API key for the LLM")
    model: str = Field(default="gemini-2.0-flash", description="Model to use")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    system_message: str = Field(..., description="System message for the agent")
    code_execution: bool = Field(default=True, description="Whether to allow code execution")

class DataScienceTeam:
    """AutoGen-based implementation of the data science team."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data science team.
        
        Args:
            config: Configuration for the data science team.
        """
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "gemini-2.0-flash")
        self.temperature = config.get("temperature", 0.7)
        
        # Set up environment
        os.environ["GOOGLE_API_KEY"] = self.api_key
        
        # Set up the code generator for safe code execution
        self.code_generator = CodeGenerator(locals_dict={}, security_check=True)
        
        # Set up the code executor
        self.code_executor = SafeCodeExecutor(self.code_generator)
        
        # Register the executor with AutoGen
        CodeExecutorRegistry.register_executor("safe_executor", lambda *args, **kwargs: self.code_executor)
        
        # Initialize the AutoGen agents
        self.initialize_agents()
        
        # Set up chat history
        self.chat_history = []
    
    def initialize_agents(self):
        """Initialize the AutoGen agents."""
        # Configure the LLM
        config_list = [
            {
                "model": self.model,
                "api_key": self.api_key,
                "api_type": "google",
            }
        ]
        
        # Create the user proxy agent
        self.user_proxy = autogen.UserProxyAgent(
            name="DataScientistUser",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=5,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={"executor": "safe_executor"}
        )
        
        # Create the data explorer agent
        self.data_explorer = autogen.AssistantAgent(
            name="DataExplorer",
            system_message="""You are an expert data scientist specializing in exploratory data analysis (EDA).
            Your task is to help analyze datasets to uncover insights, patterns, and potential issues.
            
            You have access to Python code execution. You should leverage pandas, matplotlib, seaborn, and plotly for analysis and visualization.
            
            Here are your primary responsibilities:
            1. Understand and summarize datasets (size, structure, data types, etc.)
            2. Identify and handle missing values appropriately
            3. Detect and address outliers
            4. Analyze distributions of variables
            5. Identify correlations between variables
            6. Generate informative visualizations
            7. Provide clear explanations of your findings
            
            Always be thorough in your analysis and communicate your findings clearly.
            When generating code, ensure it is well-documented and follows best practices.
            """,
            llm_config={"config_list": config_list}
        )
        
        # Create the feature engineer agent
        self.feature_engineer = autogen.AssistantAgent(
            name="FeatureEngineer",
            system_message="""You are an expert data scientist specializing in feature engineering and selection.
            Your task is to help create, transform, and select features for machine learning models.
            
            You have access to Python code execution. You should leverage pandas, numpy, scikit-learn, and category_encoders for feature engineering.
            
            Here are your primary responsibilities:
            1. Analyze and understand the dataset and the target variable
            2. Create new features through transformations, interactions, and aggregations
            3. Handle categorical features through appropriate encoding methods
            4. Normalize or standardize numeric features when necessary
            5. Perform dimensionality reduction when appropriate
            6. Select important features based on statistical tests and model-based methods
            7. Handle time-series data features when applicable
            
            Always explain your reasoning behind each feature engineering decision.
            When generating code, ensure it is well-documented and follows best practices.
            """,
            llm_config={"config_list": config_list}
        )
        
        # Create the model builder agent
        self.model_builder = autogen.AssistantAgent(
            name="ModelBuilder",
            system_message="""You are an expert data scientist specializing in machine learning model building, training, and evaluation.
            Your task is to help select, configure, train, and evaluate machine learning models.
            
            You have access to Python code execution. You should leverage scikit-learn, XGBoost, LightGBM, and other ML libraries.
            
            Here are your primary responsibilities:
            1. Select appropriate models based on the problem type and data characteristics
            2. Configure and tune model hyperparameters
            3. Train models on the provided data
            4. Evaluate models using appropriate metrics
            5. Compare different models and select the best one
            6. Perform cross-validation to assess model generalization
            7. Provide clear explanations of model performance and interpretation
            
            Always be thorough in your model evaluation and communicate your findings clearly.
            When generating code, ensure it is well-documented and follows best practices.
            """,
            llm_config={"config_list": config_list}
        )
    
    def explore_data(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Explore data using the data explorer agent.
        
        Args:
            data: DataFrame to explore.
            query: Query for data exploration.
            
        Returns:
            Results of the data exploration.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Reset the chat
        self.data_explorer.reset()
        self.user_proxy.reset()
        
        # Create the data context message
        data_context = f"""
        Dataset Information:
        - Shape: {data.shape}
        - Columns: {', '.join(data.columns)}
        - Data Types:
        {data.dtypes.to_string()}
        - Sample Data:
        {data.head().to_string()}
        
        {query}
        
        Use the provided dataframe (df) directly. DO NOT create or generate any synthetic data.
        """
        
        # Initiate the chat
        chat_result = self.user_proxy.initiate_chat(
            self.data_explorer,
            message=data_context
        )
        
        # Process the chat result
        final_response = chat_result.chat_history[-1]['content']
        self.chat_history.extend(chat_result.chat_history)
        
        return {
            "success": True,
            "response": final_response,
            "locals": self.code_generator.get_locals(),
            "chat_history": self.chat_history
        }
    
    def engineer_features(self, data: pd.DataFrame, target_variable: Optional[str] = None) -> Dict[str, Any]:
        """Engineer features using the feature engineer agent.
        
        Args:
            data: DataFrame to engineer features for.
            target_variable: Optional target variable name.
            
        Returns:
            Results of the feature engineering.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Reset the chat
        self.feature_engineer.reset()
        self.user_proxy.reset()
        
        # Create the data context message
        data_context = f"""
        Dataset Information:
        - Shape: {data.shape}
        - Columns: {', '.join(data.columns)}
        - Data Types:
        {data.dtypes.to_string()}
        - Sample Data:
        {data.head().to_string()}
        
        Please engineer appropriate features for this dataset to improve machine learning model performance.
        Consider creating new features through transformations, interactions, and aggregations, handling categorical
        features, normalizing or standardizing numeric features, and selecting important features.
        """
        
        if target_variable:
            data_context += f"\nThe target variable is '{target_variable}'."
            self.code_generator.locals_dict["target_variable"] = target_variable
        
        # Initiate the chat
        chat_result = self.user_proxy.initiate_chat(
            self.feature_engineer,
            message=data_context
        )
        
        # Process the chat result
        final_response = chat_result.chat_history[-1]['content']
        self.chat_history.extend(chat_result.chat_history)
        
        # Get the engineered dataframe if available
        engineered_data = None
        if "engineered_df" in self.code_generator.locals_dict:
            engineered_data = self.code_generator.locals_dict["engineered_df"]
        elif "df_processed" in self.code_generator.locals_dict:
            engineered_data = self.code_generator.locals_dict["df_processed"]
        elif "df_engineered" in self.code_generator.locals_dict:
            engineered_data = self.code_generator.locals_dict["df_engineered"]
        elif "X" in self.code_generator.locals_dict:
            engineered_data = self.code_generator.locals_dict["X"]
        
        return {
            "success": True,
            "response": final_response,
            "locals": self.code_generator.get_locals(),
            "chat_history": self.chat_history,
            "engineered_data": engineered_data
        }
    
    def build_model(self, data: pd.DataFrame, target_variable: str) -> Dict[str, Any]:
        """Build a model using the model builder agent.
        
        Args:
            data: DataFrame to build a model for.
            target_variable: Target variable name.
            
        Returns:
            Results of the model building.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        self.code_generator.locals_dict["target_variable"] = target_variable
        
        # Reset the chat
        self.model_builder.reset()
        self.user_proxy.reset()
        
        # Create the data context message
        data_context = f"""
        Dataset Information:
        - Shape: {data.shape}
        - Columns: {', '.join(data.columns)}
        - Data Types:
        {data.dtypes.to_string()}
        - Sample Data:
        {data.head().to_string()}
        - Target Variable: {target_variable}
        
        Please build a machine learning model to predict the target variable '{target_variable}'.
        Select appropriate models based on the problem type and data characteristics,
        configure and tune model hyperparameters, train models on the provided data,
        evaluate models using appropriate metrics, and provide clear explanations of model performance.
        """
        
        # Initiate the chat
        chat_result = self.user_proxy.initiate_chat(
            self.model_builder,
            message=data_context
        )
        
        # Process the chat result
        final_response = chat_result.chat_history[-1]['content']
        self.chat_history.extend(chat_result.chat_history)
        
        # Get model results if available
        model_results = {}
        for key, value in self.code_generator.locals_dict.items():
            if key.endswith("_model") or key == "model" or key == "best_model":
                model_results[key] = value
        
        return {
            "success": True,
            "response": final_response,
            "locals": self.code_generator.get_locals(),
            "chat_history": self.chat_history,
            "model_results": model_results
        }
    
    def custom_analysis(self, data: pd.DataFrame, query: str, agent_type: str = "explorer") -> Dict[str, Any]:
        """Perform a custom analysis using the specified agent.
        
        Args:
            data: DataFrame to analyze.
            query: Custom analysis query.
            agent_type: Type of agent to use (explorer, engineer, or model).
            
        Returns:
            Results of the custom analysis.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Select the appropriate agent
        if agent_type == "explorer":
            agent = self.data_explorer
        elif agent_type == "engineer":
            agent = self.feature_engineer
        elif agent_type == "model":
            agent = self.model_builder
        else:
            return {
                "success": False,
                "response": f"Invalid agent type: {agent_type}. Must be one of: explorer, engineer, model.",
                "chat_history": self.chat_history
            }
        
        # Reset the chat
        agent.reset()
        self.user_proxy.reset()
        
        # Create the data context message
        data_context = f"""
        Dataset Information:
        - Shape: {data.shape}
        - Columns: {', '.join(data.columns)}
        - Data Types:
        {data.dtypes.to_string()}
        - Sample Data:
        {data.head().to_string()}
        
        {query}
        """
        
        # Initiate the chat
        chat_result = self.user_proxy.initiate_chat(
            agent,
            message=data_context
        )
        
        # Process the chat result
        final_response = chat_result.chat_history[-1]['content']
        self.chat_history.extend(chat_result.chat_history)
        
        return {
            "success": True,
            "response": final_response,
            "locals": self.code_generator.get_locals(),
            "chat_history": self.chat_history
        }
    
    def get_locals(self) -> Dict[str, Any]:
        """Get the local variables.
        
        Returns:
            Dictionary of local variables.
        """
        return self.code_generator.get_locals() 