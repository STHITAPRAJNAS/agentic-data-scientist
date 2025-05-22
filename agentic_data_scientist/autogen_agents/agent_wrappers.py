"""Agent wrappers for AutoGen-based implementation."""
import os
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from agentic_data_scientist.autogen_agents.autogen_framework import DataScienceTeam, AutoGenAgentConfig

class AutoGenDataExplorerAgentConfig(AutoGenAgentConfig):
    """Configuration for the AutoGen-based data explorer agent."""
    def __init__(self, **data):
        """Initialize with default values for data explorer agent."""
        if "name" not in data:
            data["name"] = "DataExplorerAgent"
        if "description" not in data:
            data["description"] = "Agent for exploring and analyzing data"
        if "system_message" not in data:
            data["system_message"] = """
            You are an expert data scientist specializing in exploratory data analysis (EDA).
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
            """
        super().__init__(**data)

class AutoGenFeatureEngineerAgentConfig(AutoGenAgentConfig):
    """Configuration for the AutoGen-based feature engineer agent."""
    def __init__(self, **data):
        """Initialize with default values for feature engineer agent."""
        if "name" not in data:
            data["name"] = "FeatureEngineerAgent"
        if "description" not in data:
            data["description"] = "Agent for feature engineering and selection"
        if "system_message" not in data:
            data["system_message"] = """
            You are an expert data scientist specializing in feature engineering and selection.
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
            """
        super().__init__(**data)

class AutoGenModelBuilderAgentConfig(AutoGenAgentConfig):
    """Configuration for the AutoGen-based model builder agent."""
    def __init__(self, **data):
        """Initialize with default values for model builder agent."""
        if "name" not in data:
            data["name"] = "ModelBuilderAgent"
        if "description" not in data:
            data["description"] = "Agent for building models"
        if "system_message" not in data:
            data["system_message"] = """
            You are an expert data scientist specializing in machine learning model building, training, and evaluation.
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
            """
        super().__init__(**data)

# Singleton instance of DataScienceTeam
_data_science_team = None

def get_data_science_team(config: Dict[str, Any] = None) -> DataScienceTeam:
    """Get or create a DataScienceTeam singleton.
    
    Args:
        config: Configuration for the data science team.
        
    Returns:
        The DataScienceTeam singleton.
    """
    global _data_science_team
    if _data_science_team is None and config is not None:
        _data_science_team = DataScienceTeam(config)
    return _data_science_team

class AutoGenDataExplorerAgent:
    """AutoGen-based data explorer agent wrapper."""
    
    def __init__(self, config: AutoGenDataExplorerAgentConfig):
        """Initialize the AutoGen-based data explorer agent.
        
        Args:
            config: Configuration for the agent.
        """
        self.config = config
        self.name = config.name
        self.description = config.description
        
        # Create the data science team
        self.data_science_team = get_data_science_team({
            "api_key": config.api_key,
            "model": config.model,
            "temperature": config.temperature,
        })
        
        # Set up the chat history
        self.chat_history = []
    
    def summarize_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize a dataset.
        
        Args:
            data: DataFrame to summarize.
            
        Returns:
            Summary of the dataset.
        """
        result = self.data_science_team.explore_data(
            data,
            "Please provide a comprehensive summary of this dataset, including its structure, "
            "data types, basic statistics, and any notable characteristics you observe. "
            "Include visualizations where appropriate."
        )
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in a dataset.
        
        Args:
            data: DataFrame to analyze.
            
        Returns:
            Analysis of missing values.
        """
        result = self.data_science_team.explore_data(
            data,
            "Please analyze the missing values in this dataset. Identify columns with missing values, "
            "calculate the percentage of missing values for each column, and suggest appropriate "
            "strategies for handling them. Include visualizations to illustrate the patterns of missing data."
        )
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of variables in a dataset.
        
        Args:
            data: DataFrame to analyze.
            
        Returns:
            Analysis of distributions.
        """
        result = self.data_science_team.explore_data(
            data,
            "Please analyze the distributions of variables in this dataset. For numeric variables, "
            "calculate statistics like mean, median, standard deviation, etc., and create histograms "
            "or density plots. For categorical variables, calculate frequencies and create bar charts. "
            "Identify any skewness or notable patterns in the distributions."
        )
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between variables in a dataset.
        
        Args:
            data: DataFrame to analyze.
            
        Returns:
            Analysis of correlations.
        """
        result = self.data_science_team.explore_data(
            data,
            "Please analyze the correlations between variables in this dataset. Calculate correlation "
            "coefficients, create a correlation matrix heatmap, and identify the strongest relationships. "
            "Discuss any interesting or unexpected correlations you find."
        )
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in a dataset.
        
        Args:
            data: DataFrame to analyze.
            
        Returns:
            Analysis of outliers.
        """
        result = self.data_science_team.explore_data(
            data,
            "Please analyze outliers in this dataset. Identify numeric columns with outliers using "
            "appropriate methods (e.g., IQR, Z-scores), visualize them using box plots or scatter plots, "
            "and suggest strategies for handling them."
        )
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def generate_full_eda_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a full EDA report for a dataset.
        
        Args:
            data: DataFrame to analyze.
            
        Returns:
            Full EDA report.
        """
        result = self.data_science_team.explore_data(
            data,
            "Please perform a comprehensive exploratory data analysis (EDA) on this dataset. "
            "Your analysis should include:\n"
            "1. Data structure and summary statistics\n"
            "2. Analysis of data types and any necessary conversions\n"
            "3. Missing data analysis and visualization\n"
            "4. Distribution analysis for key variables (with appropriate visualizations)\n"
            "5. Correlation analysis between variables\n"
            "6. Outlier detection and visualization\n"
            "7. Key insights and observations\n\n"
            "Use high-quality visualizations throughout and provide clear explanations of your findings."
        )
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def custom_analysis(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform a custom analysis on a dataset.
        
        Args:
            data: DataFrame to analyze.
            query: Custom analysis query.
            
        Returns:
            Results of the custom analysis.
        """
        result = self.data_science_team.explore_data(data, query)
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def run(self, query: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run a custom query on a dataset.
        
        Args:
            query: Query to run.
            data: Optional DataFrame to use.
            
        Returns:
            Results of the query.
        """
        if data is None:
            return {
                "success": False,
                "response": "No dataset provided. Please load a dataset first.",
                "chat_history": self.chat_history
            }
        
        result = self.data_science_team.explore_data(data, query)
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def reset(self):
        """Reset the agent."""
        # Reset the chat history
        self.chat_history = []

class AutoGenFeatureEngineerAgent:
    """AutoGen-based feature engineer agent wrapper."""
    
    def __init__(self, config: AutoGenFeatureEngineerAgentConfig):
        """Initialize the AutoGen-based feature engineer agent.
        
        Args:
            config: Configuration for the agent.
        """
        self.config = config
        self.name = config.name
        self.description = config.description
        
        # Create the data science team
        self.data_science_team = get_data_science_team({
            "api_key": config.api_key,
            "model": config.model,
            "temperature": config.temperature,
        })
        
        # Set up the chat history
        self.chat_history = []
    
    def engineer_features(self, data: pd.DataFrame, target_variable=None) -> Dict[str, Any]:
        """Engineer features for a dataset.
        
        Args:
            data: DataFrame to engineer features for.
            target_variable: Optional target variable name.
            
        Returns:
            Results with engineered features.
        """
        result = self.data_science_team.engineer_features(data, target_variable)
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def encode_categorical_features(self, data: pd.DataFrame, categorical_columns=None) -> Dict[str, Any]:
        """Encode categorical features.
        
        Args:
            data: DataFrame to encode features for.
            categorical_columns: Optional list of categorical column names.
            
        Returns:
            Results with encoded features.
        """
        query = (
            "Please encode the categorical features in this dataset using appropriate encoding methods "
            "(e.g., one-hot encoding, label encoding, target encoding, etc.). Explain your choice of "
            "encoding method for each feature."
        )
        
        if categorical_columns:
            query += f" The categorical columns are: {categorical_columns}."
        
        result = self.data_science_team.custom_analysis(data, query, "engineer")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def normalize_features(self, data: pd.DataFrame, numeric_columns=None) -> Dict[str, Any]:
        """Normalize or standardize numeric features.
        
        Args:
            data: DataFrame to normalize features for.
            numeric_columns: Optional list of numeric column names.
            
        Returns:
            Results with normalized features.
        """
        query = (
            "Please normalize or standardize the numeric features in this dataset using appropriate methods "
            "(e.g., Min-Max scaling, Z-score standardization, robust scaling, etc.). Explain your choice of "
            "scaling method for each feature."
        )
        
        if numeric_columns:
            query += f" The numeric columns are: {numeric_columns}."
        
        result = self.data_science_team.custom_analysis(data, query, "engineer")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def select_features(self, data: pd.DataFrame, target_variable, n_features=None) -> Dict[str, Any]:
        """Select important features.
        
        Args:
            data: DataFrame to select features from.
            target_variable: Target variable name.
            n_features: Optional number of features to select.
            
        Returns:
            Results with selected features.
        """
        query = (
            f"Please select the most important features for predicting '{target_variable}' using appropriate "
            "feature selection methods (e.g., statistical tests, model-based methods, etc.). Explain your "
            "choice of selection method and why each selected feature is important."
        )
        
        if n_features:
            query += f" Please select approximately {n_features} features."
        
        result = self.data_science_team.custom_analysis(data, query, "engineer")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def reduce_dimensionality(self, data: pd.DataFrame, n_components=None) -> Dict[str, Any]:
        """Reduce dimensionality of the dataset.
        
        Args:
            data: DataFrame to reduce dimensionality for.
            n_components: Optional number of components to reduce to.
            
        Returns:
            Results with dimensionality reduction.
        """
        query = (
            "Please reduce the dimensionality of this dataset using appropriate methods "
            "(e.g., PCA, t-SNE, UMAP, etc.). Explain your choice of dimensionality reduction "
            "method and how to interpret the results."
        )
        
        if n_components:
            query += f" Please reduce to approximately {n_components} components."
        
        result = self.data_science_team.custom_analysis(data, query, "engineer")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def handle_time_series_features(self, data: pd.DataFrame, time_column) -> Dict[str, Any]:
        """Handle time series features.
        
        Args:
            data: DataFrame with time series data.
            time_column: Name of the time column.
            
        Returns:
            Results with time series features.
        """
        query = (
            f"Please create appropriate time series features from the time column '{time_column}'. "
            "Consider extracting time components (e.g., hour, day, month, year, day of week), "
            "creating lag features, rolling window features, and other relevant time series features. "
            "Explain why each created feature might be useful for modeling."
        )
        
        result = self.data_science_team.custom_analysis(data, query, "engineer")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def custom_feature_engineering(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform custom feature engineering.
        
        Args:
            data: DataFrame to engineer features for.
            query: Custom feature engineering query.
            
        Returns:
            Results with custom feature engineering.
        """
        result = self.data_science_team.custom_analysis(data, query, "engineer")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def run(self, query: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run a custom query on a dataset.
        
        Args:
            query: Query to run.
            data: Optional DataFrame to use.
            
        Returns:
            Results of the query.
        """
        if data is None:
            return {
                "success": False,
                "response": "No dataset provided. Please load a dataset first.",
                "chat_history": self.chat_history
            }
        
        result = self.data_science_team.custom_analysis(data, query, "engineer")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def reset(self):
        """Reset the agent."""
        # Reset the chat history
        self.chat_history = []

class AutoGenModelBuilderAgent:
    """AutoGen-based model builder agent wrapper."""
    
    def __init__(self, config: AutoGenModelBuilderAgentConfig):
        """Initialize the AutoGen-based model builder agent.
        
        Args:
            config: Configuration for the agent.
        """
        self.config = config
        self.name = config.name
        self.description = config.description
        
        # Create the data science team
        self.data_science_team = get_data_science_team({
            "api_key": config.api_key,
            "model": config.model,
            "temperature": config.temperature,
        })
        
        # Set up the chat history
        self.chat_history = []
    
    def build_model(self, data: pd.DataFrame, target_variable: str, model_type: str = None) -> Dict[str, Any]:
        """Build a machine learning model.
        
        Args:
            data: DataFrame to build a model for.
            target_variable: Target variable name.
            model_type: Optional type of model to build.
            
        Returns:
            Results of model building.
        """
        if model_type:
            query = f"Please build a {model_type} model to predict the target variable '{target_variable}'."
            result = self.data_science_team.custom_analysis(data, query, "model")
        else:
            result = self.data_science_team.build_model(data, target_variable)
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def evaluate_model(self, data: pd.DataFrame, target_variable: str, model=None) -> Dict[str, Any]:
        """Evaluate a machine learning model.
        
        Args:
            data: DataFrame to evaluate the model on.
            target_variable: Target variable name.
            model: Optional model to evaluate.
            
        Returns:
            Results of model evaluation.
        """
        query = f"Please evaluate the model for predicting '{target_variable}' using appropriate metrics."
        if model:
            # Store the model in the locals dict
            self.data_science_team.code_generator.locals_dict["model"] = model
            query += " Use the provided 'model' for evaluation."
        else:
            query += " Train a model first if needed."
        
        result = self.data_science_team.custom_analysis(data, query, "model")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def explain_model(self, data: pd.DataFrame, target_variable: str, model=None) -> Dict[str, Any]:
        """Explain a machine learning model.
        
        Args:
            data: DataFrame to use for explanation.
            target_variable: Target variable name.
            model: Optional model to explain.
            
        Returns:
            Results of model explanation.
        """
        query = f"Please explain the model for predicting '{target_variable}' using appropriate techniques like feature importance, SHAP values, or partial dependence plots."
        if model:
            # Store the model in the locals dict
            self.data_science_team.code_generator.locals_dict["model"] = model
            query += " Use the provided 'model' for explanation."
        else:
            query += " Train a model first if needed."
        
        result = self.data_science_team.custom_analysis(data, query, "model")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def tune_hyperparameters(self, data: pd.DataFrame, target_variable: str, model_type: str) -> Dict[str, Any]:
        """Tune hyperparameters for a machine learning model.
        
        Args:
            data: DataFrame to use for tuning.
            target_variable: Target variable name.
            model_type: Type of model to tune.
            
        Returns:
            Results of hyperparameter tuning.
        """
        query = f"Please tune the hyperparameters for a {model_type} model to predict '{target_variable}'. Use appropriate techniques like grid search, random search, or Bayesian optimization."
        
        result = self.data_science_team.custom_analysis(data, query, "model")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def cross_validate(self, data: pd.DataFrame, target_variable: str, model_type: str = None, n_splits: int = 5) -> Dict[str, Any]:
        """Perform cross-validation for a machine learning model.
        
        Args:
            data: DataFrame to use for cross-validation.
            target_variable: Target variable name.
            model_type: Optional type of model to validate.
            n_splits: Number of cross-validation splits.
            
        Returns:
            Results of cross-validation.
        """
        query = f"Please perform {n_splits}-fold cross-validation for predicting '{target_variable}'."
        if model_type:
            query += f" Use a {model_type} model."
        
        result = self.data_science_team.custom_analysis(data, query, "model")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def custom_model_building(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform custom model building.
        
        Args:
            data: DataFrame to use for model building.
            query: Custom model building query.
            
        Returns:
            Results of custom model building.
        """
        result = self.data_science_team.custom_analysis(data, query, "model")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def run(self, query: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run a custom query on a dataset.
        
        Args:
            query: Query to run.
            data: Optional DataFrame to use.
            
        Returns:
            Results of the query.
        """
        if data is None:
            return {
                "success": False,
                "response": "No dataset provided. Please load a dataset first.",
                "chat_history": self.chat_history
            }
        
        result = self.data_science_team.custom_analysis(data, query, "model")
        
        self.chat_history = result.get("chat_history", [])
        return result
    
    def reset(self):
        """Reset the agent."""
        # Reset the chat history
        self.chat_history = [] 