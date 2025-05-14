"""Feature engineering agent for the Agentic Data Scientist."""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from agentic_data_scientist.agents.base_agent import BaseAgent, BaseAgentConfig

class FeatureEngineerAgentConfig(BaseAgentConfig):
    """Configuration for the feature engineer agent."""
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

class FeatureEngineerAgent(BaseAgent):
    """Agent for feature engineering and selection."""
    
    def __init__(self, config: FeatureEngineerAgentConfig):
        """Initialize the feature engineer agent.
        
        Args:
            config: Configuration for the agent.
        """
        super().__init__(config)
    
    def engineer_features(self, data: pd.DataFrame, target_variable: Optional[str] = None) -> Dict[str, Any]:
        """Engineer features for a dataset.
        
        Args:
            data: DataFrame to engineer features for.
            target_variable: Optional target variable name.
            
        Returns:
            Results with engineered features.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        query = (
            "Please engineer appropriate features for this dataset to improve machine learning model performance. "
            "Consider creating new features through transformations, interactions, and aggregations, handling categorical "
            "features, normalizing or standardizing numeric features, and selecting important features."
        )
        
        if target_variable:
            self.code_generator.locals_dict["target_variable"] = target_variable
            query += f" The target variable is '{target_variable}'."
        
        # Run the feature engineering query
        result = self.run(query)
        
        return result
    
    def encode_categorical_features(self, data: pd.DataFrame, categorical_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Encode categorical features.
        
        Args:
            data: DataFrame to encode features for.
            categorical_columns: Optional list of categorical column names.
            
        Returns:
            Results with encoded features.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        query = (
            "Please encode the categorical features in this dataset using appropriate encoding methods "
            "(e.g., one-hot encoding, label encoding, target encoding, etc.). Explain your choice of "
            "encoding method for each feature."
        )
        
        if categorical_columns:
            self.code_generator.locals_dict["categorical_columns"] = categorical_columns
            query += f" The categorical columns are: {categorical_columns}."
        
        # Run the categorical encoding query
        result = self.run(query)
        
        return result
    
    def normalize_features(self, data: pd.DataFrame, numeric_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Normalize or standardize numeric features.
        
        Args:
            data: DataFrame to normalize features for.
            numeric_columns: Optional list of numeric column names.
            
        Returns:
            Results with normalized features.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        query = (
            "Please normalize or standardize the numeric features in this dataset using appropriate methods "
            "(e.g., Min-Max scaling, Z-score standardization, robust scaling, etc.). Explain your choice of "
            "scaling method for each feature."
        )
        
        if numeric_columns:
            self.code_generator.locals_dict["numeric_columns"] = numeric_columns
            query += f" The numeric columns are: {numeric_columns}."
        
        # Run the normalization query
        result = self.run(query)
        
        return result
    
    def select_features(self, data: pd.DataFrame, target_variable: str, n_features: Optional[int] = None) -> Dict[str, Any]:
        """Select important features.
        
        Args:
            data: DataFrame to select features from.
            target_variable: Target variable name.
            n_features: Optional number of features to select.
            
        Returns:
            Results with selected features.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        self.code_generator.locals_dict["target_variable"] = target_variable
        
        query = (
            f"Please select the most important features for predicting '{target_variable}' using appropriate "
            "feature selection methods (e.g., statistical tests, model-based methods, etc.). Explain your "
            "choice of selection method and why each selected feature is important."
        )
        
        if n_features:
            self.code_generator.locals_dict["n_features"] = n_features
            query += f" Please select approximately {n_features} features."
        
        # Run the feature selection query
        result = self.run(query)
        
        return result
    
    def reduce_dimensionality(self, data: pd.DataFrame, n_components: Optional[int] = None) -> Dict[str, Any]:
        """Reduce dimensionality of the dataset.
        
        Args:
            data: DataFrame to reduce dimensionality for.
            n_components: Optional number of components to reduce to.
            
        Returns:
            Results with dimensionality reduction.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        query = (
            "Please reduce the dimensionality of this dataset using appropriate methods "
            "(e.g., PCA, t-SNE, UMAP, etc.). Explain your choice of dimensionality reduction "
            "method and how to interpret the results."
        )
        
        if n_components:
            self.code_generator.locals_dict["n_components"] = n_components
            query += f" Please reduce to approximately {n_components} components."
        
        # Run the dimensionality reduction query
        result = self.run(query)
        
        return result
    
    def handle_time_series_features(self, data: pd.DataFrame, time_column: str) -> Dict[str, Any]:
        """Handle time series features.
        
        Args:
            data: DataFrame with time series data.
            time_column: Name of the time column.
            
        Returns:
            Results with time series features.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        self.code_generator.locals_dict["time_column"] = time_column
        
        # Run the time series features query
        result = self.run(
            f"Please create appropriate time series features from the time column '{time_column}'. "
            "Consider extracting time components (e.g., hour, day, month, year, day of week), "
            "creating lag features, rolling window features, and other relevant time series features. "
            "Explain why each created feature might be useful for modeling."
        )
        
        return result
    
    def custom_feature_engineering(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform custom feature engineering.
        
        Args:
            data: DataFrame to engineer features for.
            query: Custom feature engineering query.
            
        Returns:
            Results with custom feature engineering.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Run the custom feature engineering query
        result = self.run(query)
        
        return result 