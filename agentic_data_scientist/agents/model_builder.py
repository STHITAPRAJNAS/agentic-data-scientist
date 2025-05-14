"""Model builder agent for the Agentic Data Scientist."""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from agentic_data_scientist.agents.base_agent import BaseAgent, BaseAgentConfig

class ModelBuilderAgentConfig(BaseAgentConfig):
    """Configuration for the model builder agent."""
    def __init__(self, **data):
        """Initialize with default values for model builder agent."""
        if "name" not in data:
            data["name"] = "ModelBuilderAgent"
        if "description" not in data:
            data["description"] = "Agent for building and evaluating machine learning models"
        if "system_message" not in data:
            data["system_message"] = """
            You are an expert data scientist specializing in building and evaluating machine learning models.
            Your task is to help create, train, tune, and evaluate models for various data science tasks.
            
            You have access to Python code execution. You should leverage scikit-learn, XGBoost, LightGBM, and other ML libraries.
            
            Here are your primary responsibilities:
            1. Analyze and understand the dataset and the target variable
            2. Split the data appropriately into training and testing sets
            3. Select appropriate models for the task (classification, regression, clustering, etc.)
            4. Train models with appropriate parameters
            5. Tune hyperparameters to optimize performance
            6. Evaluate models using appropriate metrics
            7. Provide interpretations of model results and feature importance
            8. Make recommendations for model deployment or improvement
            
            Always explain your reasoning behind each modeling decision.
            When generating code, ensure it is well-documented and follows best practices.
            """
        super().__init__(**data)

class ModelBuilderAgent(BaseAgent):
    """Agent for building and evaluating machine learning models."""
    
    def __init__(self, config: ModelBuilderAgentConfig):
        """Initialize the model builder agent.
        
        Args:
            config: Configuration for the agent.
        """
        super().__init__(config)
    
    def build_model(self, df: pd.DataFrame, target_variable, model_type=None, test_size=0.2):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {"success": False, "response": "No dataset provided. Please load a dataset first.", "chat_history": self.chat_history}
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = df
        self.code_generator.locals_dict["target_variable"] = target_variable
        self.code_generator.locals_dict["test_size"] = test_size
        
        # Construct the query
        query = f"Please build a machine learning model to predict '{target_variable}'. "
        
        if model_type:
            query += f"This is a {model_type} task. "
        
        query += (
            f"Split the data into training and testing sets (test_size={test_size}). "
            "Select an appropriate model, train it, and evaluate its performance using appropriate metrics. "
            "Provide visualizations of the model's performance and feature importance if applicable."
        )
        
        # Run the model building query - explicitly pass df as data parameter
        result = self.run(query, data=df)
        
        return result
    
    def compare_models(self, df: pd.DataFrame, target_variable, model_type=None, test_size=0.2):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {"success": False, "response": "No dataset provided. Please load a dataset first.", "chat_history": self.chat_history}
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = df
        self.code_generator.locals_dict["target_variable"] = target_variable
        self.code_generator.locals_dict["test_size"] = test_size
        
        # Construct the query
        query = f"Please compare multiple machine learning models to predict '{target_variable}'. "
        
        if model_type:
            query += f"This is a {model_type} task. "
        
        query += (
            f"Split the data into training and testing sets (test_size={test_size}). "
            "Try at least 3 different models (e.g., Random Forest, XGBoost, LightGBM, etc.), train them, and compare "
            "their performance using appropriate metrics. Create visualizations to compare the models and identify "
            "the best performing one. Explain the strengths and weaknesses of each model."
        )
        
        # Run the model comparison query - explicitly pass df as data parameter
        result = self.run(query, data=df)
        
        return result
    
    def tune_hyperparameters(self, df: pd.DataFrame, target_variable, model_type, test_size=0.2):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {"success": False, "response": "No dataset provided. Please load a dataset first.", "chat_history": self.chat_history}
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = df
        self.code_generator.locals_dict["target_variable"] = target_variable
        self.code_generator.locals_dict["model_type"] = model_type
        self.code_generator.locals_dict["test_size"] = test_size
        
        # Run the hyperparameter tuning query - explicitly pass df as data parameter
        result = self.run(
            f"Please tune the hyperparameters for a {model_type} model to predict '{target_variable}'. "
            f"Split the data into training and testing sets (test_size={test_size}). "
            "Use an appropriate hyperparameter tuning method (e.g., GridSearchCV, RandomizedSearchCV, Bayesian optimization, etc.). "
            "Evaluate the tuned model's performance and compare it with a baseline model. "
            "Explain the impact of different hyperparameters on the model's performance.",
            data=df
        )
        
        return result
    
    def evaluate_model(self, 
                      data: pd.DataFrame, 
                      target_variable: str,
                      model_object_name: str) -> Dict[str, Any]:
        """Evaluate a trained model.
        
        Args:
            data: DataFrame with the test data.
            target_variable: Target variable name.
            model_object_name: Name of the model object in the code generator's locals.
            
        Returns:
            Results with the model evaluation.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["test_df"] = data
        self.code_generator.locals_dict["target_variable"] = target_variable
        
        # Check if the model object exists
        if model_object_name not in self.code_generator.locals_dict:
            return {
                "success": False,
                "error": f"Model object '{model_object_name}' not found in locals"
            }
        
        # Run the model evaluation query
        result = self.run(
            f"Please evaluate the trained model '{model_object_name}' on the provided test data. "
            f"The target variable is '{target_variable}'. Calculate appropriate evaluation metrics "
            "based on the model type (classification, regression, etc.). Create visualizations to "
            "help interpret the model's performance (e.g., ROC curve, confusion matrix, residual plots, etc.). "
            "Provide insights into the model's strengths and weaknesses."
        )
        
        return result
    
    def feature_importance(self, 
                          model_object_name: str,
                          feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze feature importance for a trained model.
        
        Args:
            model_object_name: Name of the model object in the code generator's locals.
            feature_names: Optional list of feature names.
            
        Returns:
            Results with the feature importance analysis.
        """
        # Check if the model object exists
        if model_object_name not in self.code_generator.locals_dict:
            return {
                "success": False,
                "error": f"Model object '{model_object_name}' not found in locals"
            }
        
        # Add feature names to the code generator if provided
        if feature_names:
            self.code_generator.locals_dict["feature_names"] = feature_names
        
        # Run the feature importance query
        result = self.run(
            f"Please analyze the feature importance for the trained model '{model_object_name}'. "
            "Calculate and visualize the importance of each feature in the model. Use appropriate methods "
            "based on the model type (e.g., feature_importances_, coefficients, SHAP values, permutation importance, etc.). "
            "Explain why certain features are more important than others and how this can inform future modeling decisions."
        )
        
        return result
    
    def cross_validation(self, df: pd.DataFrame, target_variable, model_type, n_splits=5):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {"success": False, "response": "No dataset provided. Please load a dataset first.", "chat_history": self.chat_history}
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = df
        self.code_generator.locals_dict["target_variable"] = target_variable
        self.code_generator.locals_dict["model_type"] = model_type
        self.code_generator.locals_dict["n_splits"] = n_splits
        
        # Run the cross-validation query - explicitly pass df as data parameter
        result = self.run(
            f"Please perform {n_splits}-fold cross-validation for a {model_type} model to predict '{target_variable}'. "
            "Calculate appropriate evaluation metrics for each fold and the overall model performance. "
            "Visualize the distribution of the evaluation metrics across folds and discuss the model's stability. "
            "Provide insights into the model's performance and any potential issues with the data.",
            data=df
        )
        
        return result
    
    def custom_modeling(self, df: pd.DataFrame, query: str):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {"success": False, "response": "No dataset provided. Please load a dataset first.", "chat_history": self.chat_history}
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = df
        
        # Run the custom modeling query - explicitly pass df as data parameter
        result = self.run(query, data=df)
        
        return result 