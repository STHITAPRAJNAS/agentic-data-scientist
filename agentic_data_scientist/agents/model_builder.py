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
    
    def build_model(self, 
                    data: pd.DataFrame, 
                    target_variable: str,
                    model_type: Optional[str] = None,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Dict[str, Any]:
        """Build a model for a dataset.
        
        Args:
            data: DataFrame to build a model for.
            target_variable: Target variable name.
            model_type: Optional model type ('classification', 'regression', 'clustering', etc.).
            test_size: Size of the test set (default: 0.2).
            random_state: Random state for reproducibility (default: 42).
            
        Returns:
            Results with the trained model.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        self.code_generator.locals_dict["target_variable"] = target_variable
        self.code_generator.locals_dict["test_size"] = test_size
        self.code_generator.locals_dict["random_state"] = random_state
        
        # Construct the query
        query = f"Please build a machine learning model to predict '{target_variable}'. "
        
        if model_type:
            query += f"This is a {model_type} task. "
        
        query += (
            f"Split the data into training and testing sets (test_size={test_size}, random_state={random_state}). "
            "Select an appropriate model, train it, and evaluate its performance using appropriate metrics. "
            "Provide visualizations of the model's performance and feature importance if applicable."
        )
        
        # Run the model building query
        result = self.run(query)
        
        return result
    
    def compare_models(self, 
                       data: pd.DataFrame, 
                       target_variable: str,
                       model_type: Optional[str] = None,
                       test_size: float = 0.2,
                       random_state: int = 42) -> Dict[str, Any]:
        """Compare multiple models for a dataset.
        
        Args:
            data: DataFrame to build models for.
            target_variable: Target variable name.
            model_type: Optional model type ('classification', 'regression', 'clustering', etc.).
            test_size: Size of the test set (default: 0.2).
            random_state: Random state for reproducibility (default: 42).
            
        Returns:
            Results with the compared models.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        self.code_generator.locals_dict["target_variable"] = target_variable
        self.code_generator.locals_dict["test_size"] = test_size
        self.code_generator.locals_dict["random_state"] = random_state
        
        # Construct the query
        query = f"Please compare multiple machine learning models to predict '{target_variable}'. "
        
        if model_type:
            query += f"This is a {model_type} task. "
        
        query += (
            f"Split the data into training and testing sets (test_size={test_size}, random_state={random_state}). "
            "Try at least 3 different models (e.g., Random Forest, XGBoost, LightGBM, etc.), train them, and compare "
            "their performance using appropriate metrics. Create visualizations to compare the models and identify "
            "the best performing one. Explain the strengths and weaknesses of each model."
        )
        
        # Run the model comparison query
        result = self.run(query)
        
        return result
    
    def tune_hyperparameters(self, 
                            data: pd.DataFrame, 
                            target_variable: str,
                            model_type: str,
                            test_size: float = 0.2,
                            random_state: int = 42) -> Dict[str, Any]:
        """Tune hyperparameters for a model.
        
        Args:
            data: DataFrame to tune hyperparameters for.
            target_variable: Target variable name.
            model_type: Model type (e.g., 'RandomForest', 'XGBoost', 'LightGBM', etc.).
            test_size: Size of the test set (default: 0.2).
            random_state: Random state for reproducibility (default: 42).
            
        Returns:
            Results with the tuned model.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        self.code_generator.locals_dict["target_variable"] = target_variable
        self.code_generator.locals_dict["model_type"] = model_type
        self.code_generator.locals_dict["test_size"] = test_size
        self.code_generator.locals_dict["random_state"] = random_state
        
        # Run the hyperparameter tuning query
        result = self.run(
            f"Please tune the hyperparameters for a {model_type} model to predict '{target_variable}'. "
            f"Split the data into training and testing sets (test_size={test_size}, random_state={random_state}). "
            "Use an appropriate hyperparameter tuning method (e.g., GridSearchCV, RandomizedSearchCV, Bayesian optimization, etc.). "
            "Evaluate the tuned model's performance and compare it with a baseline model. "
            "Explain the impact of different hyperparameters on the model's performance."
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
    
    def cross_validation(self, 
                        data: pd.DataFrame, 
                        target_variable: str,
                        model_type: str,
                        n_splits: int = 5,
                        random_state: int = 42) -> Dict[str, Any]:
        """Perform cross-validation for a model.
        
        Args:
            data: DataFrame to perform cross-validation on.
            target_variable: Target variable name.
            model_type: Model type (e.g., 'RandomForest', 'XGBoost', 'LightGBM', etc.).
            n_splits: Number of cross-validation splits (default: 5).
            random_state: Random state for reproducibility (default: 42).
            
        Returns:
            Results with the cross-validation.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        self.code_generator.locals_dict["target_variable"] = target_variable
        self.code_generator.locals_dict["model_type"] = model_type
        self.code_generator.locals_dict["n_splits"] = n_splits
        self.code_generator.locals_dict["random_state"] = random_state
        
        # Run the cross-validation query
        result = self.run(
            f"Please perform {n_splits}-fold cross-validation for a {model_type} model to predict '{target_variable}'. "
            "Calculate appropriate evaluation metrics for each fold and the overall model performance. "
            "Visualize the distribution of the evaluation metrics across folds and discuss the model's stability. "
            "Provide insights into the model's performance and any potential issues with the data."
        )
        
        return result
    
    def custom_modeling(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform custom modeling tasks.
        
        Args:
            data: DataFrame to perform modeling on.
            query: Custom modeling query.
            
        Returns:
            Results with the custom modeling.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Run the custom modeling query
        result = self.run(query)
        
        return result 