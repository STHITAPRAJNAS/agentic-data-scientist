"""Data explorer agent for the Agentic Data Scientist."""
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from agentic_data_scientist.agents.base_agent import BaseAgent, BaseAgentConfig

class DataExplorerAgentConfig(BaseAgentConfig):
    """Configuration for the data explorer agent."""
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

class DataExplorerAgent(BaseAgent):
    """Agent for exploring and analyzing data."""
    
    def __init__(self, config: DataExplorerAgentConfig):
        """Initialize the data explorer agent.
        
        Args:
            config: Configuration for the agent.
        """
        super().__init__(config)
    
    def summarize_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Summarize a dataset.
        
        Args:
            data: DataFrame to summarize.
            
        Returns:
            Summary of the dataset.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Run the summarization query
        result = self.run(
            "Please provide a comprehensive summary of this dataset, including its structure, "
            "data types, basic statistics, and any notable characteristics you observe. "
            "Include visualizations where appropriate."
        )
        
        return result
    
    def analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in a dataset.
        
        Args:
            data: DataFrame to analyze.
            
        Returns:
            Analysis of missing values.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Run the missing values analysis query
        result = self.run(
            "Please analyze the missing values in this dataset. Identify columns with missing values, "
            "calculate the percentage of missing values for each column, and suggest appropriate "
            "strategies for handling them. Include visualizations to illustrate the patterns of missing data."
        )
        
        return result
    
    def analyze_distributions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of variables in a dataset.
        
        Args:
            data: DataFrame to analyze.
            
        Returns:
            Analysis of distributions.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Run the distributions analysis query
        result = self.run(
            "Please analyze the distributions of variables in this dataset. For numeric variables, "
            "calculate statistics like mean, median, standard deviation, etc., and create histograms "
            "or density plots. For categorical variables, calculate frequencies and create bar charts. "
            "Identify any skewness or notable patterns in the distributions."
        )
        
        return result
    
    def analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between variables in a dataset.
        
        Args:
            data: DataFrame to analyze.
            
        Returns:
            Analysis of correlations.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Run the correlations analysis query
        result = self.run(
            "Please analyze the correlations between variables in this dataset. Calculate correlation "
            "coefficients, create a correlation matrix heatmap, and identify the strongest relationships. "
            "Discuss any interesting or unexpected correlations you find."
        )
        
        return result
    
    def analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in a dataset.
        
        Args:
            data: DataFrame to analyze.
            
        Returns:
            Analysis of outliers.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Run the outliers analysis query
        result = self.run(
            "Please analyze outliers in this dataset. Identify numeric columns with outliers using "
            "appropriate methods (e.g., IQR, Z-scores), visualize them using box plots or scatter plots, "
            "and suggest strategies for handling them."
        )
        
        return result
    
    def generate_full_eda_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a full EDA report for a dataset.
        
        Args:
            data: DataFrame to analyze.
            
        Returns:
            Full EDA report.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Run the full EDA report query
        result = self.run(
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
        
        return result
    
    def custom_analysis(self, data: pd.DataFrame, query: str) -> Dict[str, Any]:
        """Perform a custom analysis on a dataset.
        
        Args:
            data: DataFrame to analyze.
            query: Custom analysis query.
            
        Returns:
            Results of the custom analysis.
        """
        # Add the data to the code generator
        self.code_generator.locals_dict["df"] = data
        
        # Run the custom analysis query
        result = self.run(query)
        
        return result 