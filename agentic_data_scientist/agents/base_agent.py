"""Base agent class for the Agentic Data Scientist."""
import os
from typing import Any, Dict, List, Optional, Union, Callable
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from pydantic import BaseModel, Field
import pandas as pd
from agentic_data_scientist.utils.code_generator import CodeGenerator
import streamlit as st

class BaseAgentConfig(BaseModel):
    """Configuration for the base agent."""
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent")
    api_key: str = Field(..., description="API key for the LLM")
    model: str = Field(default="gemini-2.0-flash", description="Model to use")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    system_message: str = Field(..., description="System message for the agent")
    memory_size: int = Field(default=5, description="Number of messages to keep in agent memory")
    max_iterations: int = Field(default=10, description="Maximum iterations for agent execution")
    code_execution: bool = Field(default=True, description="Whether to allow code execution")

class BaseAgent:
    """Base agent for the Agentic Data Scientist."""
    
    def __init__(self, config: BaseAgentConfig):
        """Initialize the base agent.
        
        Args:
            config: Configuration for the agent.
        """
        self.config = config
        self.name = config.name
        self.description = config.description
        
        # Set up the API key
        os.environ["GOOGLE_API_KEY"] = config.api_key
        genai.configure(api_key=config.api_key)
        
        # Set up the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=config.model,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            convert_system_message_to_human=True
        )
        
        # Set up the code generator
        self.code_generator = CodeGenerator(locals_dict={}, security_check=True)
        
        # Set up the chat history
        self.chat_history = []
    
    def run(self, query: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Run the agent.
        
        Args:
            query: Query to run.
            data: Optional DataFrame to use.
            
        Returns:
            Results from the agent.
        """
        if data is None:
            return {
                "success": False,
                "response": "No dataset provided. Please load a dataset first.",
                "chat_history": self.chat_history
            }
            
        # Add data to the code generator locals
        self.code_generator.locals_dict["df"] = data
        
        # Create a data context message
        data_context = f"""
        Dataset Information:
        - Shape: {data.shape}
        - Columns: {', '.join(data.columns)}
        - Data Types:
        {data.dtypes.to_string()}
        - Sample Data:
        {data.head().to_string()}
        """
        
        # Create a prompt template
        template = f"""
        {self.config.system_message}
        
        Current Dataset Context:
        {data_context}
        
        User Query: {{query}}
        
        IMPORTANT INSTRUCTIONS:
        1. Use the provided dataset (df) directly. DO NOT create or generate any synthetic data.
        2. All analysis should be performed on the provided dataset.
        3. When writing code, use the variable 'df' which contains the provided dataset.
        4. Include code to visualize the results where appropriate.
        5. Make sure to handle any potential errors in the code.
        
        Please provide a detailed response and generate any necessary code to help with this task.
        """
        
        prompt = PromptTemplate(
            input_variables=["query"],
            template=template
        )
        
        # Create a chain
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        # Run the chain
        response = chain.run(query=query)
        
        # Add to chat history
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Extract code from response and execute if needed
        if self.config.code_execution:
            try:
                # Extract code blocks from response
                code_blocks = self.extract_code_blocks(response)
                for code in code_blocks:
                    # Add necessary imports if not present
                    if "import" not in code:
                        code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
""" + code
                    
                    # Execute the code
                    result = self.code_generator.execute_code(code)
                    
                    # If there are any figures created, display them
                    if "plt" in self.code_generator.locals_dict:
                        plt = self.code_generator.locals_dict["plt"]
                        if plt.get_fignums():
                            st.pyplot(plt.gcf())
                            plt.close('all')
                    
                    # If there are any plotly figures, display them
                    for name, value in self.code_generator.locals_dict.items():
                        if isinstance(value, (go.Figure, px.Figure)):
                            st.plotly_chart(value)
                    
                    if not result["success"]:
                        response += f"\n\nError executing code: {result['error']}"
                    else:
                        # Add the output to the response
                        if result["output"]:
                            response += f"\n\nCode Output:\n{result['output']}"
                        
                        # Add any created DataFrames to the response
                        for name, value in self.code_generator.locals_dict.items():
                            if isinstance(value, pd.DataFrame) and name != "df":
                                response += f"\n\nCreated DataFrame '{name}':\n{value.head().to_string()}"
                
            except Exception as e:
                response += f"\n\nError executing code: {str(e)}"
        
        return {
            "success": True,
            "response": response,
            "locals": self.code_generator.get_locals(),
            "last_output": self.code_generator.last_output,
            "last_result": self.code_generator.last_result,
            "chat_history": self.chat_history
        }
    
    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from text.
        
        Args:
            text: Text to extract code blocks from.
            
        Returns:
            List of code blocks.
        """
        code_blocks = []
        lines = text.split("\n")
        in_code_block = False
        current_block = []
        
        for line in lines:
            if line.strip().startswith("```"):
                if in_code_block:
                    code_blocks.append("\n".join(current_block))
                    current_block = []
                in_code_block = not in_code_block
            elif in_code_block:
                current_block.append(line)
        
        return code_blocks
    
    def reset(self):
        """Reset the agent."""
        # Reset the code generator
        self.code_generator = CodeGenerator(locals_dict={}, security_check=True)
        
        # Reset the chat history
        self.chat_history = []
    
    def get_code_generator(self) -> CodeGenerator:
        """Get the code generator.
        
        Returns:
            The code generator.
        """
        return self.code_generator
    
    def get_locals(self) -> Dict[str, Any]:
        """Get the local variables.
        
        Returns:
            Dictionary of local variables.
        """
        return self.code_generator.get_locals() 