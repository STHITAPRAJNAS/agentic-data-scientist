"""Code generator utility for the Agentic Data Scientist."""
import sys
import io
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

class CodeGenerator:
    """Code generator for safely executing LLM-generated code."""
    
    def __init__(self, locals_dict: Dict[str, Any], security_check: bool = True):
        """Initialize the code generator.
        
        Args:
            locals_dict: Dictionary of local variables to use in code execution.
            security_check: Whether to perform security checks on the code.
        """
        self.locals_dict = locals_dict
        self.security_check = security_check
        self.last_output = ""
        self.last_result = None
    
    def execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code safely.
        
        Args:
            code: Code to execute.
            
        Returns:
            Dictionary containing execution results.
        """
        if self.security_check:
            if not self._check_code_security(code):
                return {
                    "success": False,
                    "error": "Code failed security check",
                    "output": ""
                }
        
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()
        sys.stdout = stdout
        sys.stderr = stderr
        
        try:
            # Execute the code
            exec(code, self.locals_dict)
            
            # Get the output
            output = stdout.getvalue()
            error = stderr.getvalue()
            
            # Store the output
            self.last_output = output
            self.last_result = {
                "success": True,
                "output": output,
                "error": error
            }
            
            return self.last_result
            
        except Exception as e:
            error_msg = f"{str(e)}\n{stderr.getvalue()}"
            self.last_output = ""
            self.last_result = {
                "success": False,
                "output": "",
                "error": error_msg
            }
            return self.last_result
            
        finally:
            # Restore stdout and stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
    
    def _check_code_security(self, code: str) -> bool:
        """Check if code is safe to execute.
        
        Args:
            code: Code to check.
            
        Returns:
            Whether the code is safe.
        """
        # List of allowed modules
        allowed_modules = {
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly',
            'sklearn', 'scipy', 'statsmodels', 'yellowbrick'
        }
        
        # List of forbidden operations
        forbidden_ops = {
            'os.', 'subprocess.', 'sys.', 'shutil.', 'socket.',
            'multiprocessing.', 'threading.', 'ctypes.', 'builtins.'
        }
        
        # Check for forbidden operations
        for op in forbidden_ops:
            if op in code:
                return False
        
        # Check for import statements
        import_lines = [line for line in code.split('\n') if line.strip().startswith('import ') or line.strip().startswith('from ')]
        for line in import_lines:
            module = line.split()[1].split('.')[0]
            if module not in allowed_modules:
                return False
        
        return True
    
    def get_locals(self) -> Dict[str, Any]:
        """Get the local variables.
        
        Returns:
            Dictionary of local variables.
        """
        return self.locals_dict 