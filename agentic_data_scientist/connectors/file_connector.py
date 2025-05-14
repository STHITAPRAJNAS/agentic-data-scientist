"""File connector for the Agentic Data Scientist."""
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import io
import json
import pickle

class FileConnector:
    """File connector for handling file operations."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the file connector.
        
        Args:
            data_dir: Directory for data storage.
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def save_upload(self, file_obj: io.BytesIO, filename: str) -> str:
        """Save an uploaded file.
        
        Args:
            file_obj: File object.
            filename: Name of the file.
            
        Returns:
            Path to the saved file.
        """
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, "wb") as f:
            f.write(file_obj.getvalue())
        return file_path
    
    def read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Read a CSV file.
        
        Args:
            file_path: Path to the CSV file.
            **kwargs: Additional arguments for pd.read_csv.
            
        Returns:
            DataFrame with the CSV data.
        """
        return pd.read_csv(file_path, **kwargs)
    
    def read_excel(self, file_path: str, **kwargs) -> Dict[str, pd.DataFrame]:
        """Read an Excel file.
        
        Args:
            file_path: Path to the Excel file.
            **kwargs: Additional arguments for pd.read_excel.
            
        Returns:
            Dictionary mapping sheet names to DataFrames.
        """
        excel_file = pd.ExcelFile(file_path)
        return {sheet: pd.read_excel(excel_file, sheet_name=sheet, **kwargs) 
                for sheet in excel_file.sheet_names}
    
    def read_json(self, file_path: str) -> Any:
        """Read a JSON file.
        
        Args:
            file_path: Path to the JSON file.
            
        Returns:
            Parsed JSON data.
        """
        with open(file_path, "r") as f:
            return json.load(f)
    
    def read_pickle(self, file_path: str) -> Any:
        """Read a pickle file.
        
        Args:
            file_path: Path to the pickle file.
            
        Returns:
            Unpickled data.
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)
    
    def read_text(self, file_path: str) -> str:
        """Read a text file.
        
        Args:
            file_path: Path to the text file.
            
        Returns:
            Contents of the text file.
        """
        with open(file_path, "r") as f:
            return f.read()
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, format: str = "csv") -> str:
        """Save a DataFrame to a file.
        
        Args:
            df: DataFrame to save.
            filename: Name of the file.
            format: Format to save the DataFrame in ('csv', 'excel', 'json', 'pickle').
            
        Returns:
            Path to the saved file.
        """
        file_path = os.path.join(self.data_dir, filename)
        
        if format == "csv":
            df.to_csv(file_path, index=False)
        elif format == "excel":
            df.to_excel(file_path, index=False)
        elif format == "json":
            df.to_json(file_path, orient="records")
        elif format == "pickle":
            df.to_pickle(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return file_path
    
    def save_model(self, model: Any, filename: str) -> str:
        """Save a model to a file.
        
        Args:
            model: Model to save.
            filename: Name of the file.
            
        Returns:
            Path to the saved file.
        """
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        return file_path
    
    def load_model(self, filename: str) -> Any:
        """Load a model from a file.
        
        Args:
            filename: Name of the file.
            
        Returns:
            Loaded model.
        """
        file_path = os.path.join(self.data_dir, filename)
        return self.read_pickle(file_path)
    
    def list_files(self, extension: Optional[str] = None) -> List[str]:
        """List files in the data directory.
        
        Args:
            extension: Optional file extension to filter by.
            
        Returns:
            List of file names.
        """
        files = os.listdir(self.data_dir)
        if extension is not None:
            files = [f for f in files if f.endswith(extension)]
        return files
    
    def detect_file_type(self, filename: str) -> str:
        """Detect the type of a file based on its extension.
        
        Args:
            filename: Name of the file.
            
        Returns:
            Type of the file ('csv', 'excel', 'json', 'pickle', 'text', 'unknown').
        """
        lower_filename = filename.lower()
        if lower_filename.endswith(".csv"):
            return "csv"
        elif lower_filename.endswith((".xls", ".xlsx", ".xlsm")):
            return "excel"
        elif lower_filename.endswith(".json"):
            return "json"
        elif lower_filename.endswith((".pkl", ".pickle")):
            return "pickle"
        elif lower_filename.endswith((".txt", ".md", ".py", ".sql")):
            return "text"
        else:
            return "unknown"
    
    def infer_dataframe(self, file_path: str) -> Tuple[pd.DataFrame, str, str]:
        """Infer a DataFrame from a file.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            DataFrame, file type, and sheet name (if applicable).
        """
        filename = os.path.basename(file_path)
        file_type = self.detect_file_type(filename)
        sheet_name = None
        
        if file_type == "csv":
            df = self.read_csv(file_path)
        elif file_type == "excel":
            sheets = self.read_excel(file_path)
            sheet_name = list(sheets.keys())[0]
            df = sheets[sheet_name]
        elif file_type == "json":
            data = self.read_json(file_path)
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
        elif file_type == "pickle":
            data = self.read_pickle(file_path)
            if isinstance(data, pd.DataFrame):
                df = data
            else:
                raise ValueError(f"Pickle file does not contain a DataFrame: {filename}")
        else:
            raise ValueError(f"Cannot infer DataFrame from file: {filename}")
        
        return df, file_type, sheet_name 