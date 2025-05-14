"""SQLite connector for the Agentic Data Scientist."""
import os
import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, inspect

class SQLiteConnector:
    """SQLite connector for handling database operations."""
    
    def __init__(self, db_path: str):
        """Initialize the SQLite connector.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        self.inspector = inspect(self.engine)
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection to the database.
        
        Returns:
            A connection to the database.
        """
        return sqlite3.connect(self.db_path)
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute a SQL query.
        
        Args:
            query: SQL query to execute.
            params: Parameters for the query.
            
        Returns:
            DataFrame with the query results.
        """
        if params is None:
            params = {}
        return pd.read_sql_query(query, self.engine, params=params)
    
    def execute_raw_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a raw SQL query without returning a DataFrame.
        
        Args:
            query: SQL query to execute.
            params: Parameters for the query.
            
        Returns:
            The result of the query.
        """
        if params is None:
            params = {}
        with self.engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(query), params)
            return result
    
    def list_tables(self) -> List[str]:
        """List all tables in the database.
        
        Returns:
            List of table names.
        """
        return self.inspector.get_table_names()
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get the schema of a table.
        
        Args:
            table_name: Name of the table.
            
        Returns:
            List of dictionaries with column information.
        """
        return self.inspector.get_columns(table_name)
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists.
        
        Args:
            table_name: Name of the table.
            
        Returns:
            True if the table exists, False otherwise.
        """
        return self.inspector.has_table(table_name)
    
    def get_table_data(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Get data from a table.
        
        Args:
            table_name: Name of the table.
            limit: Maximum number of rows to return.
            
        Returns:
            DataFrame with the table data.
        """
        query = f"SELECT * FROM {table_name}"
        if limit is not None:
            query += f" LIMIT {limit}"
        return self.execute_query(query)
    
    def load_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace') -> None:
        """Load a DataFrame into a table.
        
        Args:
            df: DataFrame to load.
            table_name: Name of the table.
            if_exists: What to do if the table exists ('fail', 'replace', or 'append').
        """
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
    
    def create_table_for_csv(self, csv_path: str, table_name: str) -> None:
        """Create a table from a CSV file.
        
        Args:
            csv_path: Path to the CSV file.
            table_name: Name of the table to create.
        """
        df = pd.read_csv(csv_path)
        self.load_dataframe(df, table_name)
    
    def get_table_preview(self, table_name: str, n: int = 5) -> pd.DataFrame:
        """Get a preview of a table.
        
        Args:
            table_name: Name of the table.
            n: Number of rows to preview.
            
        Returns:
            DataFrame with the preview.
        """
        if not self.table_exists(table_name):
            raise ValueError(f"Table {table_name} does not exist.")
        return self.get_table_data(table_name, limit=n)
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database.
        
        Returns:
            Dictionary with database information.
        """
        tables = self.list_tables()
        info = {
            "database_path": self.db_path,
            "tables": []
        }
        
        for table_name in tables:
            schema = self.get_table_schema(table_name)
            row_count = self.execute_query(f"SELECT COUNT(*) as count FROM {table_name}").iloc[0]["count"]
            
            table_info = {
                "name": table_name,
                "columns": [{"name": col["name"], "type": str(col["type"])} for col in schema],
                "row_count": row_count
            }
            info["tables"].append(table_info)
        
        return info
    
    def execute_script(self, script: str) -> None:
        """Execute a SQL script.
        
        Args:
            script: SQL script to execute.
        """
        with self.get_connection() as conn:
            conn.executescript(script)
    
    def close(self) -> None:
        """Close the connection to the database."""
        self.engine.dispose() 