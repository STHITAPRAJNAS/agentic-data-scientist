import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

# Load environment variables from .env file if it exists
load_dotenv()

class LLMConfig(BaseModel):
    """Configuration for the LLM."""
    provider: str = Field(default="gemini", description="LLM provider: gemini")
    api_key: Optional[str] = Field(default=None, description="API key for the LLM provider")
    model: str = Field(default="gemini-pro", description="Model to use")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")

class DatabaseConfig(BaseModel):
    """Configuration for the database."""
    provider: str = Field(default="sqlite", description="Database provider: sqlite")
    path: str = Field(default="data/database.db", description="Path to the SQLite database")
    connection_string: Optional[str] = Field(default=None, description="Connection string for other databases")

class AgentConfig(BaseModel):
    """Configuration for the agents."""
    memory_size: int = Field(default=5, description="Number of messages to keep in agent memory")
    timeout: int = Field(default=120, description="Timeout for agent execution in seconds")
    max_iterations: int = Field(default=10, description="Maximum iterations for agent execution")

class AppConfig(BaseModel):
    """Main application configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    debug: bool = Field(default=False, description="Enable debug mode")
    data_dir: str = Field(default="data", description="Directory for data storage")
    temp_dir: str = Field(default="temp", description="Directory for temporary files")
    max_upload_size_mb: int = Field(default=50, description="Maximum upload size in MB")

def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    # Override defaults with environment variables if they exist
    llm_config = LLMConfig(
        provider=os.getenv("LLM_PROVIDER", "gemini"),
        api_key=os.getenv("GEMINI_API_KEY"),
        model=os.getenv("LLM_MODEL", "gemini-pro"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
    )
    
    db_config = DatabaseConfig(
        provider=os.getenv("DB_PROVIDER", "sqlite"),
        path=os.getenv("DB_PATH", "data/database.db"),
        connection_string=os.getenv("DB_CONNECTION_STRING"),
    )
    
    agent_config = AgentConfig(
        memory_size=int(os.getenv("AGENT_MEMORY_SIZE", "5")),
        timeout=int(os.getenv("AGENT_TIMEOUT", "120")),
        max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "10")),
    )
    
    return AppConfig(
        llm=llm_config,
        database=db_config,
        agent=agent_config,
        debug=os.getenv("DEBUG", "False").lower() == "true",
        data_dir=os.getenv("DATA_DIR", "data"),
        temp_dir=os.getenv("TEMP_DIR", "temp"),
        max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "50")),
    )

# Create a global config instance
config = load_config() 