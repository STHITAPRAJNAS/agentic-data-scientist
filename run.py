#!/usr/bin/env python
"""Helper script to run the Agentic Data Scientist application."""
import os
import sys
import subprocess
import argparse

def main():
    """Main function to help run the Agentic Data Scientist."""
    parser = argparse.ArgumentParser(description="Agentic Data Scientist Runner")
    parser.add_argument('command', nargs='?', choices=['setup', 'app', 'example'], 
                       help='Command to run (setup: set up test database, app: run Streamlit app, example: run basic example)')
    
    args = parser.parse_args()
    
    if not args.command:
        print_help()
        return
    
    if args.command == 'setup':
        setup_database()
    elif args.command == 'app':
        run_app()
    elif args.command == 'example':
        run_example()

def print_help():
    """Print help information."""
    print("===== Agentic Data Scientist =====")
    print("This script helps you run the Agentic Data Scientist application.\n")
    print("Available commands:")
    print("  python run.py setup    - Set up the test database with sample data")
    print("  python run.py app      - Run the Streamlit application")
    print("  python run.py example  - Run the basic usage example")
    print("\nYou'll need a Google Gemini API key to use the application.")
    print("You can set it in the Streamlit UI or as an environment variable GOOGLE_API_KEY.")

def setup_database():
    """Set up the test database."""
    print("Setting up test database with sample data...")
    
    try:
        subprocess.run(["python", "scripts/setup_test_db.py"], check=True)
        print("\nDatabase setup complete!")
        print("You can now run the application with: python run.py app")
    except subprocess.CalledProcessError as e:
        print(f"Error setting up database: {e}")
        sys.exit(1)

def run_app():
    """Run the Streamlit application."""
    print("Starting Streamlit application...")
    print("You'll need to provide your Google Gemini API key in the sidebar when the app opens.")
    
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit application: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: streamlit command not found. Please install streamlit with:")
        print("pip install streamlit")
        sys.exit(1)

def run_example():
    """Run the basic usage example."""
    print("Running basic usage example...")
    print("You'll need to provide your Google Gemini API key if not set as an environment variable.")
    
    try:
        subprocess.run(["python", "examples/basic_usage.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running example: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 