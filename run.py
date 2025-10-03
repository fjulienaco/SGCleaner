#!/usr/bin/env python3
"""
Script to run the enhanced Style Guide Cleaner application with OpenAI integration
"""

import subprocess
import sys
import os

def main():
    """Run the enhanced Streamlit application"""
    try:
        # Check if we're in the right directory
        if not os.path.exists('app.py'):
            print("Error: app.py not found. Please run this script from the SGCleaner directory.")
            sys.exit(1)
        
        print("🚀 Starting Style Guide Cleaner Pro...")
        print("Features:")
        print("  ✅ Enhanced DOCX processing")
        print("  ✅ AI-powered content optimization")
        print("  ✅ Advanced cleaning algorithms")
        print("  ✅ Detailed document analysis")
        print("\nThe application will open in your default web browser.")
        print("Press Ctrl+C to stop the application.\n")
        
        # Check for environment file
        if os.path.exists('.env'):
            print("📁 Found .env file - OpenAI integration available")
        else:
            print("⚠️  No .env file found - you can still use the app without OpenAI")
            print("   Create a .env file with your OPENAI_API_KEY for AI optimization")
        
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
