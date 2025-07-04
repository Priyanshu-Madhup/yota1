#!/usr/bin/env python3
"""
Test script to validate API key environment variable setup
"""
import os
import sys
from dotenv import load_dotenv

def test_env_setup():
    print("🔍 Testing environment variable setup...")
    
    # Load environment variables
    load_dotenv()
    
    # Check for required API keys
    required_keys = ["GROQ_API_KEY", "COHERE_API_KEY", "GEMINI_API_KEY"]
    missing_keys = []
    placeholder_keys = []
    
    for key in required_keys:
        value = os.environ.get(key)
        print(f"  {key}: {'✅' if value else '❌'} {value or 'NOT SET'}")
        if not value:
            missing_keys.append(key)
        elif value.startswith("your_") and value.endswith("_here"):
            placeholder_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing environment variables: {', '.join(missing_keys)}")
        return False
    
    if placeholder_keys:
        print(f"⚠️  Placeholder values detected for: {', '.join(placeholder_keys)}")
        print("   Please replace these with your actual API keys!")
        return True  # Still valid setup, just needs real keys
    
    print("✅ All environment variables are properly configured!")
    return True

def test_imports():
    print("🔍 Testing critical imports...")
    
    try:
        from fastapi import FastAPI
        print("✅ FastAPI import successful")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        from groq import Groq
        print("✅ Groq import successful")
    except ImportError as e:
        print(f"❌ Groq import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv import successful")
    except ImportError as e:
        print(f"❌ python-dotenv import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 FastAPI Chatbot Environment Test")
    print("=" * 40)
    
    imports_ok = test_imports()
    env_ok = test_env_setup()
    
    if imports_ok and env_ok:
        print("\n🎉 Setup validation completed successfully!")
        print("   You can now run the FastAPI chatbot.")
        sys.exit(0)
    else:
        print("\n❌ Setup validation failed!")
        print("   Please fix the issues above before running the chatbot.")
        sys.exit(1)
