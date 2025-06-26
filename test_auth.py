#!/usr/bin/env python3
"""
Test script for the FastAPI authentication system
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_registration():
    """Test user registration"""
    print("Testing user registration...")
    
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123"
    }
    
    response = requests.post(f"{BASE_URL}/register", json=user_data)
    
    if response.status_code == 200:
        print("✓ Registration successful")
        data = response.json()
        print(f"✓ Received token: {data['access_token'][:20]}...")
        return data['access_token']
    else:
        print(f"✗ Registration failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def test_login():
    """Test user login"""
    print("\nTesting user login...")
    
    login_data = {
        "username": "testuser",
        "password": "testpass123"
    }
    
    response = requests.post(f"{BASE_URL}/login", json=login_data)
    
    if response.status_code == 200:
        print("✓ Login successful")
        data = response.json()
        print(f"✓ Received token: {data['access_token'][:20]}...")
        return data['access_token']
    else:
        print(f"✗ Login failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None

def test_protected_endpoint(token):
    """Test accessing a protected endpoint"""
    print("\nTesting protected endpoint...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/profile", headers=headers)
    
    if response.status_code == 200:
        print("✓ Protected endpoint access successful")
        data = response.json()
        print(f"✓ User profile: {data}")
        return True
    else:
        print(f"✗ Protected endpoint access failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_chat(token):
    """Test the chat endpoint with authentication"""
    print("\nTesting authenticated chat...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    chat_data = {
        "messages": [{"role": "user", "content": "Hello, this is a test message!"}],
        "model": "llama-3.3-70b-versatile"
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_data, headers=headers)
    
    if response.status_code == 200:
        print("✓ Authenticated chat successful")
        data = response.json()
        print(f"✓ Chat response: {data['message'][:100]}...")
        return True
    else:
        print(f"✗ Authenticated chat failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def test_chat_history(token):
    """Test the chat history endpoint"""
    print("\nTesting chat history...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/chat-history", headers=headers)
    
    if response.status_code == 200:
        print("✓ Chat history access successful")
        data = response.json()
        print(f"✓ Number of chats: {len(data.get('chats', []))}")
        return True
    else:
        print(f"✗ Chat history access failed: {response.status_code}")
        print(f"Error: {response.text}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting FastAPI Authentication System Tests\n")
    
    # Test registration
    token = test_registration()
    if not token:
        print("\n❌ Cannot proceed without successful registration")
        return
    
    # Test login
    login_token = test_login()
    if not login_token:
        print("\n❌ Login test failed")
        return
    
    # Test protected endpoint
    if not test_protected_endpoint(login_token):
        print("\n❌ Protected endpoint test failed")
        return
    
    # Test chat
    if not test_chat(login_token):
        print("\n❌ Chat test failed")
        return
    
    # Test chat history
    if not test_chat_history(login_token):
        print("\n❌ Chat history test failed")
        return
    
    print("\n✅ All tests passed! Authentication system is working correctly.")

if __name__ == "__main__":
    main()
