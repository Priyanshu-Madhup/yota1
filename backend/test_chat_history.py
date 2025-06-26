#!/usr/bin/env python3
"""
Test script for chat history functionality
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_login():
    """Test login with existing user"""
    print("Testing login...")
    
    # Try to login with existing user
    login_data = {
        "username": "abcd",
        "password": "abcd"
    }
    
    response = requests.post(f"{BASE_URL}/login", json=login_data)
    print(f"Login response: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        token = data["access_token"]
        print(f"Login success! Token: {token[:50]}...")
        return token
    else:
        print(f"Login error: {response.text}")
        return None

def test_chat_history(token):
    """Test chat history retrieval"""
    print("\nTesting chat history...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(f"{BASE_URL}/chat-history", headers=headers)
    print(f"Chat history response: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Chat history success! Found {len(data['chats'])} chats")
        for chat in data["chats"][:3]:  # Show first 3 chats
            print(f"  - Chat {chat['id']}: '{chat['title']}' ({chat['message_count']} messages)")
        return True
    else:
        print(f"Chat history error: {response.text}")
        return False

def test_send_chat(token):
    """Send a test chat message"""
    print("\nSending test chat...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    chat_data = {
        "messages": [
            {"role": "user", "content": "Hello, test message for chat history"}
        ],
        "model": "llama-3.3-70b-versatile",
        "chat_id": "test_chat_history_123"
    }
    
    response = requests.post(f"{BASE_URL}/chat", json=chat_data, headers=headers)
    print(f"Chat response: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Chat success! Response: {data['message'][:100]}...")
        return True
    else:
        print(f"Chat error: {response.text}")
        return False

def test_save_chat(token):
    """Test saving a chat"""
    print("\nTesting save chat...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    save_data = {
        "chat_id": "test_save_chat_456",
        "title": "Test Chat for History",
        "messages": [
            {"role": "user", "content": "Test user message"},
            {"role": "assistant", "content": "Test assistant response"}
        ]
    }
    
    response = requests.post(f"{BASE_URL}/save-chat", json=save_data, headers=headers)
    print(f"Save chat response: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Save chat success! Chat ID: {data['chat_id']}")
        return True
    else:
        print(f"Save chat error: {response.text}")
        return False

def main():
    """Main test function"""
    print("=== Testing Chat History Functionality ===")
    
    # Test login
    token = test_login()
    if not token:
        print("❌ Cannot proceed without valid token")
        return
    
    # Test sending a chat (which should add to memory)
    test_send_chat(token)
    
    # Test saving a chat
    test_save_chat(token)
    
    # Test retrieving chat history
    success = test_chat_history(token)
    
    if success:
        print("\n✅ Chat history functionality appears to be working!")
    else:
        print("\n❌ Chat history functionality has issues!")

if __name__ == "__main__":
    main()
