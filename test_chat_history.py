#!/usr/bin/env python3
"""
Test script to verify chat history functionality works correctly.
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_chat_history():
    print("=== Testing Chat History Functionality ===")
    
    # Step 1: Login with the provided credentials
    print("\n1. Logging in with user 'abcd'...")
    login_data = {
        "username": "abcd",
        "password": "abc123*"
    }
    
    login_response = requests.post(f"{BASE_URL}/login", json=login_data)
    print(f"Login status: {login_response.status_code}")
    
    if login_response.status_code != 200:
        print(f"Login failed: {login_response.text}")
        return False
    
    login_result = login_response.json()
    token = login_result["access_token"]
    user_info = login_result["user"]
    print(f"Login successful! User: {user_info['username']} (ID: {user_info['user_id']})")
    
    # Headers for authenticated requests
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Step 2: Test chat history retrieval
    print("\n2. Testing chat history retrieval...")
    history_response = requests.get(f"{BASE_URL}/chat-history", headers=headers)
    print(f"Chat history status: {history_response.status_code}")
    
    if history_response.status_code == 200:
        history_data = history_response.json()
        print(f"✅ Chat history retrieved successfully!")
        print(f"Number of chats: {len(history_data['chats'])}")
        
        for i, chat in enumerate(history_data['chats'][:3]):  # Show first 3 chats
            print(f"  Chat {i+1}: {chat['title']} (ID: {chat['id']}, Messages: {chat['message_count']})")
    else:
        print(f"❌ Chat history failed: {history_response.text}")
        return False
    
    # Step 3: If there are existing chats, test loading one
    if history_data['chats']:
        print(f"\n3. Testing chat loading...")
        first_chat = history_data['chats'][0]
        load_data = {"chat_id": first_chat['id']}
        
        load_response = requests.post(f"{BASE_URL}/load-chat", json=load_data, headers=headers)
        print(f"Load chat status: {load_response.status_code}")
        
        if load_response.status_code == 200:
            load_result = load_response.json()
            print(f"✅ Chat loaded successfully!")
            print(f"Chat title: {load_result['title']}")
            print(f"Number of messages: {len(load_result['messages'])}")
            
            # Show first few messages
            for i, msg in enumerate(load_result['messages'][:2]):
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"  Message {i+1} ({msg['role']}): {content}")
        else:
            print(f"❌ Chat loading failed: {load_response.text}")
            return False
    else:
        print("\n3. No existing chats found to test loading.")
    
    # Step 4: Test creating a new chat by sending a message
    print(f"\n4. Testing new chat creation...")
    chat_data = {
        "messages": [
            {"role": "user", "content": "Hello, this is a test message for chat history!"}
        ],
        "model": "llama-3.3-70b-versatile"
    }
    
    chat_response = requests.post(f"{BASE_URL}/chat", json=chat_data, headers=headers)
    print(f"Chat status: {chat_response.status_code}")
    
    if chat_response.status_code == 200:
        chat_result = chat_response.json()
        print(f"✅ Chat successful!")
        print(f"Response preview: {chat_result['message'][:150]}...")
    else:
        print(f"❌ Chat failed: {chat_response.text}")
        return False
    
    # Step 5: Check chat history again to see if new chat appears
    print(f"\n5. Checking updated chat history...")
    history_response2 = requests.get(f"{BASE_URL}/chat-history", headers=headers)
    
    if history_response2.status_code == 200:
        history_data2 = history_response2.json()
        print(f"✅ Updated chat history retrieved!")
        print(f"Number of chats: {len(history_data2['chats'])}")
        
        if len(history_data2['chats']) > len(history_data['chats']):
            print("✅ New chat was added to history!")
        
        # Show most recent chat
        if history_data2['chats']:
            recent_chat = history_data2['chats'][0]  # Should be most recent
            print(f"Most recent chat: {recent_chat['title']} (Messages: {recent_chat['message_count']})")
    else:
        print(f"❌ Updated chat history failed: {history_response2.text}")
        return False
    
    print(f"\n🎉 All chat history tests passed successfully!")
    return True

if __name__ == "__main__":
    success = test_chat_history()
    if success:
        print("\n✅ Chat history functionality is working correctly!")
    else:
        print("\n❌ Chat history functionality has issues!")
