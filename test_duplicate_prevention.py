#!/usr/bin/env python3
"""
Test script to check for duplicate conversations and clean them up.
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_duplicate_prevention():
    print("=== Testing Duplicate Prevention ===")
    
    # Login
    print("\n1. Logging in...")
    login_data = {
        "username": "abcd",
        "password": "abc123*"
    }
    
    login_response = requests.post(f"{BASE_URL}/login", json=login_data)
    if login_response.status_code != 200:
        print(f"Login failed: {login_response.text}")
        return False
    
    login_result = login_response.json()
    token = login_result["access_token"]
    user_info = login_result["user"]
    print(f"✅ Login successful! User: {user_info['username']}")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Check current chat count
    print("\n2. Checking current chat history...")
    history_response = requests.get(f"{BASE_URL}/chat-history", headers=headers)
    if history_response.status_code == 200:
        current_chats = history_response.json()['chats']
        print(f"Current number of chats: {len(current_chats)}")
        
        # Show chat titles to identify duplicates
        print("Current chats:")
        for i, chat in enumerate(current_chats[:10]):  # Show first 10
            print(f"  {i+1}. {chat['title']} (ID: {chat['id']}, Messages: {chat['message_count']})")
    else:
        print(f"❌ Failed to get chat history: {history_response.text}")
        return False
    
    # Test creating the same message multiple times
    print(f"\n3. Testing duplicate prevention by sending same message 3 times...")
    test_message = "This is a duplicate test message"
    
    for i in range(3):
        print(f"  Sending message {i+1}/3...")
        chat_data = {
            "messages": [
                {"role": "user", "content": test_message}
            ],
            "model": "llama-3.3-70b-versatile",
            "chat_id": "duplicate_test_chat"  # Use same chat_id
        }
        
        chat_response = requests.post(f"{BASE_URL}/chat", json=chat_data, headers=headers)
        if chat_response.status_code == 200:
            print(f"    ✅ Message {i+1} sent successfully")
        else:
            print(f"    ❌ Message {i+1} failed: {chat_response.text}")
    
    # Check if duplicates were prevented
    print(f"\n4. Checking for duplicates after test...")
    history_response2 = requests.get(f"{BASE_URL}/chat-history", headers=headers)
    if history_response2.status_code == 200:
        updated_chats = history_response2.json()['chats']
        print(f"Updated number of chats: {len(updated_chats)}")
        
        # Look for our test chat
        test_chat = None
        for chat in updated_chats:
            if chat['id'] == 'duplicate_test_chat':
                test_chat = chat
                break
        
        if test_chat:
            print(f"Test chat found: {test_chat['title']} (Messages: {test_chat['message_count']})")
            
            # Load the chat to see actual messages
            load_data = {"chat_id": "duplicate_test_chat"}
            load_response = requests.post(f"{BASE_URL}/load-chat", json=load_data, headers=headers)
            
            if load_response.status_code == 200:
                load_result = load_response.json()
                messages = load_result['messages']
                print(f"Actual messages in test chat: {len(messages)}")
                
                # Count duplicate user messages
                user_messages = [msg for msg in messages if msg['role'] == 'user']
                duplicate_count = sum(1 for msg in user_messages if msg['content'] == test_message)
                
                if duplicate_count == 1:
                    print("✅ Duplicate prevention working! Only 1 copy of the test message found.")
                else:
                    print(f"❌ Duplicate prevention failed! Found {duplicate_count} copies of the test message.")
            else:
                print(f"❌ Failed to load test chat: {load_response.text}")
        else:
            print("❌ Test chat not found")
    else:
        print(f"❌ Failed to get updated chat history: {history_response2.text}")
    
    print(f"\n=== Duplicate Prevention Test Complete ===")
    return True
    
    login_response = requests.post(f"{BASE_URL}/login", json=login_data)
    if login_response.status_code != 200:
        print(f"Login failed: {login_response.text}")
        return False
    
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    # Step 2: Create a unique test chat
    test_chat_id = f"test_duplicate_{int(time.time())}"
    print(f"\n2. Creating test chat with ID: {test_chat_id}")
    
    test_messages = [
        {"role": "user", "content": "This is a test message for duplicate prevention"},
        {"role": "assistant", "content": "This is a test response for duplicate prevention"}
    ]
    
    # Step 3: Save the same chat multiple times (simulating the bug)
    print(f"\n3. Saving the same chat 3 times (simulating duplicate bug)...")
    for i in range(3):
        save_data = {
            "chat_id": test_chat_id,
            "title": f"Duplicate Test Chat",
            "messages": test_messages
        }
        
        save_response = requests.post(f"{BASE_URL}/save-chat", json=save_data, headers=headers)
        print(f"   Save attempt {i+1}: {save_response.status_code}")
        
        if save_response.status_code != 200:
            print(f"   Save failed: {save_response.text}")
    
    # Step 4: Check chat history to see if duplicates were created
    print(f"\n4. Checking if duplicates were prevented...")
    history_response = requests.get(f"{BASE_URL}/chat-history", headers=headers)
    
    if history_response.status_code == 200:
        history_data = history_response.json()
        
        # Count how many chats have our test ID
        duplicate_count = sum(1 for chat in history_data['chats'] if chat['id'] == test_chat_id)
        
        print(f"   Number of chats with test ID '{test_chat_id}': {duplicate_count}")
        
        if duplicate_count == 1:
            print("   ✅ SUCCESS: No duplicate chats were created!")
        else:
            print(f"   ❌ FAILURE: {duplicate_count} duplicate chats found!")
            
        # Show recent chats
        print(f"\n   Recent chats:")
        for chat in history_data['chats'][:5]:
            print(f"   - {chat['title']} (ID: {chat['id']}, Messages: {chat['message_count']})")
    
    # Step 5: Clean up - delete the test chat
    print(f"\n5. Cleaning up test chat...")
    delete_response = requests.delete(f"{BASE_URL}/delete-chat/{test_chat_id}", headers=headers)
    print(f"   Delete status: {delete_response.status_code}")
    
    return True

if __name__ == "__main__":
    success = test_duplicate_prevention()
    if success:
        print(f"\n🎉 Duplicate prevention test completed!")
    else:
        print(f"\n❌ Test failed!")
