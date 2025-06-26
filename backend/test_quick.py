import requests
import json

def test_auth():
    # Test registration
    print('Testing registration...')
    user_data = {
        'username': 'testuser123',
        'email': 'test123@example.com', 
        'password': 'testpass123'
    }

    try:
        response = requests.post('http://localhost:8000/register', json=user_data)
        print(f'Registration response: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f'Success! Token: {data["access_token"][:20]}...')
            
            # Test login
            print('Testing login...')
            login_data = {
                'username': 'testuser123',
                'password': 'testpass123'
            }
            
            login_response = requests.post('http://localhost:8000/login', json=login_data)
            print(f'Login response: {login_response.status_code}')
            if login_response.status_code == 200:
                login_data = login_response.json()
                print(f'Login success! Token: {login_data["access_token"][:20]}...')
            else:
                print(f'Login error: {login_response.text}')
        else:
            print(f'Registration error: {response.text}')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    test_auth()
