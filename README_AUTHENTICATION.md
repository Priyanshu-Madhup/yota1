# FastAPI Chatbot with Authentication System

## Overview

This FastAPI + React chatbot now includes a complete authentication system with per-user isolated memory storage. Each user has their own separate chat histories and RAG (Retrieval-Augmented Generation) memory, ensuring complete data isolation between users.

## 🔐 Authentication Features

### Backend Authentication
- **JWT-based Authentication**: Secure token-based authentication
- **User Registration & Login**: Complete user management system
- **Password Hashing**: Secure bcrypt password hashing
- **Protected Endpoints**: All chat endpoints require authentication
- **Per-User Memory**: Isolated conversation storage and RAG memory per user

### Frontend Authentication
- **Login/Register UI**: Beautiful animated login page with floating "YOTA AI" background
- **Token Management**: Automatic token storage and refresh
- **Authentication Flow**: Seamless login/logout experience
- **User Session**: Persistent authentication across browser sessions

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js 14+
- Required API keys: Groq API, Gemini API (optional)

### Environment Setup

1. **Create environment file**:
   ```bash
   cd backend
   cp .env.example .env
   ```

2. **Configure your .env file**:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   JWT_SECRET_KEY=your_super_secret_jwt_key_here_make_it_long_and_random
   LOG_LEVEL=INFO
   ```

### Backend Setup

1. **Install dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Start the server**:
   ```bash
   python main.py
   # or
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm start
   ```

## 🔑 API Endpoints

### Authentication Endpoints
- `POST /register` - Register a new user
- `POST /login` - Login with username/password
- `GET /profile` - Get current user profile (protected)

### Chat Endpoints (All require authentication)
- `POST /chat` - Send a message and get AI response
- `POST /chat-with-image` - Analyze images with AI
- `POST /generate-image` - Generate images with AI
- `POST /upload-image` - Upload images for analysis

### Chat Management (All require authentication)
- `GET /chat-history` - Get user's chat history
- `POST /load-chat` - Load a specific chat
- `DELETE /delete-chat/{chat_id}` - Delete a chat
- `POST /save-chat` - Save chat to storage

## 📂 Per-User Memory Storage

### Directory Structure
```
backend/memory_storage/users/
├── user_123/
│   ├── conversation_memory.pkl
│   ├── conversation_faiss.index
│   └── chat_metadata.json
├── user_456/
│   ├── conversation_memory.pkl
│   ├── conversation_faiss.index
│   └── chat_metadata.json
└── ...
```

### Memory Features
- **Isolated Storage**: Each user has their own memory files
- **FAISS Indexing**: Fast semantic search within user's conversations
- **Conversation Context**: RAG retrieval from user's previous chats
- **Chat Persistence**: All conversations saved and retrievable

## 🎨 Frontend Features

### Login Page
- Animated floating "YOTA AI" background text
- Glassmorphism design with purple gradients
- Toggle between login and signup modes
- Form validation and error handling
- Password visibility toggles

### Chat Interface
- User-specific chat history in sidebar
- Welcome message with user's name
- Logout button in sidebar
- All existing chat features (web search, image analysis, image generation)

## 🧪 Testing

### Test the Authentication System
```bash
# Run the test script
python test_auth.py
```

This will test:
- User registration
- User login
- Protected endpoint access
- Authenticated chat
- Chat history retrieval

### Manual Testing
1. Start both backend and frontend servers
2. Navigate to `http://localhost:3000`
3. Register a new account or login
4. Test chat functionality
5. Verify chat history is user-specific

## 🔒 Security Features

- **JWT Tokens**: Secure, stateless authentication
- **Password Hashing**: bcrypt with salt for password security
- **Token Expiration**: Configurable token lifetime
- **CORS Protection**: Configured for secure cross-origin requests
- **Data Isolation**: Complete separation of user data

## 📝 User Management

### User Data Storage
- User accounts stored in `backend/users.json`
- Passwords hashed with bcrypt
- Unique user IDs for memory isolation

### Authentication Flow
1. User registers or logs in
2. Backend returns JWT token
3. Frontend stores token in localStorage
4. All API requests include Authorization header
5. Backend validates token and extracts user info
6. User-specific operations use extracted user_id

## 🛠️ Development Notes

### Adding New Protected Endpoints
```python
@app.get("/new-endpoint")
async def new_endpoint(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    # Your endpoint logic here
```

### Frontend API Calls
```javascript
// The apiClient automatically includes authentication headers
const response = await apiClient.post('/chat', chatData);
```

### User Context in Components
```javascript
// User info is available in the main App component
const user = JSON.parse(localStorage.getItem('user'));
console.log(`Current user: ${user.username}`);
```

## 🚨 Troubleshooting

### Common Issues

1. **401 Unauthorized Errors**
   - Check if JWT_SECRET_KEY is set in .env
   - Verify token is not expired
   - Ensure Authorization header is included

2. **CORS Issues**
   - Verify CORS origins in main.py
   - Check frontend is running on correct port

3. **Memory Storage Issues**
   - Ensure backend/memory_storage/users/ directory exists
   - Check file permissions for the backend process

### Debug Mode
Set `LOG_LEVEL=DEBUG` in .env for detailed logging.

## 📦 Dependencies

### Backend
- FastAPI
- PyJWT
- bcrypt
- python-multipart
- And all existing dependencies

### Frontend
- All existing React dependencies
- No additional dependencies for authentication

## 🔄 Migration from Non-Auth Version

If upgrading from a version without authentication:

1. Existing conversation data will not be accessible (new user-based storage)
2. Users will need to register new accounts
3. Previous chat history will be lost (consider data migration if needed)

## 🎯 Future Enhancements

- Password reset functionality
- Email verification
- User profile management
- Admin panel for user management
- OAuth integration (Google, GitHub, etc.)
- Rate limiting per user
- User subscription tiers
