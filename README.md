# FastAPI + React Chatbot

A modern chatbot application built with FastAPI backend and React frontend, powered by Groq's LLM API.

## Features

- 🤖 AI-powered conversations using Groq's Llama model
- 💬 Real-time chat interface
- 🎨 Modern, responsive UI
- 🚀 Fast API backend with automatic documentation
- 🔄 CORS enabled for seamless frontend-backend communication

## Project Structure

```
FastAPI chatbot/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── .env                # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   ├── App.css         # Styling
│   │   ├── index.js        # React entry point
│   │   └── index.css       # Global styles
│   ├── public/
│   │   └── index.html      # HTML template
│   └── package.json        # Node.js dependencies
└── README.md               # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 16+
- Groq API key ([Get one here](https://console.groq.com/))

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Set up environment variables:
   - Copy the `.env.example` file to `.env`:
   ```bash
   # Windows
   copy .env.example .env
   
   # macOS/Linux
   cp .env.example .env
   ```
   - Edit the `.env` file and add your actual API keys:
   ```
   GROQ_API_KEY=your_actual_groq_api_key_here
   COHERE_API_KEY=your_actual_cohere_api_key_here
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   ```
   
   **Important**: 
   - Get your Groq API key from [Groq Console](https://console.groq.com/)
   - Get your Cohere API key from [Cohere Platform](https://dashboard.cohere.ai/)
   - Get your Gemini API key from [Google AI Studio](https://aistudio.google.com/)

6. Run the FastAPI server:
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the React development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## API Endpoints

### Backend API

- `GET /` - Welcome message
- `POST /chat` - Send chat message
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)

### Example API Usage

```javascript
// Send a chat message
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    messages: [
      { role: 'user', content: 'Hello, how are you?' }
    ],
    model: 'llama-3.3-70b-versatile'
  })
});

const data = await response.json();
console.log(data.message); // AI response
```

## Usage

1. Start both the backend and frontend servers
2. Open your browser to `http://localhost:3000`
3. Start chatting with the AI assistant!

## Development

### Backend Development

- The FastAPI server includes automatic reload during development
- Visit `http://localhost:8000/docs` for interactive API documentation
- Check `http://localhost:8000/health` for server status

### Frontend Development

- The React app includes hot reload during development
- Modify components in `src/` to see changes instantly
- The app is configured to proxy API requests to the backend

## Environment Variables

### Backend (.env)
```
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
HOST=0.0.0.0
PORT=8000
```

**Required API Keys:**
- **GROQ_API_KEY**: For chat completions and text generation
- **COHERE_API_KEY**: For RAG (Retrieval-Augmented Generation) embeddings and context memory
- **GEMINI_API_KEY**: For image generation, editing, and video analysis features

## Troubleshooting

### Common Issues

1. **CORS errors**: Make sure the backend is running on port 8000 and frontend on port 3000
2. **API key errors**: Verify your API keys are correctly set in the `.env` file:
   - Check that GROQ_API_KEY, COHERE_API_KEY, and GEMINI_API_KEY are all configured
   - Ensure there are no extra spaces or quotes around the API keys
3. **Connection refused**: Ensure both servers are running before testing

### Logs

- Backend logs are displayed in the terminal where you run `python main.py`
- Frontend logs are available in the browser console (F12)

## License

This project is open source and available under the MIT License.
