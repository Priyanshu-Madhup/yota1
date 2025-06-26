# Deployment Guide for Yota Chatbot

## Overview
This guide covers deploying the Yota Chatbot to Render, a cloud platform that offers free hosting for web applications.

## Prerequisites
1. GitHub account with your code repository
2. Render account (free tier available)
3. Required API keys (see Environment Variables section)

## Architecture
- **Backend**: FastAPI Python application
- **Frontend**: React static site
- **Database**: File-based storage (conversation memory and user data)

## Deployment Steps

### 1. Prepare Your Repository
Ensure your repository is pushed to GitHub with all the latest changes:
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Set Up Render Services

#### Connect to Render:
1. Go to [render.com](https://render.com) and sign up/log in
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Select the repository containing your chatbot code

#### Render will automatically:
- Read the `render.yaml` configuration
- Create two services:
  - `yota-chatbot-backend` (Web Service)
  - `yota-chatbot-frontend` (Static Site)

### 3. Configure Environment Variables

In your Render dashboard, for the **backend service**, set these environment variables:

#### Required API Keys:
- `GROQ_API_KEY`: Your Groq API key for AI responses
- `GEMINI_API_KEY`: Your Google Gemini API key (backup AI model)
- `JWT_SECRET_KEY`: A secure random string for JWT token encryption
- `SERPER_API_KEY`: Already configured (for web search functionality)

#### System Variables (auto-configured):
- `PORT`: 8000 (already set in render.yaml)
- `REACT_APP_API_URL`: Auto-configured to point to backend service

### 4. Deployment Process

Once configured, Render will:

1. **Backend Deployment**:
   - Install Python dependencies from `backend/requirements.txt`
   - Start the FastAPI server on port 8000
   - Create health check endpoint at `/health`

2. **Frontend Deployment**:
   - Install Node.js dependencies
   - Build the React application
   - Deploy as a static site
   - Configure API URL to point to backend

### 5. Post-Deployment

#### Access Your Application:
- **Frontend URL**: `https://yota-chatbot-frontend.onrender.com`
- **Backend API**: `https://yota-chatbot-backend.onrender.com`

#### Test the Deployment:
1. Visit the frontend URL
2. Try logging in with test credentials:
   - Username: `abcd`
   - Password: `abc123*`
3. Send a test message to verify AI responses
4. Check that chat history persists

## Environment Variables Details

### Backend Environment Variables

| Variable | Purpose | Required | Example |
|----------|---------|----------|---------|
| `GROQ_API_KEY` | Primary AI model API key | Yes | `gsk_...` |
| `GEMINI_API_KEY` | Backup AI model API key | Yes | `AIza...` |
| `JWT_SECRET_KEY` | JWT token encryption | Yes | `your-super-secret-key-here` |
| `SERPER_API_KEY` | Web search functionality | No | Already configured |
| `PORT` | Server port | No | 8000 (auto-set) |

### Frontend Environment Variables

| Variable | Purpose | Auto-configured |
|----------|---------|-----------------|
| `REACT_APP_API_URL` | Backend API endpoint | Yes |

## File Structure for Deployment

```
project-root/
├── render.yaml              # Render configuration
├── backend/
│   ├── requirements.txt     # Python dependencies
│   ├── main.py             # FastAPI application
│   ├── users.json          # User authentication data
│   └── memory_storage/     # Chat history storage
└── frontend/
    ├── package.json        # Node.js dependencies
    ├── public/
    └── src/
        ├── App.js          # Main React component
        └── ...
```

## Troubleshooting

### Common Issues:

1. **Backend fails to start**:
   - Check that all required environment variables are set
   - Verify `requirements.txt` includes all dependencies
   - Check build logs for Python errors

2. **Frontend can't connect to backend**:
   - Verify `REACT_APP_API_URL` is properly configured
   - Check CORS settings in backend
   - Ensure backend service is running

3. **Authentication issues**:
   - Verify `JWT_SECRET_KEY` is set and consistent
   - Check `users.json` file exists and is properly formatted

### Logs and Monitoring:
- Access logs through Render dashboard
- Monitor service health via health check endpoint
- Check resource usage on free tier limits

## Free Tier Limitations

Render's free tier includes:
- 750 hours/month of compute time
- Services sleep after 15 minutes of inactivity
- Cold start delays when waking up sleeping services
- Limited bandwidth and storage

## Scaling Considerations

For production use, consider:
- Upgrading to paid Render plans
- Implementing proper database (PostgreSQL, MongoDB)
- Adding Redis for session management
- Setting up CI/CD pipelines
- Implementing comprehensive logging and monitoring

## Security Notes

- Never commit API keys to version control
- Use strong, unique JWT secret keys
- Consider implementing rate limiting
- Regular security updates for dependencies
- Monitor access logs for suspicious activity

## Support

For deployment issues:
1. Check Render documentation: [render.com/docs](https://render.com/docs)
2. Review this project's GitHub issues
3. Check Render community forum
4. Contact Render support for platform-specific issues
