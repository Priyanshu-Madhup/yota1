import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import Login from './Login';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Axios instance with auth configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

// Add request interceptor to include auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle auth errors
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

// Utility function to convert HTML back to markdown-like format
const htmlToMarkdown = (html) => {
  if (!html) return '';
  
  return html
    // Convert headings
    .replace(/<h2[^>]*>(.*?)<\/h2>/gi, '\n## $1\n')
    .replace(/<h3[^>]*>(.*?)<\/h3>/gi, '\n### $1\n')
    .replace(/<h4[^>]*>(.*?)<\/h4>/gi, '\n#### $1\n')
    // Convert paragraphs
    .replace(/<p[^>]*>(.*?)<\/p>/gi, '\n$1\n')
    // Convert lists
    .replace(/<ul[^>]*>/gi, '\n')
    .replace(/<\/ul>/gi, '\n')
    .replace(/<ol[^>]*>/gi, '\n')
    .replace(/<\/ol>/gi, '\n')
    .replace(/<li[^>]*>(.*?)<\/li>/gi, '• $1\n')
    // Convert code blocks (preserve them)
    .replace(/<pre[^>]*><code[^>]*>(.*?)<\/code><\/pre>/gis, '\n```\n$1\n```\n')
    // Convert inline code
    .replace(/<code[^>]*>(.*?)<\/code>/gi, '`$1`')
    // Convert bold
    .replace(/<strong[^>]*>(.*?)<\/strong>/gi, '**$1**')
    // Convert links
    .replace(/<a[^>]*href="([^"]*)"[^>]*>(.*?)<\/a>/gi, '[$2]($1)')
    // Remove remaining HTML tags
    .replace(/<[^>]+>/g, '')
    // Clean up extra whitespace
    .replace(/\n\s*\n/g, '\n\n')
    .trim();
};

// Custom components for react-markdown
const markdownComponents = {
  code({ node, inline, className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || '');
    const language = match ? match[1] : 'text';
    
    if (!inline) {
      return (
        <div className="code-block-container">
          <div className="code-block-header">
            <span className="code-language">{language}</span>
            <button
              className="copy-btn"
              onClick={() => {
                navigator.clipboard.writeText(String(children).replace(/\n$/, ''));
              }}
            >
              Copy
            </button>
          </div>
          <SyntaxHighlighter
            style={vscDarkPlus}
            language={language}
            PreTag="div"
            {...props}
          >
            {String(children).replace(/\n$/, '')}
          </SyntaxHighlighter>
        </div>
      );
    }
    
    return (
      <code className="inline-code" {...props}>
        {children}
      </code>
    );
  },
  
  h1: ({ children }) => <h1 className="markdown-h1">{children}</h1>,
  h2: ({ children }) => <h2 className="markdown-h2">{children}</h2>,
  h3: ({ children }) => <h3 className="markdown-h3">{children}</h3>,
  h4: ({ children }) => <h4 className="markdown-h4">{children}</h4>,
  
  p: ({ children }) => <p className="markdown-p">{children}</p>,
  
  ul: ({ children }) => <ul className="markdown-ul">{children}</ul>,
  ol: ({ children }) => <ol className="markdown-ol">{children}</ol>,
  li: ({ children }) => <li className="markdown-li">{children}</li>,
  
  a: ({ href, children }) => (
    <a href={href} className="markdown-link" target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
  
  strong: ({ children }) => <strong className="markdown-strong">{children}</strong>,
};

function App() {
  // Authentication state
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null);
  const [isLoadingAuth, setIsLoadingAuth] = useState(true);
  
  // Existing states
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);
  const [imageGenerationEnabled, setImageGenerationEnabled] = useState(false);
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);
  const [lastSavedChatId, setLastSavedChatId] = useState(null);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Check authentication on mount
  useEffect(() => {
    const checkAuth = async () => {
      const token = localStorage.getItem('token');
      const userData = localStorage.getItem('user');
      
      if (token && userData) {
        try {
          // Verify token is still valid by making an authenticated request
          await apiClient.get('/profile');
          setUser(JSON.parse(userData));
          setIsAuthenticated(true);
          
          // Load chat history after authentication
          await loadChatHistory();
        } catch (error) {
          console.error('Token validation failed:', error);
          localStorage.removeItem('token');
          localStorage.removeItem('user');
        }
      }
      setIsLoadingAuth(false);
    };
    
    checkAuth();
  }, []);

  // Load chat history
  const loadChatHistory = async () => {
    try {
      const response = await apiClient.get('/chat-history');
      if (response.status === 200) {
        // Backend now handles deduplication, so we can directly use the data
        setChatHistory(response.data.chats);
      }
    } catch (error) {
      console.error('Error loading chat history:', error);
    }
  };

  // Authentication handlers
  const handleLogin = async (credentials) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/login`, credentials);
      const { access_token, user } = response.data;
      
      localStorage.setItem('token', access_token);
      localStorage.setItem('user', JSON.stringify(user));
      
      setUser(user);
      setIsAuthenticated(true);
      
      // Load chat history after login
      await loadChatHistory();
      
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Login failed' 
      };
    }
  };

  const handleRegister = async (userData) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/register`, userData);
      const { access_token, user } = response.data;
      
      localStorage.setItem('token', access_token);
      localStorage.setItem('user', JSON.stringify(user));
      
      setUser(user);
      setIsAuthenticated(true);
      
      return { success: true };
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.detail || 'Registration failed' 
      };
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setIsAuthenticated(false);
    setMessages([]);
    setChatHistory([]);
    setCurrentChatId(null);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Auto-focus input field when component mounts and after messages update
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [messages]); // Focus after messages change (after sending/receiving)

  useEffect(() => {
    // Set initial sidebar state based on screen size
    const handleResize = () => {
      // Always start with sidebar closed, regardless of screen size
      // User must manually open it with hamburger button
      if (window.innerWidth <= 768) {
        setSidebarOpen(false); // Closed on mobile
      }
      // On desktop, don't auto-open, let user control it
    };

    // Set initial state (always closed)
    setSidebarOpen(false);

    // Add resize listener
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // This useEffect is now handled by the authentication check above

  // Save current chat to backend when navigating away
  useEffect(() => {
    const saveCurrentChat = async () => {
      // Only save if we have messages, a chat ID, and haven't already saved this chat recently
      if (messages.length > 0 && currentChatId && lastSavedChatId !== currentChatId) {
        try {
          const chatTitle = generateChatTitle(messages.find(m => m.role === 'user')?.content || 'New Chat');
          await apiClient.post('/save-chat', {
            chat_id: currentChatId,
            title: chatTitle,
            messages: messages
          });
        } catch (error) {
          console.error('Error saving chat on unload:', error);
        }
      }
    };

    const handleBeforeUnload = () => {
      saveCurrentChat();
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [messages, currentChatId, lastSavedChatId]);

  const generateChatTitle = (firstMessage) => {
    return firstMessage.length > 30 ? firstMessage.substring(0, 30) + '...' : firstMessage;
  };

  const startNewChat = async () => {
    // Start new chat without saving - chats are auto-saved when messages are sent
    setMessages([]);
    setCurrentChatId(null);  // Will be generated when first message is sent
    setLastSavedChatId(null); // Reset saved chat tracking
    setSidebarOpen(false);
    // Reset toggle states
    setWebSearchEnabled(false);
    setImageGenerationEnabled(false);
    setIsGeneratingImage(false);
    
    // Auto-focus input field for new chat
    setTimeout(() => {
      if (inputRef.current) {
        inputRef.current.focus();
      }
    }, 100);
  };

  const loadChat = async (chat) => {
    try {
      const response = await apiClient.post('/load-chat', {
        chat_id: chat.id
      });
      
      if (response.status === 200) {
        const data = response.data;
        setMessages(data.messages);
        setCurrentChatId(data.chat_id);
        setSidebarOpen(false);
        
        // Auto-focus input field after loading chat
        setTimeout(() => {
          if (inputRef.current) {
            inputRef.current.focus();
          }
        }, 100);
      }
    } catch (error) {
      console.error('Error loading chat:', error);
    }
  };

  const deleteChat = async (chatId) => {
    try {
      const response = await apiClient.delete(`/delete-chat/${chatId}`);
      
      if (response.status === 200) {
        setChatHistory(prev => prev.filter(chat => chat.id !== chatId));
        if (currentChatId === chatId) {
          setMessages([]);
          setCurrentChatId(null);
        }
      }
    } catch (error) {
      console.error('Error deleting chat:', error);
    }
  };

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  const toggleWebSearch = () => {
    setWebSearchEnabled(!webSearchEnabled);
  };

  const toggleImageGeneration = () => {
    setImageGenerationEnabled(!imageGenerationEnabled);
    console.log('Image generation toggled:', !imageGenerationEnabled);
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if ((!inputMessage.trim() && !selectedImage) || isLoading) return;

    // Generate chat ID if this is a new conversation
    if (!currentChatId) {
      setCurrentChatId(Date.now().toString());
    }

    // Store current values before clearing
    const currentImage = selectedImage;
    const currentImagePreview = imagePreview;
    const currentInputMessage = inputMessage;

    const userMessage = { 
      role: 'user', 
      content: currentInputMessage || "Please analyze this image",
      image: currentImagePreview // Store image for display
    };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    
    // Clear input and image immediately after creating the message
    setInputMessage('');
    clearImage();
    
    if (imageGenerationEnabled) {
      setIsGeneratingImage(true);
    } else {
      setIsLoading(true);
    }

    try {
      let response;
      
      console.log('Debug - currentImage:', !!currentImage, 'imageGenerationEnabled:', imageGenerationEnabled, 'webSearchEnabled:', webSearchEnabled);
      
      if (currentImage) {
        console.log('Taking image analysis path');
        // Upload image first
        const formData = new FormData();
        formData.append('file', currentImage);
        
        const uploadResponse = await apiClient.post('/upload-image', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        
        // Use the new chat-with-image endpoint for vision model
        response = await apiClient.post('/chat-with-image', {
          image_data: uploadResponse.data.image_data,
          prompt: currentInputMessage || "Describe what you see in this image in detail.",
          messages: messages.filter(msg => !msg.image) // Don't send image messages to avoid token limits
        });
      } else if (imageGenerationEnabled) {
        console.log('Taking image generation path');
        // Image generation mode
        const promptText = currentInputMessage.trim() || 'Generate a creative image';
        response = await apiClient.post('/generate-image', {
          prompt: promptText,
          messages: newMessages.filter(msg => !msg.image) // Don't send image messages to avoid token limits
        });
        
        console.log('Image generation completed successfully');
        // Auto-switch back to normal chat mode after image generation
        setImageGenerationEnabled(false);
        console.log('Auto-switched back to normal chat mode after image generation');
      } else {
        console.log('Taking normal chat path with Groq - webSearchEnabled:', webSearchEnabled);
        // Regular text chat with Llama 3.3 70B
        // Send ALL messages to maintain full conversation context and memory
        
        // Clean messages but keep all of them for full context
        const cleanedMessages = newMessages.map(msg => ({
          role: msg.role,
          content: msg.content.replace(/<[^>]*>/g, '').substring(0, 2000) // Remove HTML and limit to 2000 chars per message
        }));
        
        console.log('Sending ALL', cleanedMessages.length, 'messages to Groq for complete conversation memory');
        response = await apiClient.post('/chat', {
          messages: cleanedMessages,
          model: "llama-3.3-70b-versatile",
          use_web_search: webSearchEnabled
        });
      }

      const assistantMessage = { role: 'assistant', content: response.data.message };
      const finalMessages = [...newMessages, assistantMessage];
      setMessages(finalMessages);
      
      // Auto-save chat after first AI response (when we have 2+ messages)
      if (finalMessages.length >= 2 && currentChatId) {
        try {
          const chatTitle = generateChatTitle(finalMessages.find(m => m.role === 'user')?.content || 'New Chat');
          await apiClient.post('/save-chat', {
            chat_id: currentChatId,
            title: chatTitle,
            messages: finalMessages
          });
          
          // Mark this chat as saved to prevent duplicate saves
          setLastSavedChatId(currentChatId);
          
          // Check if this chat is already in the sidebar (multiple checks to prevent duplicates)
          const existingChatById = chatHistory.find(chat => chat.id === currentChatId);
          const existingChatByTitle = chatHistory.find(chat => chat.title === chatTitle);
          
          if (!existingChatById && !existingChatByTitle) {
            // Add the new chat directly to the chat history without needing a full refresh
            const newChat = {
              id: currentChatId,
              title: chatTitle,
              timestamp: new Date().toISOString(),
              message_count: finalMessages.length
            };
            
            // Add to the beginning of the chat history (most recent first)
            setChatHistory(prev => {
              // Double-check for duplicates in the existing array before adding
              const hasExistingId = prev.some(chat => chat.id === currentChatId);
              const hasExistingTitle = prev.some(chat => chat.title === chatTitle);
              
              if (!hasExistingId && !hasExistingTitle) {
                return [newChat, ...prev];
              }
              return prev; // Return unchanged if duplicate found
            });
          }
        } catch (error) {
          console.error('Error auto-saving chat:', error);
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      console.error('Error details:', {
        message: error.message,
        response: error.response?.data,
        status: error.response?.status,
        imageGenerationEnabled,
        webSearchEnabled
      });
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      };
      const finalMessages = [...newMessages, errorMessage];
      setMessages(finalMessages);
      
      // Don't auto-save on error - let the beforeunload handler save if needed
    } finally {
      setIsLoading(false);
      setIsGeneratingImage(false);
      
      // Auto-focus input field after message is sent
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
        }
      }, 100); // Small delay to ensure DOM is updated
    }
  };

  const clearChat = () => {
    startNewChat();
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        setSelectedImage(file);
        const reader = new FileReader();
        reader.onload = (e) => {
          setImagePreview(e.target.result);
        };
        reader.readAsDataURL(file);
      } else {
        alert('Please select an image file');
      }
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  useEffect(() => {
    // Set initial sidebar state based on screen size
    const handleResize = () => {
      // Always start with sidebar closed, regardless of screen size
      // User must manually open it with hamburger button
      if (window.innerWidth <= 768) {
        setSidebarOpen(false); // Closed on mobile
      }
      // On desktop, don't auto-open, let user control it
    };

    // Set initial state (always closed)
    setSidebarOpen(false);

    // Add resize listener
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Show loading screen while checking authentication
  if (isLoadingAuth) {
    return (
      <div className="App">
        <div className="loading-screen">
          <img src="./yota_logo.png" alt="Yota Logo" className="welcome-yota-logo" />
          <h2>Loading...</h2>
        </div>
      </div>
    );
  }

  // Show login screen if not authenticated
  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} onRegister={handleRegister} />;
  }

  return (
    <div className="App">
      {/* Sidebar */}
      <div className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-header">
          <div className="user-info">
            <span className="user-name">Welcome, {user?.username}!</span>
            <button onClick={handleLogout} className="logout-btn">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                <polyline points="16,17 21,12 16,7"/>
                <path d="M21 12H9"/>
              </svg>
              Logout
            </button>
          </div>
          <button onClick={startNewChat} className="new-chat-btn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14M5 12h14"/>
            </svg>
            New Chat
          </button>
        </div>
        
        <div className="chat-history">
          {chatHistory.length > 0 && (
            <>
              <div className="history-section">
                <h3>Recent Chats</h3>
              </div>
              {chatHistory.map((chat) => (
                <div key={chat.id} className={`history-item ${currentChatId === chat.id ? 'active' : ''}`}>
                  <div className="history-content" onClick={() => loadChat(chat)}>
                    <div className="history-title">{chat.title}</div>
                    <div className="history-time">
                      {new Date(chat.timestamp).toLocaleDateString()}
                    </div>
                  </div>
                  <button 
                    className="delete-chat-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteChat(chat.id);
                    }}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/>
                    </svg>
                  </button>
                </div>
              ))}
            </>
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="main-content">
        <div className={`chat-container ${messages.length === 0 ? 'empty-state' : ''}`}>
          <div className="chat-header">
            <button onClick={toggleSidebar} className="hamburger-btn">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="3" y1="6" x2="21" y2="6"/>
                <line x1="3" y1="12" x2="21" y2="12"/>
                <line x1="3" y1="18" x2="21" y2="18"/>
              </svg>
            </button>
            <div className="chat-header-title">
              <img src="./yota_logo.png" alt="Yota Logo" className="yota-logo" />
              <h1>Yota</h1>
            </div>
            <button onClick={clearChat} className="clear-btn">
              Clear Chat
            </button>
          </div>

          {/* Centered welcome section for empty state */}
          {messages.length === 0 && (
            <div className="centered-welcome">
              <div className="welcome-title-container">
                <img src="./yota_logo.png" alt="Yota Logo" className="welcome-yota-logo" />
                <h1>Yota AI Assistant</h1>
              </div>
              <p>How can I help you today?</p>
            </div>
          )}
          
          <div className="messages-container">
            {messages.length === 0 && (
              <div className="welcome-message">
                <h2>Hey there! What's on your mind today?</h2>
                <p>Ask me anything and I'll help you out.</p>
              </div>
            )}
            
            {messages.map((message, index) => (
              <div key={index} className={`message ${message.role}`}>
                <div className="message-content">
                  {message.image && (
                    <div className="message-image">
                      <img src={message.image} alt="User uploaded" />
                    </div>
                  )}
                  <div className="message-text">
                    {message.role === 'assistant' ? (
                      <ReactMarkdown 
                        components={markdownComponents}
                      >
                        {htmlToMarkdown(message.content)}
                      </ReactMarkdown>
                    ) : (
                      <ReactMarkdown 
                        components={markdownComponents}
                      >
                        {message.content}
                      </ReactMarkdown>
                    )}
                  </div>
                </div>
              </div>
            ))}
            
            {(isLoading || isGeneratingImage) && (
              <div className="message assistant">
                <div className="message-content">
                  <p className="typing">
                    {isGeneratingImage ? 'Crafting your image...' : 'Thinking...'}
                  </p>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
          
          <form onSubmit={sendMessage} className="input-form">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageUpload}
              accept="image/*"
              style={{ display: 'none' }}
            />
            
            {imagePreview && (
              <div className="image-preview">
                <img src={imagePreview} alt="Preview" />
                <button type="button" onClick={clearImage} className="remove-image-btn">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
              </div>
            )}
            
            <div className="input-container">
              <button type="button" className="input-action-btn attachment-btn" onClick={openFileDialog}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66L9.64 16.2a2 2 0 0 1-2.83-2.83l8.49-8.48"/>
                </svg>
              </button>
              
              <input
                ref={inputRef} // Add ref to input field
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder={selectedImage ? "Describe what you want to know about this image..." : "How can Yota help?"}
                disabled={isLoading}
                className="message-input"
              />
              
              <div className="input-actions">
                <div className="dropdown-container">
                  <button 
                    type="button" 
                    className={`input-action-btn dropdown-btn ${webSearchEnabled ? 'web-search-active' : ''}`}
                    onClick={toggleWebSearch}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="11" cy="11" r="8"/>
                      <path d="m21 21-4.35-4.35"/>
                    </svg>
                    Web Search
                    {webSearchEnabled && (
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M20 6L9 17l-5-5"/>
                      </svg>
                    )}
                  </button>
                </div>
                
                <button 
                  type="button" 
                  className={`input-action-btn think-btn ${imageGenerationEnabled ? 'web-search-active' : ''}`}
                  onClick={toggleImageGeneration}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21,15 16,10 5,21"/>
                  </svg>
                  Generate Image
                  {imageGenerationEnabled && (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M20 6L9 17l-5-5"/>
                    </svg>
                  )}
                </button>
                
                <button type="submit" disabled={isLoading || isGeneratingImage || (!inputMessage.trim() && !selectedImage)} className="send-btn">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                  </svg>
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>

      {/* Overlay for mobile */}
      {sidebarOpen && <div className="overlay" onClick={toggleSidebar}></div>}
    </div>
  );
}

export default App;
