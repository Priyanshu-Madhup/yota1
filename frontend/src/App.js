import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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
    // Convert images - preserve them as markdown
    .replace(/<img[^>]*src="([^"]*)"[^>]*alt="([^"]*)"[^>]*>/gi, '![$2]($1)')
    .replace(/<img[^>]*src="([^"]*)"[^>]*>/gi, '![Image]($1)')
    // Remove remaining HTML tags EXCEPT img tags that might not have been converted
    .replace(/<(?!img\b)[^>]+>/g, '')
    // Clean up extra whitespace
    .replace(/\n\s*\n/g, '\n\n')
    .trim();
};

// Function to check if content contains HTML that should be rendered as HTML
const shouldRenderAsHtml = (content) => {
  if (!content) return false;
  
  // Check if content contains img tags, styled divs, complex HTML, or web search results
  const htmlPatterns = [
    /<img[^>]*>/i,
    /<div[^>]*style[^>]*>/i,
    /data:image\//i,
    /<h3>.*?🎨.*?Generated Image.*?<\/h3>/i,
    // Web search patterns
    /<h3>.*?🔍.*?Search Results.*?<\/h3>/i,
    /<h3>.*?📚.*?Knowledge Graph.*?<\/h3>/i,
    /<h3>.*?📰.*?Latest News.*?<\/h3>/i,
    // General anchor tag pattern (for any content with clickable links)
    /<a[^>]*href=[^>]*target="_blank"[^>]*>/i,
    // Pattern for content with multiple HTML elements (likely formatted search results)
    /<h4>.*?<\/h4>[\s\S]*?<p>.*?<a[^>]*href/i
  ];
  
  const shouldRender = htmlPatterns.some(pattern => pattern.test(content));
  
  // Debug logging
  if (shouldRender) {
    console.log('� Content should render as HTML (includes links):', content.substring(0, 200) + '...');
  }
  
  return shouldRender;
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
  
  img: ({ src, alt }) => (
    <img 
      src={src} 
      alt={alt} 
      className="markdown-img"
      style={{
        maxWidth: '100%',
        height: 'auto',
        borderRadius: '8px',
        margin: '10px 0',
        boxShadow: '0 4px 8px rgba(0, 0, 0, 0.3)',
        display: 'block'
      }}
    />
  ),
};

// Global function to download images
window.downloadImage = (dataUrl, filename) => {
  const link = document.createElement('a');
  link.href = dataUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

function App() {
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
  const [imageEditingEnabled, setImageEditingEnabled] = useState(false);
  const [isGeneratingImage, setIsGeneratingImage] = useState(false);
  const [editingImage, setEditingImage] = useState(null);
  const [editingImagePreview, setEditingImagePreview] = useState(null);
  const [videoAnalysisEnabled, setVideoAnalysisEnabled] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoPreview, setVideoPreview] = useState(null);
  const [isAnalyzingVideo, setIsAnalyzingVideo] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const inputRef = useRef(null); // Add ref for input field

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
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

  // Load chat history from backend on mount
  useEffect(() => {
    const loadChatHistory = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/chat-history`);
        if (response.ok) {
          const data = await response.json();
          setChatHistory(data.chats);
        }
      } catch (error) {
        console.error('Error loading chat history:', error);
        // Fallback to localStorage
        const savedHistory = localStorage.getItem('chatHistory');
        if (savedHistory) {
          setChatHistory(JSON.parse(savedHistory));
        }
      }
    };
    
    loadChatHistory();
  }, []);

  // Save current chat to backend when navigating away
  useEffect(() => {
    const saveCurrentChat = async () => {
      if (messages.length > 0 && currentChatId) {
        try {
          const chatTitle = generateChatTitle(messages.find(m => m.role === 'user')?.content || 'New Chat');
          await fetch(`${API_BASE_URL}/save-chat`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              chat_id: currentChatId,
              title: chatTitle,
              messages: messages
            })
          });
        } catch (error) {
          console.error('Error saving chat:', error);
        }
      }
    };

    const handleBeforeUnload = () => {
      saveCurrentChat();
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [messages, currentChatId]);

  const generateChatTitle = (firstMessage) => {
    return firstMessage.length > 30 ? firstMessage.substring(0, 30) + '...' : firstMessage;
  };

  const startNewChat = async () => {
    // Save current chat to backend if it has messages
    if (messages.length > 0 && currentChatId) {
      try {
        const chatTitle = generateChatTitle(messages.find(m => m.role === 'user')?.content || 'New Chat');
        await fetch(`${API_BASE_URL}/save-chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            chat_id: currentChatId,
            title: chatTitle,
            messages: messages
          })
        });
        
        // Refresh chat history
        const response = await fetch(`${API_BASE_URL}/chat-history`);
        if (response.ok) {
          const data = await response.json();
          setChatHistory(data.chats);
        }
      } catch (error) {
        console.error('Error saving chat:', error);
      }
    }
    
    // Start new chat
    setMessages([]);
    setCurrentChatId(Date.now().toString());
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
      const response = await fetch(`${API_BASE_URL}/load-chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chat_id: chat.id
        })
      });
      
      if (response.ok) {
        const data = await response.json();
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
      const response = await fetch(`${API_BASE_URL}/delete-chat/${chatId}`, {
        method: 'DELETE',
      });
      
      if (response.ok) {
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

  const toggleImageEditing = () => {
    setImageEditingEnabled(!imageEditingEnabled);
    console.log('Image editing toggled:', !imageEditingEnabled);
    // Clear any selected editing image when toggling off
    if (imageEditingEnabled) {
      setEditingImage(null);
      setEditingImagePreview(null);
    }
  };

  const toggleVideoAnalysis = () => {
    setVideoAnalysisEnabled(!videoAnalysisEnabled);
    console.log('Video analysis toggled:', !videoAnalysisEnabled);
    // Clear any selected video when toggling off
    if (videoAnalysisEnabled) {
      clearVideo();
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if ((!inputMessage.trim() && !selectedImage && !editingImage && !selectedVideo) || isLoading) return;

    // Generate chat ID if this is a new conversation
    if (!currentChatId) {
      setCurrentChatId(Date.now().toString());
    }

    // Store current values before clearing
    const currentImage = selectedImage;
    const currentImagePreview = imagePreview;
    const currentEditingImage = editingImage;
    const currentEditingImagePreview = editingImagePreview;
    const currentVideo = selectedVideo;
    const currentVideoPreview = videoPreview;
    const currentInputMessage = inputMessage;

    // Create user message with appropriate media preview
    let userMessage;
    if (videoAnalysisEnabled && currentVideo) {
      userMessage = { 
        role: 'user', 
        content: currentInputMessage || "Analyze this video",
        video: currentVideoPreview // Store video for display
      };
    } else if (imageEditingEnabled && currentEditingImage) {
      userMessage = { 
        role: 'user', 
        content: currentInputMessage || "Edit this image",
        image: currentEditingImagePreview // Store editing image for display
      };
    } else {
      userMessage = { 
        role: 'user', 
        content: currentInputMessage || "Please analyze this image",
        image: currentImagePreview // Store regular image for display
      };
    }
    
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    
    // Clear input and media immediately after creating the message
    setInputMessage('');
    clearImage();
    clearEditingImage(); // Also clear editing image
    clearVideo(); // Also clear video
    
    if (imageGenerationEnabled) {
      setIsGeneratingImage(true);
    } else if (videoAnalysisEnabled) {
      setIsAnalyzingVideo(true);
    } else {
      setIsLoading(true);
    }

    try {
      let response;
      
      console.log('Debug - currentVideo:', !!currentVideo, 'currentImage:', !!currentImage, 'imageGenerationEnabled:', imageGenerationEnabled, 'videoAnalysisEnabled:', videoAnalysisEnabled, 'webSearchEnabled:', webSearchEnabled);
      
      if (videoAnalysisEnabled && currentVideo) {
        console.log('Taking video analysis path');
        // Upload video first
        const formData = new FormData();
        formData.append('file', currentVideo);
        
        const uploadResponse = await axios.post(`${API_BASE_URL}/upload-video`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        
        // Use the video analysis endpoint
        response = await axios.post(`${API_BASE_URL}/analyze-video`, {
          video_data: uploadResponse.data.video_data,
          prompt: currentInputMessage || "Analyze and summarize this video in detail.",
          messages: messages.filter(msg => !msg.image && !msg.video) // Don't send media messages to avoid token limits
        });
        
        console.log('Video analysis completed successfully');
        // Auto-switch back to normal chat mode after video analysis
        setVideoAnalysisEnabled(false);
        console.log('Auto-switched back to normal chat mode after video analysis');
      } else if (currentImage) {
        console.log('Taking image analysis path');
        // Upload image first
        const formData = new FormData();
        formData.append('file', currentImage);
        
        const uploadResponse = await axios.post(`${API_BASE_URL}/upload-image`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });
        
        // Use the new chat-with-image endpoint for vision model
        response = await axios.post(`${API_BASE_URL}/chat-with-image`, {
          image_data: uploadResponse.data.image_data,
          prompt: currentInputMessage || "Describe what you see in this image in detail.",
          messages: messages.filter(msg => !msg.image) // Don't send image messages to avoid token limits
        });
      } else if (imageEditingEnabled && currentEditingImage) {
        console.log('Taking image editing path');
        // Image editing mode - send the selected editing image along with prompt
        const promptText = currentInputMessage.trim() || 'Edit this image creatively';
        response = await axios.post(`${API_BASE_URL}/edit-image`, {
          image_data: currentEditingImage, // base64 image data
          prompt: promptText,
          messages: newMessages.filter(msg => !msg.image) // Don't send image messages to avoid token limits
        });
        
        console.log('Image editing completed successfully');
        // Auto-switch back to normal chat mode after image editing
        setImageEditingEnabled(false);
        console.log('Auto-switched back to normal chat mode after image editing');
      } else if (imageGenerationEnabled) {
        console.log('Taking image generation path');
        // Image generation mode
        const promptText = currentInputMessage.trim() || 'Generate a creative image';
        response = await axios.post(`${API_BASE_URL}/generate-image`, {
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
        response = await axios.post(`${API_BASE_URL}/chat`, {
          messages: cleanedMessages,
          model: "llama-3.3-70b-versatile",
          use_web_search: webSearchEnabled
        });
      }

      const assistantMessage = { role: 'assistant', content: response.data.message };
      const finalMessages = [...newMessages, assistantMessage];
      
      // Debug logging for image generation responses
      if (imageGenerationEnabled) {
        console.log('🖼️ Image generation response received:', {
          contentLength: response.data.message.length,
          contentPreview: response.data.message.substring(0, 300),
          containsImg: response.data.message.includes('<img'),
          containsBase64: response.data.message.includes('data:image')
        });
      }
      
      setMessages(finalMessages);
      
      // Auto-save chat after first AI response (when we have 2+ messages)
      if (finalMessages.length >= 2 && currentChatId) {
        try {
          const chatTitle = generateChatTitle(finalMessages.find(m => m.role === 'user')?.content || 'New Chat');
          await fetch(`${API_BASE_URL}/save-chat`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              chat_id: currentChatId,
              title: chatTitle,
              messages: finalMessages
            })
          });
          
          // Refresh chat history to show the new chat immediately
          const historyResponse = await fetch(`${API_BASE_URL}/chat-history`);
          if (historyResponse.ok) {
            const data = await historyResponse.json();
            setChatHistory(data.chats);
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
      
      // Auto-save chat even on error if it's the first response
      if (finalMessages.length >= 2 && currentChatId) {
        try {
          const chatTitle = generateChatTitle(finalMessages.find(m => m.role === 'user')?.content || 'New Chat');
          await fetch(`${API_BASE_URL}/save-chat`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              chat_id: currentChatId,
              title: chatTitle,
              messages: finalMessages
            })
          });
          
          // Refresh chat history
          const historyResponse = await fetch(`${API_BASE_URL}/chat-history`);
          if (historyResponse.ok) {
            const data = await historyResponse.json();
            setChatHistory(data.chats);
          }
        } catch (saveError) {
          console.error('Error auto-saving chat:', saveError);
        }
      }
    } finally {
      setIsLoading(false);
      setIsGeneratingImage(false);
      setIsAnalyzingVideo(false);
      
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
        // Clear video selection if any
        clearVideo();
      } else if (file.type.startsWith('video/')) {
        setSelectedVideo(file);
        const reader = new FileReader();
        reader.onload = (e) => {
          setVideoPreview(e.target.result);
        };
        reader.readAsDataURL(file);
        // Clear image selection if any
        clearImage();
      } else {
        alert('Please select an image or video file');
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

  const clearVideo = () => {
    setSelectedVideo(null);
    setVideoPreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleEditingImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        // Convert to base64 for editing
        const reader = new FileReader();
        reader.onload = (e) => {
          const base64Data = e.target.result.split(',')[1]; // Remove data:image/png;base64, prefix
          setEditingImage(base64Data);
          setEditingImagePreview(e.target.result);
        };
        reader.readAsDataURL(file);
      } else {
        alert('Please select an image file for editing');
      }
    }
  };

  const clearEditingImage = () => {
    setEditingImage(null);
    setEditingImagePreview(null);
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

  return (
    <div className="App">
      {/* Sidebar */}
      <div className={`sidebar ${sidebarOpen ? 'sidebar-open' : ''}`}>
        <div className="sidebar-header">
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
                      <img src={message.image} alt="Uploaded content" />
                    </div>
                  )}
                  {message.video && (
                    <div className="message-video">
                      <video src={message.video} controls style={{ maxWidth: '300px', height: 'auto' }} />
                    </div>
                  )}
                  <div className="message-text">
                    {message.role === 'assistant' ? (
                      shouldRenderAsHtml(message.content) ? (
                        <div 
                          className="html-content"
                          dangerouslySetInnerHTML={{ __html: message.content }}
                        />
                      ) : (
                        <ReactMarkdown 
                          components={markdownComponents}
                        >
                          {htmlToMarkdown(message.content)}
                        </ReactMarkdown>
                      )
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
            
            {(isLoading || isGeneratingImage || isAnalyzingVideo) && (
              <div className="message assistant">
                <div className="message-content">
                  <p className="typing">
                    {isGeneratingImage ? 'Crafting your image...' : 
                     isAnalyzingVideo ? 'Analyzing your video...' : 'Thinking...'}
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
              accept="image/*,video/*"
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

            {videoPreview && (
              <div className="video-preview">
                <video src={videoPreview} controls style={{ maxWidth: '150px', maxHeight: '150px' }} />
                <button type="button" onClick={clearVideo} className="remove-video-btn">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
              </div>
            )}

            {/* Image Editing Section */}
            {imageEditingEnabled && (
              <div className="image-editing-section">
                <div className="editing-controls">
                  <label className="edit-image-upload-btn">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                      <circle cx="8.5" cy="8.5" r="1.5"/>
                      <polyline points="21,15 16,10 5,21"/>
                    </svg>
                    Select Image to Edit
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleEditingImageUpload}
                      style={{ display: 'none' }}
                    />
                  </label>
                </div>

                {editingImagePreview && (
                  <div className="editing-image-preview">
                    <img src={editingImagePreview} alt="Preview for editing" />
                    <button type="button" onClick={clearEditingImage} className="remove-image-btn">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18"/>
                        <line x1="6" y1="6" x2="18" y2="18"/>
                      </svg>
                    </button>
                  </div>
                )}
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
                placeholder={
                  selectedVideo ? "Describe what you want to know about this video..." :
                  selectedImage ? "Describe what you want to know about this image..." :
                  imageEditingEnabled && editingImage ? "Describe how you want to edit this image..." :
                  imageGenerationEnabled ? "Describe the image you want to generate..." :
                  videoAnalysisEnabled ? "Upload a video to analyze..." :
                  "How can Yota help?"
                }
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
                
                <button 
                  type="button" 
                  className={`input-action-btn think-btn ${imageEditingEnabled ? 'web-search-active' : ''}`}
                  onClick={toggleImageEditing}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <path d="M12 8v8M8 12h8"/>
                    <path d="M3 12a9 9 0 1 0 18 0 9 9 0 1 0-18 0"/>
                  </svg>
                  Edit Images
                  {imageEditingEnabled && (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M20 6L9 17l-5-5"/>
                    </svg>
                  )}
                </button>
                
                <button 
                  type="button" 
                  className={`input-action-btn think-btn ${videoAnalysisEnabled ? 'web-search-active' : ''}`}
                  onClick={toggleVideoAnalysis}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polygon points="23 7 16 12 23 17 23 7"/>
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                  </svg>
                  Analyze Video
                  {videoAnalysisEnabled && (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M20 6L9 17l-5-5"/>
                    </svg>
                  )}
                </button>
                
                <button type="submit" disabled={
                  isLoading || isGeneratingImage || isAnalyzingVideo || 
                  (!inputMessage.trim() && !selectedImage && !selectedVideo && !(imageEditingEnabled && editingImage))
                } className="send-btn">
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
