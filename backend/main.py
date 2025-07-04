import os
import base64
import http.client
import json
import logging
import pickle
import uuid
import tempfile
import traceback
import uvicorn
import PIL.Image
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv
from PIL import Image
from datetime import datetime

# Conditional imports for RAG functionality with enhanced error handling
try:
    import faiss
    import numpy as np
    print("✅ FAISS library loaded successfully")
    FAISS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  FAISS not available: {e}")
    faiss = None
    np = None
    FAISS_AVAILABLE = False

try:
    import cohere
    print("✅ Cohere library loaded successfully")
    COHERE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Cohere not available: {e}")
    cohere = None
    COHERE_AVAILABLE = False

# RAG is available if we have both FAISS and Cohere
RAG_AVAILABLE = FAISS_AVAILABLE and COHERE_AVAILABLE

if RAG_AVAILABLE:
    print("🧠 RAG system fully available with Cohere embeddings")
else:
    print("📝 Using simple conversation memory system")

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot API", version="1.0.0")

# CORS middleware to allow React frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# RAG Chat Storage System with Cohere embeddings
class ConversationRAG:
    """
    RAG system for conversation memory using FAISS + Cohere embeddings.
    Maintains chat history functionality while using Cohere for embeddings.
    """
    
    def __init__(self, max_conversations=1000, embedding_model="embed-v4.0"):
        self.max_conversations = max_conversations
        self.conversations = []  # List of conversation dictionaries
        self.chat_titles = {}  # Store chat titles
        self.memory_file = "conversation_memory.pkl"
        self.index_file = "conversation_faiss.index"
        self.rag_enabled = False
        
        # Cohere embedding configuration
        self.embedding_model = embedding_model
        self.cohere_client = None
        
        # FAISS index configuration
        self.index = None
        self.dimension = None
        self.embeddings_list = []
        
        # Initialize the system
        self._initialize_rag_system()
        self._load_memory()
    
    def _initialize_rag_system(self):
        """Initialize RAG system with Cohere embeddings"""
        try:
            if RAG_AVAILABLE:
                # Initialize Cohere client
                cohere_api_key = os.environ.get("COHERE_API_KEY")
                if not cohere_api_key:
                    raise ValueError("COHERE_API_KEY environment variable is required")
                self.cohere_client = cohere.ClientV2(
                    api_key=cohere_api_key
                )
                self.rag_enabled = True
                print("🚀 Using Cohere-based RAG system")
            else:
                print("⚠️  Cohere or FAISS not available, using simple memory")
                self.rag_enabled = False
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.rag_enabled = False
    
    def _get_cohere_embeddings(self, texts: List[str], input_type: str = "search_document"):
        """Get embeddings from Cohere API"""
        try:
            if not self.cohere_client:
                return np.array([])
            
            response = self.cohere_client.embed(
                texts=texts,
                model=self.embedding_model,
                input_type=input_type,
                embedding_types=["float"],
            )
            
            # Extract embeddings from response
            embeddings = []
            for embedding_response in response.embeddings.float:
                embeddings.append(embedding_response)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"Error getting Cohere embeddings: {e}")
            return np.array([])
    
    def _initialize_faiss_index(self, dimension: int):
        """Initialize FAISS index with given dimension"""
        try:
            if not FAISS_AVAILABLE:
                return False
            
            self.dimension = dimension
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            logger.info(f"Initialized FAISS index with dimension: {dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            return False
    
    def add_conversation(self, user_input: str, assistant_response: str, chat_id: str, chat_title: str = None):
        """Add a conversation to memory with RAG indexing using Cohere embeddings"""
        try:
            conversation = {
                'user_input': user_input,
                'assistant_response': assistant_response,
                'chat_id': chat_id,
                'timestamp': datetime.now().isoformat(),
                'id': len(self.conversations)
            }
            
            # Store chat title
            if chat_title:
                self.chat_titles[chat_id] = chat_title
            
            # Add to conversations list
            self.conversations.append(conversation)
            
            # Create embedding for this conversation using Cohere
            if self.rag_enabled and self.cohere_client:
                conversation_text = f"User: {user_input} Assistant: {assistant_response}"
                embeddings = self._get_cohere_embeddings([conversation_text], input_type="search_document")
                
                if embeddings.size > 0:
                    # Initialize FAISS index if not already done
                    if self.index is None:
                        self._initialize_faiss_index(embeddings.shape[1])
                    
                    # Normalize embeddings for cosine similarity
                    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                    
                    # Add to FAISS index
                    if self.index is not None:
                        self.index.add(normalized_embeddings.astype('float32'))
                        self.embeddings_list.append(normalized_embeddings[0])
            
            # Cleanup old conversations if needed
            self._cleanup_old_conversations()
            
            # Save to disk
            self._save_memory_to_disk()
            
            logger.info(f"Added conversation to RAG memory. Total: {len(self.conversations)}")
            
        except Exception as e:
            logger.error(f"Error adding conversation: {e}")
    

    def search_relevant_conversations(self, query: str, chat_id: str = None, top_k: int = 4):
        """Search for relevant conversations using Cohere embeddings and FAISS"""
        try:
            if not self.rag_enabled or not self.index or len(self.conversations) == 0:
                logger.info(f"RAG search skipped: enabled={self.rag_enabled}, index={self.index is not None}, conversations={len(self.conversations)}")
                return []
            
            # Get embedding for query using Cohere
            query_embeddings = self._get_cohere_embeddings([query], input_type="search_query")
            
            if query_embeddings.size == 0:
                logger.warning("Failed to get query embedding, falling back to keyword search")
                return self._fallback_keyword_search(query, chat_id, top_k)
            
            # Normalize query embedding
            query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            
            # Search FAISS index
            k = min(top_k * 2, len(self.conversations))  # Search for more than needed
            similarities, indices = self.index.search(query_embeddings.astype('float32'), k)
            
            logger.info(f"FAISS search for '{query}': Found {len(indices[0])} results")
            
            relevant_conversations = []
            for i, idx in enumerate(indices[0]):
                if idx >= len(self.conversations):
                    continue
                
                conversation = self.conversations[idx]
                
                # Filter by chat_id if specified
                if chat_id and conversation.get('chat_id') != chat_id:
                    continue
                
                # Use similarity score directly (higher is better with inner product)
                similarity = float(similarities[0][i])
                
                logger.info(f"Conversation {idx}: similarity={similarity:.4f}, text='{conversation['user_input'][:50]}...'")
                
                relevant_conversations.append({
                    'conversation': conversation,
                    'score': similarity,
                    'index': idx
                })
                
                if len(relevant_conversations) >= top_k:
                    break
            
            # Sort by similarity (highest first)
            relevant_conversations.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Returning {len(relevant_conversations)} relevant conversations")
            return relevant_conversations[:top_k]
            
        except Exception as e:
            logger.error(f"Error in FAISS search: {e}")
            return self._fallback_keyword_search(query, chat_id, top_k)
    
    def _fallback_keyword_search(self, query: str, chat_id: str = None, top_k: int = 4):
        """Fallback keyword search when FAISS fails"""
        try:
            query_words = query.lower().split()
            relevant_conversations = []
            
            for idx, conversation in enumerate(self.conversations):
                # Filter by chat_id if specified
                if chat_id and conversation.get('chat_id') != chat_id:
                    continue
                
                # Simple keyword matching
                text = f"{conversation['user_input']} {conversation['assistant_response']}".lower()
                matches = sum(1 for word in query_words if word in text)
                
                if matches > 0:
                    score = matches / len(query_words)
                    relevant_conversations.append({
                        'conversation': conversation,
                        'score': score,
                        'index': idx
                    })
            
            # Sort by score and return top k
            relevant_conversations.sort(key=lambda x: x['score'], reverse=True)
            return relevant_conversations[:top_k]
            
        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            return []

    def get_context_for_prompt(self, current_query: str, chat_id: str = None, max_context_length: int = 1500) -> str:
        """Get relevant context for the current query with improved personal information recall"""
        try:
            # Enhanced personal information detection
            personal_keywords = [
                'name', 'called', 'who am i', 'my name', 'remember', 'told you', 'what is my',
                'you know', 'i am', 'call me', 'i told you', 'mentioned', 'said my name',
                'what do you know about me', 'who is', 'about me', 'i said', 'earlier'
            ]
            
            is_personal_query = any(keyword in current_query.lower() for keyword in personal_keywords)
            
            logger.info(f"Query analysis: '{current_query}' - Personal query: {is_personal_query}")
            
            # For personal queries, search more comprehensively
            if is_personal_query:
                # First, try semantic search with higher top_k
                relevant_convos = self.search_relevant_conversations(current_query, chat_id, top_k=10)
                
                # If no good results, search for personal information patterns
                if not relevant_convos or (relevant_convos and max([conv['score'] for conv in relevant_convos]) < 0.3):
                    logger.info("Low relevance scores for personal query, searching for personal information patterns...")
                    personal_convos = self._search_for_personal_info(chat_id)
                    if personal_convos:
                        relevant_convos = personal_convos[:8]  # Take top 8 personal info conversations
                        logger.info(f"Found {len(relevant_convos)} conversations with personal information")
                    
                # Also search for any conversations with proper nouns (names, places, etc.)
                if not relevant_convos:
                    logger.info("No personal info found, searching for proper nouns...")
                    noun_convos = self._search_for_proper_nouns(chat_id)
                    if noun_convos:
                        relevant_convos = noun_convos[:6]
                        logger.info(f"Found {len(relevant_convos)} conversations with proper nouns")
            else:
                # For non-personal queries, use standard search
                relevant_convos = self.search_relevant_conversations(current_query, chat_id, top_k=4)
            
            # Build context with priority for personal information if it's a personal query
            context = self._build_context_string(relevant_convos, max_context_length, prioritize_personal=is_personal_query)
            
            if context:
                logger.info(f"Built context: {len(context)} characters")
                if is_personal_query:
                    logger.info(f"Personal query context preview: {context[:200]}...")
            else:
                logger.info("No context found")
            
            return context
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""
    
    def _search_for_personal_info(self, chat_id: str = None) -> List[Dict]:
        """Search for conversations containing personal information"""
        try:
            personal_patterns = [
                'name is', 'i am', 'call me', 'my name', 'i\'m', 'i work', 'i live', 'i like',
                'i love', 'i hate', 'i enjoy', 'i prefer', 'i studied', 'i graduated',
                'i was born', 'i grew up', 'my job', 'my hobby', 'my favorite'
            ]
            
            personal_convos = []
            
            for i, conv in enumerate(self.conversations):
                if chat_id and conv.get('chat_id') != chat_id:
                    continue
                
                text = f"{conv['user_input']} {conv['assistant_response']}".lower()
                
                # Check for personal information patterns
                score = 0
                for pattern in personal_patterns:
                    if pattern in text:
                        score += 1
                
                # Check for name-like patterns (capitalized words)
                words = conv['user_input'].split()
                for word in words:
                    if word.istitle() and word.isalpha() and len(word) > 2:
                        score += 0.5
                
                if score > 0:
                    personal_convos.append({
                        'conversation': conv,
                        'score': score,
                        'index': i
                    })
            
            # Sort by score descending
            personal_convos.sort(key=lambda x: x['score'], reverse=True)
            return personal_convos
            
        except Exception as e:
            logger.error(f"Error searching for personal info: {e}")
            return []
    
    def _search_for_proper_nouns(self, chat_id: str = None) -> List[Dict]:
        """Search for conversations containing proper nouns (names, places, etc.)"""
        try:
            noun_convos = []
            
            for i, conv in enumerate(self.conversations):
                if chat_id and conv.get('chat_id') != chat_id:
                    continue
                
                # Look for proper nouns in user input
                words = conv['user_input'].split()
                proper_nouns = [word for word in words if word.istitle() and word.isalpha() and len(word) > 2]
                
                if proper_nouns:
                    noun_convos.append({
                        'conversation': conv,
                        'score': len(proper_nouns) * 0.5,  # Score based on number of proper nouns
                        'index': i
                    })
            
            # Sort by score descending
            noun_convos.sort(key=lambda x: x['score'], reverse=True)
            return noun_convos
            
        except Exception as e:
            logger.error(f"Error searching for proper nouns: {e}")
            return []
    
    def _build_context_string(self, conversations: List[Dict], max_length: int, prioritize_personal: bool = False) -> str:
        """Build context string from relevant conversations with optional personal info prioritization"""
        try:
            if not conversations:
                return ""
            
            # If prioritizing personal info, sort conversations by personal relevance
            if prioritize_personal:
                conversations = self._sort_by_personal_relevance(conversations)
            
            context_parts = []
            current_length = 0
            
            for conv_data in conversations:
                conversation = conv_data['conversation']
                user_input = conversation['user_input'][:300]  # Increased limit for better context
                assistant_response = conversation['assistant_response'][:300]
                
                # Format conversation with clear markers
                context_part = f"[Previous conversation]\nUser: {user_input}\nAssistant: {assistant_response}\n"
                
                # Check if adding this would exceed max length
                if current_length + len(context_part) > max_length:
                    break
                
                context_parts.append(context_part)
                current_length += len(context_part)
            
            if context_parts:
                return "\n".join(context_parts)
            else:
                return ""
        except Exception as e:
            logger.error(f"Error building context string: {e}")
            return ""
    
    def _sort_by_personal_relevance(self, conversations: List[Dict]) -> List[Dict]:
        """Sort conversations by personal information relevance"""
        try:
            def get_personal_score(conv_data):
                conv = conv_data['conversation']
                text = f"{conv['user_input']} {conv['assistant_response']}".lower()
                
                # Personal information indicators
                personal_indicators = [
                    'name is', 'i am', 'call me', 'my name', 'i\'m', 'i work', 'i live', 'i like',
                    'i love', 'i hate', 'i enjoy', 'i prefer', 'i studied', 'i graduated',
                    'i was born', 'i grew up', 'my job', 'my hobby', 'my favorite'
                ]
                
                score = conv_data.get('score', 0)  # Start with existing score
                
                # Add points for personal indicators
                for indicator in personal_indicators:
                    if indicator in text:
                        score += 2
                
                # Add points for proper nouns (potential names)
                words = conv['user_input'].split()
                for word in words:
                    if word.istitle() and word.isalpha() and len(word) > 2:
                        score += 1
                
                return score
            
            # Sort by personal relevance score
            return sorted(conversations, key=get_personal_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error sorting by personal relevance: {e}")
            return conversations

    def _cleanup_old_conversations(self):
        """Remove old conversations if we exceed max limit"""
        try:
            if len(self.conversations) > self.max_conversations:
                # Remove oldest conversations
                remove_count = len(self.conversations) - self.max_conversations
                self.conversations = self.conversations[remove_count:]
                self.embeddings_list = self.embeddings_list[remove_count:]
                
                # Rebuild FAISS index from remaining embeddings
                if self.embeddings_list and self.rag_enabled:
                    self._rebuild_index_from_embeddings()
                
        except Exception as e:
            logger.error(f"Error cleaning up conversations: {e}")

    def _save_memory_to_disk(self):
        """Save conversation memory to disk"""
        try:
            # Save conversations and metadata
            memory_data = {
                'conversations': self.conversations,
                'chat_titles': self.chat_titles,
                'embeddings_list': self.embeddings_list,
                'rag_enabled': self.rag_enabled
            }
            
            with open(self.memory_file, 'wb') as f:
                pickle.dump(memory_data, f)
            
            # Save FAISS index if available
            if self.rag_enabled and self.index is not None and FAISS_AVAILABLE:
                faiss.write_index(self.index, self.index_file)
                
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def _load_memory_from_disk(self):
        """Load conversation memory from disk"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                    
                self.conversations = memory_data.get('conversations', [])
                self.chat_titles = memory_data.get('chat_titles', {})
                self.embeddings_list = memory_data.get('embeddings_list', [])
                
                logger.info(f"Loaded {len(self.conversations)} conversations from disk")
            
            # Load FAISS index if available
            if self.rag_enabled and os.path.exists(self.index_file) and FAISS_AVAILABLE:
                try:
                    self.index = faiss.read_index(self.index_file)
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} entries")
                except Exception as e:
                    logger.error(f"Error loading FAISS index: {e}")
                    # Rebuild index from embeddings if needed
                    if self.embeddings_list and self.rag_enabled:
                        self._rebuild_index_from_embeddings()
                    
        except Exception as e:
            logger.error(f"Error loading memory: {e}")
    
    def _rebuild_index_from_embeddings(self):
        """Rebuild FAISS index from stored embeddings"""
        try:
            if not self.embeddings_list or not FAISS_AVAILABLE:
                return
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(self.embeddings_list).astype('float32')
            
            # Initialize FAISS index
            dimension = embeddings_array.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.index.add(embeddings_array)
            
            logger.info(f"Rebuilt FAISS index with {len(self.embeddings_list)} embeddings, dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {e}")
    
    def _load_memory(self):
        """Load memory and restore vector store"""
        self._load_memory_from_disk()
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about the memory system"""
        return {
            'total_conversations': len(self.conversations),
            'rag_enabled': self.rag_enabled,
            'has_faiss_index': self.index is not None,
            'index_total': self.index.ntotal if self.index and FAISS_AVAILABLE else 0,
            'embeddings_count': len(self.embeddings_list)
        }

    def get_chat_history_list(self):
        """Get list of chats with metadata"""
        try:
            chats = {}
            for conversation in self.conversations:
                chat_id = conversation.get('chat_id')
                if not chat_id:
                    continue
                    
                if chat_id not in chats:
                    chats[chat_id] = {
                        'id': chat_id,
                        'title': self.chat_titles.get(chat_id, 'Untitled Chat'),
                        'timestamp': conversation['timestamp'],
                        'message_count': 0
                    }
                chats[chat_id]['message_count'] += 1
                
                # Update timestamp to latest
                if conversation['timestamp'] > chats[chat_id]['timestamp']:
                    chats[chat_id]['timestamp'] = conversation['timestamp']
            
            # Convert to list and sort by timestamp
            chat_list = list(chats.values())
            chat_list.sort(key=lambda x: x['timestamp'], reverse=True)
            return chat_list
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    def get_chat_messages(self, chat_id: str):
        """Get all messages for a specific chat"""
        try:
            messages = []
            for conversation in self.conversations:
                if conversation.get('chat_id') == chat_id:
                    messages.append({
                        'role': 'user',
                        'content': conversation['user_input']
                    })
                    messages.append({
                        'role': 'assistant', 
                        'content': conversation['assistant_response']
                    })
            return messages
            
        except Exception as e:
            logger.error(f"Error getting chat messages: {e}")
            return []
    
    def delete_chat(self, chat_id: str):
        """Delete all conversations for a specific chat"""
        try:
            # Find indices to remove
            indices_to_remove = []
            for i, conv in enumerate(self.conversations):
                if conv.get('chat_id') == chat_id:
                    indices_to_remove.append(i)
            
            # Remove in reverse order to maintain indices
            for idx in reversed(indices_to_remove):
                del self.conversations[idx]
                if idx < len(self.embeddings_list):
                    del self.embeddings_list[idx]
            
            # Remove chat title
            if chat_id in self.chat_titles:
                del self.chat_titles[chat_id]
            
            # Rebuild FAISS index from remaining embeddings
            if self.embeddings_list and self.rag_enabled:
                self._rebuild_index_from_embeddings()
            self._save_memory_to_disk()
            
        except Exception as e:
            logger.error(f"Error deleting chat: {e}")
    
    def save_memory(self):
        """Public method to save memory"""
        self._save_memory_to_disk()

# Initialize RAG storage
conversation_rag = ConversationRAG()

# Pydantic models for request/response
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "llama-3.3-70b-versatile"
    use_web_search: bool = False
    chat_id: Optional[str] = None  # Add chat_id parameter

class ChatResponse(BaseModel):
    message: str

class ImageAnalysisRequest(BaseModel):
    image_data: str  # base64 encoded image
    prompt: str = "Describe what you see in this image in detail."
    messages: List[Message] = []

class ImageGenerationRequest(BaseModel):
    prompt: str
    messages: List[Message] = []

class ImageEditRequest(BaseModel):
    image_data: str  # base64 encoded image to edit
    prompt: str  # editing instructions
    messages: List[Message] = []

class VideoAnalysisRequest(BaseModel):
    video_data: str  # base64 encoded video
    prompt: str = "Analyze and summarize this video in detail."
    messages: List[Message] = []

# Additional Pydantic models for chat management
class SaveChatRequest(BaseModel):
    chat_id: str
    title: str
    messages: List[Message]

class LoadChatRequest(BaseModel):
    chat_id: str

class ChatHistoryItem(BaseModel):
    id: str
    title: str
    timestamp: str
    message_count: int

class ChatHistoryResponse(BaseModel):
    chats: List[ChatHistoryItem]

class LoadChatResponse(BaseModel):
    chat_id: str
    title: str
    messages: List[Message]

@app.get("/")
async def root():
    return {"message": "Chatbot API is running with RAG-powered chat storage"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Convert Pydantic messages to dict format for Groq API
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Get the latest user message for RAG retrieval
        latest_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                latest_user_message = msg["content"]
                break
        
        # Use new ConversationRAG to get relevant context
        context_from_rag = ""
        if latest_user_message:
            # Use chat_id for context if provided, otherwise search all conversations
            context_chat_id = request.chat_id if request.chat_id else None
            context_from_rag = conversation_rag.get_context_for_prompt(latest_user_message, chat_id=context_chat_id)
            if context_from_rag:
                logger.info(f"Retrieved context from RAG: {len(context_from_rag)} characters")
                logger.info(f"RAG Context: {context_from_rag[:500]}...")  # Log first 500 chars for debugging
            else:
                logger.info(f"No RAG context found for query: '{latest_user_message}'")
                # Log some debug info
                logger.info(f"Total conversations in memory: {len(conversation_rag.conversations)}")
                if conversation_rag.conversations:
                    recent_conv = conversation_rag.conversations[-1]
                    logger.info(f"Most recent conversation: User: '{recent_conv['user_input']}' Assistant: '{recent_conv['assistant_response'][:100]}...'")
        
        # If web search is enabled, search for the latest user message
        web_search_performed = False
        web_search_content = ""
        if request.use_web_search and latest_user_message:
            # Perform web search
            search_results = search_web(latest_user_message)
            web_search_content = f"CURRENT WEB SEARCH RESULTS:\n\n{search_results}\n\nBased on these real-time search results above, provide a comprehensive response about '{latest_user_message}'. Use the information from the search results and present it using proper HTML formatting with <h3> headings, <p> paragraphs, <ul><li> lists, and <strong> emphasis. Include the clickable links from the search results in your response."
            web_search_performed = True
        
        # Create context-aware messages for Groq
        context_messages = []
        
        # Add system message
        system_content = "You are an expert AI assistant named Yota that provides comprehensive, well-structured responses using HTML formatting. For most responses, focus on using subheadings and clean structure:\n\n- <h3>Subheadings</h3> for main sections (use these primarily)\n- <h4>Minor headings</h4> for subsections when needed\n- <ul><li>Bullet points</li></ul> for lists\n- <ol><li>Numbered lists</li></ol> for sequential information\n- <strong>Bold text</strong> for emphasis\n- <a href='url' target='_blank'>Clickable links</a> when referencing sources\n- <p>Paragraphs</p> with proper spacing\n\nFOR CODE BLOCKS: Always use proper HTML formatting:\n- Use <pre><code class='language-python'>your code here</code></pre> for multi-line code blocks\n- Use <code>inline code</code> for single-line code or variables\n- Always preserve proper indentation and line breaks in code\n- Include language specification (python, javascript, etc.) in the class attribute\n\nExample of proper code formatting:\n<pre><code class='language-python'>\ndef hello_world():\n    print('Hello, World!')\n    return True\n</code></pre>\n\nNEVER put code in regular paragraphs or compress it into single lines. Always use proper code block formatting for better readability and copy functionality.\n\nOnly use <h2> for very major topics or complex responses that need large section breaks. Keep responses engaging, informative, and visually appealing with clean HTML structure."
        context_messages.append({"role": "system", "content": system_content})
        
        # Add relevant context from RAG if available
        if context_from_rag:
            context_messages.append({"role": "system", "content": f"RELEVANT CONVERSATION CONTEXT:\n{context_from_rag}"})
        
        # Add web search results if available
        if web_search_content:
            context_messages.append({"role": "system", "content": web_search_content})
        
        # Add only the current user message (not full history)
        if latest_user_message:
            context_messages.append({"role": "user", "content": latest_user_message})
        
        logger.info(f"Sending {len(context_messages)} messages to Groq (ConversationRAG-optimized)")
        
        # Make API call to Groq with context-optimized messages
        chat_completion = client.chat.completions.create(
            messages=context_messages,
            model=request.model,
            max_tokens=2000,
            temperature=0.9
        )
        
        response_content = chat_completion.choices[0].message.content
        
        # Note: Conversation storage is handled by the frontend via /save-chat endpoint
        # This prevents duplicate "Quick Chat" entries
        
        return ChatResponse(message=response_content)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.post("/chat-with-image", response_model=ChatResponse)
async def chat_with_image(request: ImageAnalysisRequest):
    try:
        # Use Groq's vision model for image analysis
        messages = [
            {
                "role": "system",
                "content": "You are an expert AI assistant that provides comprehensive, well-structured responses using HTML formatting. For image analysis, you MUST use proper HTML tags:\n\n- Use <h3>Main Section</h3> for major sections like 'Visual Elements', 'Character Details', 'Scene Analysis'\n- Use <h4>Subsection</h4> for minor headings like 'Color Scheme', 'Lighting', 'Costume'\n- Use <ul><li>Point</li></ul> for bullet lists\n- Use <ol><li>Item</li></ol> for numbered lists\n- Use <strong>text</strong> for emphasis, NOT **text**\n- Use <p>paragraph text</p> for regular paragraphs\n\nNEVER use markdown formatting like ** or ## or ###. ALWAYS use proper HTML tags. Provide detailed, engaging image analysis with clean HTML structure."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": request.prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{request.image_data}",
                        },
                    },
                ],
            }
        ]
        
        # Add conversation history if provided (text only to avoid token limits)
        if request.messages:
            text_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            # Insert text messages before the current image analysis request
            messages = [messages[0]] + text_messages + [messages[1]]
        
        # Use Llama Maverick 4 (vision model)
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            max_tokens=1500,
            temperature=0.9
        )
        
        response_content = chat_completion.choices[0].message.content
        return ChatResponse(message=response_content)
        
    except Exception as e:
        logger.error(f"Error in image chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image chat: {str(e)}")

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        
        # Convert to PIL Image for processing
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image if too large (max 1024x1024)
        max_size = 1024
        if image.width > max_size or image.height > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        processed_image_data = buffer.getvalue()
        
        # Encode to base64
        base64_image = base64.b64encode(processed_image_data).decode('utf-8')
        
        return {"image_data": base64_image, "filename": file.filename}
        
    except Exception as e:
        logger.error(f"Error processing image upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Read video data
        video_data = await file.read()
        
        # Check file size (limit to 50MB for practical purposes)
        max_size = 50 * 1024 * 1024  # 50MB
        if len(video_data) > max_size:
            raise HTTPException(status_code=400, detail="Video file too large. Maximum size is 50MB.")
        
        # Encode to base64
        base64_video = base64.b64encode(video_data).decode('utf-8')
        
        return {"video_data": base64_video, "filename": file.filename}
        
    except Exception as e:
        logger.error(f"Error processing video upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def search_web(query: str) -> str:
    """Search the web using Serper API and return formatted results"""
    try:
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
            "q": query
        })
        headers = {
            'X-API-KEY': '67c090a334109db4480037614dbb1c635f29ad83',
            'Content-Type': 'application/json'
        }
        
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        
        if res.status != 200:
            return f"Web search failed: API returned status {res.status}"
        
        data = res.read()
        search_results = json.loads(data.decode("utf-8"))
        
        # Check if there's an error in the response
        if 'error' in search_results:
            return f"Web search failed: {search_results['error']}"
        
        # Format the search results with proper HTML structure
        formatted_results = ""
        
        # Add organic results with clickable links
        if "organic" in search_results and len(search_results["organic"]) > 0:
            formatted_results += "<h3>🔍 Search Results</h3>\n"
            for i, result in enumerate(search_results["organic"][:5], 1):
                title = result.get('title', 'No title')
                snippet = result.get('snippet', 'No description')
                link = result.get('link', 'No link')
                
                formatted_results += f"<h4>{i}. {title}</h4>\n"
                formatted_results += f"<p>{snippet}</p>\n"
                formatted_results += f"<p><a href='{link}' target='_blank'>🔗 Read more</a></p>\n\n"
        
        # Add knowledge graph if available
        if "knowledgeGraph" in search_results:
            kg = search_results["knowledgeGraph"]
            title = kg.get('title', 'N/A')
            description = kg.get('description', 'N/A')
            
            formatted_results += "<h3>📚 Knowledge Graph</h3>\n"
            formatted_results += f"<h4>{title}</h4>\n"
            formatted_results += f"<p>{description}</p>\n\n"
        
        # Add news results if available
        if "news" in search_results and len(search_results["news"]) > 0:
            formatted_results += "<h3>📰 Latest News</h3>\n"
            for i, news_item in enumerate(search_results["news"][:3], 1):
                title = news_item.get('title', 'No title')
                snippet = news_item.get('snippet', 'No description')
                link = news_item.get('link', 'No link')
                source = news_item.get('source', 'Unknown source')
                
                formatted_results += f"<h4>{i}. {title}</h4>\n"
                formatted_results += f"<p><em>Source: {source}</em></p>\n"
                formatted_results += f"<p>{snippet}</p>\n"
                formatted_results += f"<p><a href='{link}' target='_blank'>🔗 Read full article</a></p>\n\n"
        
        if not formatted_results:
            formatted_results = "No search results found. The search API may be experiencing issues or the query returned no results."
        
        return formatted_results
        
    except json.JSONDecodeError as e:
        return f"Web search failed: Failed to parse search results"
    except Exception as e:
        return f"Web search failed: {str(e)}"

@app.post("/web-search")
async def web_search(query: str):
    try:
        results = search_web(query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing web search: {str(e)}")

@app.post("/save-chat")
async def save_chat(request: SaveChatRequest):
    try:
        # Add messages to ConversationRAG storage
        for i in range(0, len(request.messages), 2):
            if i + 1 < len(request.messages):
                user_msg = request.messages[i]
                assistant_msg = request.messages[i + 1]
                
                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    conversation_rag.add_conversation(
                        user_input=user_msg.content,
                        assistant_response=assistant_msg.content,
                        chat_id=request.chat_id,
                        chat_title=request.title
                    )
        
        logger.info(f"Saved chat {request.chat_id} with {len(request.messages)} messages to ConversationRAG")
        
        return {"success": True, "chat_id": request.chat_id}
        
    except Exception as e:
        logger.error(f"Error saving chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving chat: {str(e)}")

@app.get("/chat-history", response_model=ChatHistoryResponse)
async def get_chat_history():
    try:
        chats = conversation_rag.get_chat_history_list()
        chat_items = [ChatHistoryItem(
            id=chat["id"],
            title=chat["title"],
            timestamp=chat["timestamp"],
            message_count=chat["message_count"]
        ) for chat in chats]
        
        return ChatHistoryResponse(chats=chat_items)
        
    except Exception as e:
        logger.error(f"Error fetching chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

@app.post("/load-chat", response_model=LoadChatResponse)
async def load_chat(request: LoadChatRequest):
    try:
        messages_data = conversation_rag.get_chat_messages(request.chat_id)
        
        if not messages_data:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        messages = [Message(role=msg["role"], content=msg["content"]) for msg in messages_data]
        chat_title = conversation_rag.chat_titles.get(request.chat_id, "Untitled Chat")
        
        return LoadChatResponse(
            chat_id=request.chat_id,
            title=chat_title,
            messages=messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading chat: {str(e)}")

@app.delete("/delete-chat/{chat_id}")
async def delete_chat(chat_id: str):
    try:
        # Check if chat exists
        chat_messages = conversation_rag.get_chat_messages(chat_id)
        if not chat_messages:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Delete from ConversationRAG storage
        conversation_rag.delete_chat(chat_id)
        
        logger.info(f"Deleted chat {chat_id} from ConversationRAG storage")
        
        return {"success": True, "message": "Chat deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")

@app.get("/test-web-search")
async def test_web_search():
    """Test endpoint to check if web search is working"""
    try:
        test_query = "latest news today"
        results = search_web(test_query)
        return {"query": test_query, "results": results, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.post("/generate-image", response_model=ChatResponse)
async def generate_image(request: ImageGenerationRequest):
    try:
        # Import Google Gemini libraries only when needed
        from google import genai
        from google.genai import types
        
        # Initialize Gemini client only when needed
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        gemini_client = genai.Client(api_key=api_key)
        
        # Use Google Gemini for image generation
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=request.prompt,
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        generated_text = ""
        generated_images = []
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                generated_text = part.text
                logger.info(f"Found generated text: {generated_text[:100]}...")
            elif part.inline_data is not None:
                # Convert the image data to base64
                image_data = part.inline_data.data
                base64_image = base64.b64encode(image_data).decode('utf-8')
                generated_images.append(base64_image)
                logger.info(f"Generated image with MIME type: {part.inline_data.mime_type}, Base64 length: {len(base64_image)}")
                
                # Save image locally for debugging
                try:
                    import os
                    from datetime import datetime
                    
                    # Create images directory if it doesn't exist
                    images_dir = "generated_images"
                    if not os.path.exists(images_dir):
                        os.makedirs(images_dir)
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"generated_image_{timestamp}_{len(generated_images)}.png"
                    filepath = os.path.join(images_dir, filename)
                    
                    # Save the raw image data
                    with open(filepath, "wb") as f:
                        f.write(image_data)
                    
                    logger.info(f"✅ Image saved locally: {filepath}")
                    logger.info(f"📁 Image file size: {len(image_data)} bytes")
                    
                except Exception as save_error:
                    logger.error(f"❌ Failed to save image locally: {save_error}")
        
        logger.info(f"Total generated images: {len(generated_images)}, Generated text: {bool(generated_text)}")
        
        # Create response with both text and images
        if generated_images:
            image_html = ""
            for i, img_base64 in enumerate(generated_images):
                # Get the correct MIME type from the response
                mime_type = "image/png"  # Default
                if i < len(response.candidates[0].content.parts):
                    for part in response.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.mime_type:
                            mime_type = part.inline_data.mime_type
                            break
                
                # Create a safe filename from the prompt
                prompt_words = request.prompt.lower().split()[:3]  # First 3 words
                safe_filename = "_".join(word for word in prompt_words if word.isalnum())
                if not safe_filename:
                    safe_filename = "generated"
                filename = f"yota_img_{safe_filename}_{i+1}.png"
                
                image_html += f'''<div style="margin: 8px 0;">
                    <img src="data:{mime_type};base64,{img_base64}" alt="Generated Image {i+1}" style="max-width: 300px; height: auto; border-radius: 8px; display: block;" />
                    <button onclick="downloadImage('data:{mime_type};base64,{img_base64}', '{filename}')" style="margin-top: 8px; padding: 6px 12px; background-color: #667eea; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 12px;">💾 Download Image</button>
                </div>'''
            
            response_content = f"<h3>🎨 Generated Image</h3>"
            if generated_text:
                response_content += f"<p>{generated_text}</p>"
            response_content += image_html
            
            # Add some debug info
            logger.info(f"Generated image response length: {len(response_content)}")
            logger.info(f"Image HTML: {image_html[:200]}...")  # First 200 chars
        else:
            response_content = generated_text or "Image generation completed, but no image was returned."
        
        return ChatResponse(message=response_content)
        
    except Exception as e:
        logger.error(f"Error in image generation endpoint: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a more helpful error response
        error_content = f"""
        <h3>🎨 Image Generation Error</h3>
        <p><strong>Sorry, image generation failed:</strong></p>
        <p>{str(e)}</p>
        <p><em>Please try again with a different prompt.</em></p>
        """
        return ChatResponse(message=error_content)

@app.post("/edit-image", response_model=ChatResponse)
async def edit_image(request: ImageEditRequest):
    try:
        # Import Google Gemini libraries only when needed
        from google import genai
        from google.genai import types
        import PIL.Image
        
        # Initialize Gemini client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        gemini_client = genai.Client(api_key=api_key)
        
        logger.info(f"Starting image editing with prompt: {request.prompt[:100]}...")
        
        # Convert base64 image to PIL Image
        try:
            image_bytes = base64.b64decode(request.image_data)
            original_image = PIL.Image.open(BytesIO(image_bytes))
            logger.info(f"Loaded original image: {original_image.size}, mode: {original_image.mode}")
        except Exception as img_error:
            logger.error(f"Failed to load image: {img_error}")
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(img_error)}")
        
        # Use Google Gemini for image editing
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash-preview-image-generation",
            contents=[request.prompt, original_image],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        generated_text = ""
        edited_images = []
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                generated_text = part.text
                logger.info(f"Found generated text: {generated_text[:100]}...")
            elif part.inline_data is not None:
                # Convert the edited image data to base64
                image_data = part.inline_data.data
                base64_image = base64.b64encode(image_data).decode('utf-8')
                edited_images.append(base64_image)
                logger.info(f"Generated edited image with MIME type: {part.inline_data.mime_type}, Base64 length: {len(base64_image)}")
                
                # Save edited image locally for debugging
                try:
                    import os
                    from datetime import datetime
                    
                    # Create images directory if it doesn't exist
                    images_dir = "edited_images"
                    if not os.path.exists(images_dir):
                        os.makedirs(images_dir)
                    
                    # Generate filename with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"edited_image_{timestamp}_{len(edited_images)}.png"
                    filepath = os.path.join(images_dir, filename)
                    
                    # Save the raw image data
                    with open(filepath, "wb") as f:
                        f.write(image_data)
                    
                    logger.info(f"✅ Edited image saved locally: {filepath}")
                    logger.info(f"📁 Edited image file size: {len(image_data)} bytes")
                    
                except Exception as save_error:
                    logger.error(f"❌ Failed to save edited image locally: {save_error}")
        
        logger.info(f"Total edited images: {len(edited_images)}, Generated text: {bool(generated_text)}")
        
        # Create response with both text and images
        if edited_images:
            image_html = ""
            for i, img_base64 in enumerate(edited_images):
                # Get the correct MIME type from the response
                mime_type = "image/png"  # Default
                if i < len(response.candidates[0].content.parts):
                    for part in response.candidates[0].content.parts:
                        if part.inline_data and part.inline_data.mime_type:
                            mime_type = part.inline_data.mime_type
                            break
                
                # Create a safe filename from the prompt
                prompt_words = request.prompt.lower().split()[:3]  # First 3 words
                safe_filename = "_".join(word for word in prompt_words if word.isalnum())
                if not safe_filename:
                    safe_filename = "edited"
                filename = f"yota_img_{safe_filename}_{i+1}.png"
                
                image_html += f'''<div style="margin: 8px 0;">
                    <img src="data:{mime_type};base64,{img_base64}" alt="Edited Image {i+1}" style="max-width: 300px; height: auto; border-radius: 8px; display: block;" />
                    <button onclick="downloadImage('data:{mime_type};base64,{img_base64}', '{filename}')" style="margin-top: 8px; padding: 6px 12px; background-color: #667eea; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 12px;">💾 Download Image</button>
                </div>'''
            
            response_content = f"<h3>🎨 Edited Image</h3>"
            if generated_text:
                response_content += f"<p>{generated_text}</p>"
            response_content += image_html
            
            # Add some debug info
            logger.info(f"Edited image response length: {len(response_content)}")
            logger.info(f"Image HTML: {image_html[:200]}...")  # First 200 chars
        else:
            response_content = generated_text or "Image editing completed, but no edited image was returned."
        
        return ChatResponse(message=response_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image editing endpoint: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a more helpful error response
        error_content = f"""
        <h3>🎨 Image Editing Error</h3>
        <p><strong>Sorry, image editing failed:</strong></p>
        <p>{str(e)}</p>
        <p><em>Please try uploading a different image or adjusting your editing prompt.</em></p>
        """
        return ChatResponse(message=error_content)

@app.post("/analyze-video", response_model=ChatResponse)
async def analyze_video(request: VideoAnalysisRequest):
    try:
        # Import Google Gemini libraries only when needed
        from google import genai
        import tempfile
        import os
        
        # Initialize Gemini client
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        gemini_client = genai.Client(api_key=api_key)
        
        logger.info(f"Starting video analysis with prompt: {request.prompt[:100]}...")
        
        # Decode base64 video data
        try:
            video_bytes = base64.b64decode(request.video_data)
            logger.info(f"Decoded video data: {len(video_bytes)} bytes")
        except Exception as decode_error:
            logger.error(f"Failed to decode video data: {decode_error}")
            raise HTTPException(status_code=400, detail=f"Invalid video data: {str(decode_error)}")
        
        # Save video to temporary file
        temp_video_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_video_path = temp_file.name
                temp_file.write(video_bytes)
            
            logger.info(f"Saved video to temporary file: {temp_video_path}")
            
            # Upload video file to Gemini
            uploaded_file = gemini_client.files.upload(file=temp_video_path)
            logger.info(f"Uploaded video file to Gemini: {uploaded_file.name}")
            
            # Wait for the file to be processed and become ACTIVE
            import time
            max_wait_time = 60  # Maximum wait time in seconds
            wait_interval = 2   # Check every 2 seconds
            elapsed_time = 0
            
            while elapsed_time < max_wait_time:
                file_status = gemini_client.files.get(name=uploaded_file.name)
                logger.info(f"File status: {file_status.state}")
                
                if file_status.state == "ACTIVE":
                    logger.info(f"File is now ACTIVE after {elapsed_time} seconds")
                    break
                elif file_status.state == "FAILED":
                    raise HTTPException(status_code=400, detail="Video file processing failed")
                
                time.sleep(wait_interval)
                elapsed_time += wait_interval
            
            if elapsed_time >= max_wait_time:
                raise HTTPException(status_code=408, detail="Video file processing timed out")
            
            # Use Google Gemini for video analysis
            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[uploaded_file, request.prompt]
            )
            
            # Clean up uploaded file from Gemini
            try:
                gemini_client.files.delete(uploaded_file.name)
                logger.info(f"Deleted uploaded file from Gemini: {uploaded_file.name}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup Gemini file: {cleanup_error}")
            
            response_text = response.text if hasattr(response, 'text') else str(response)
            logger.info(f"Video analysis completed, response length: {len(response_text)}")
            
            # Format response with HTML
            response_content = f"""
            <h3>🎥 Video Analysis</h3>
            <p>{response_text}</p>
            """
            
            return ChatResponse(message=response_content)
            
        finally:
            # Clean up temporary file
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                    logger.info(f"Cleaned up temporary file: {temp_video_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup temporary file: {cleanup_error}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in video analysis endpoint: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a more helpful error response
        error_content = f"""
        <h3>🎥 Video Analysis Error</h3>
        <p><strong>Sorry, video analysis failed:</strong></p>
        <p>{str(e)}</p>
        <p><em>Please try uploading a different video or adjusting your analysis prompt.</em></p>
        """
        return ChatResponse(message=error_content)

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
