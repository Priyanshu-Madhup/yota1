import os
import base64
import http.client
import json
import logging
import pickle
import uuid
import hashlib
import jwt
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from groq import Groq
from typing import List, Optional, Dict
from dotenv import load_dotenv
from PIL import Image
from datetime import datetime, timedelta

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
    from sentence_transformers import SentenceTransformer
    print("✅ SentenceTransformers library loaded successfully")
    TRANSFORMERS_AVAILABLE = True
except Exception as e:  # Catch broader exceptions including RuntimeError
    print(f"⚠️  SentenceTransformers not available: {e}")
    print("📦 This might be due to PyTorch/TorchVision compatibility issues")
    SentenceTransformer = None
    TRANSFORMERS_AVAILABLE = False

# Fallback to sklearn if transformers are not available
if not TRANSFORMERS_AVAILABLE:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        print("✅ Using sklearn TF-IDF as fallback")
        SKLEARN_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️  sklearn not available: {e}")
        SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = False

# RAG is available if we have either transformers+faiss or sklearn
RAG_AVAILABLE = (FAISS_AVAILABLE and TRANSFORMERS_AVAILABLE) or SKLEARN_AVAILABLE

if RAG_AVAILABLE:
    print("🧠 RAG system fully available")
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

# JWT Configuration
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-this-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7 days

# Security
security = HTTPBearer()

# User Database (In production, use a real database)
USERS_FILE = "users.json"

def load_users():
    """Load users from JSON file"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading users: {e}")
        return {}

def save_users(users):
    """Save users to JSON file"""
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving users: {e}")

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed_password

def create_jwt_token(user_id: str) -> str:
    """Create JWT token for user"""
    payload = {
        "user_id": user_id,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token: str) -> Optional[str]:
    """Verify JWT token and return user_id"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token"""
    token = credentials.credentials
    user_id = verify_jwt_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    
    users = load_users()
    if user_id not in users:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    return {"user_id": user_id, **users[user_id]}

# Per-User RAG Chat Storage System
class ConversationRAG:
    """
    Per-user RAG system for conversation memory using FAISS + embeddings.
    Each user gets their own memory storage.
    """
    
    def __init__(self, max_conversations=1000):
        self.max_conversations = max_conversations
        self.conversations = {}  # Dict: user_id -> List of conversations
        self.embeddings_list = {}  # Dict: user_id -> List of embeddings
        self.index = {}  # Dict: user_id -> FAISS index
        self.chat_titles = {}  # Dict: user_id -> Dict of chat titles
        self.memory_dir = "memory_storage"
        self.users_dir = os.path.join(self.memory_dir, "users")
        self.rag_enabled = False
        
        # Create directories
        os.makedirs(self.users_dir, exist_ok=True)
        
        # Initialize the system
        self._initialize_rag_system()
    
    def _get_user_memory_file(self, user_id: str) -> str:
        """Get memory file path for user"""
        user_dir = os.path.join(self.users_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        return os.path.join(user_dir, "conversation_memory.pkl")
    
    def _get_user_index_file(self, user_id: str) -> str:
        """Get FAISS index file path for user"""
        user_dir = os.path.join(self.users_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        return os.path.join(user_dir, "conversation_faiss.index")
    
    def _ensure_user_initialized(self, user_id: str):
        """Ensure user's memory is initialized"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
            self.embeddings_list[user_id] = []
            self.index[user_id] = None
            self.chat_titles[user_id] = {}
            self._load_user_memory(user_id)
    
    def _initialize_rag_system(self):
        """Initialize RAG system with fallback mechanisms"""
        try:
            # Try FAISS first
            if FAISS_AVAILABLE:
                self.rag_enabled = True
                print("🚀 Using FAISS-based RAG system")
            else:
                print("⚠️  FAISS not available, using simple memory")
                self.rag_enabled = False
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            self.rag_enabled = False
    
    def _get_simple_embedding(self, text: str):
        """Create a simple embedding using TF-IDF or basic text features"""
        try:
            if not SKLEARN_AVAILABLE:
                # Very basic embedding using character frequencies and length
                import string
                features = []
                text_lower = text.lower()
                
                # Basic features
                features.append(len(text))  # Text length
                features.append(text.count(' '))  # Word count approximation
                
                # Character frequency features (a-z)
                for char in string.ascii_lowercase:
                    features.append(text_lower.count(char))
                
                # Add some name-specific features
                features.append(sum(1 for word in text.split() if word.istitle()))  # Proper nouns
                features.append(text_lower.count('name'))
                features.append(text_lower.count('called'))
                features.append(text_lower.count('my'))
                features.append(text_lower.count('i'))
                
                # Pad to make it a reasonable size
                while len(features) < 50:
                    features.append(0.0)
                
                return np.array(features[:50], dtype='float32')
            else:
                # Use TF-IDF for better embeddings
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer(max_features=100, stop_words=None, lowercase=False)
                
                # We need multiple texts to fit, so create a small corpus
                corpus = [text, "sample text", "another sample"]
                tfidf_matrix = vectorizer.fit_transform(corpus)
                embedding = tfidf_matrix[0].toarray().flatten().astype('float32')
                
                # Pad to consistent size
                if len(embedding) < 100:
                    embedding = np.pad(embedding, (0, 100 - len(embedding)), 'constant')
                
                return embedding[:100]
                
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            # Fallback to very basic embedding
            return np.random.random(50).astype('float32')
    
    def add_conversation(self, user_id: str, user_input: str, assistant_response: str, chat_id: str, chat_title: str = None):
        """Add a conversation to user's memory with RAG indexing (prevents duplicates)"""
        try:
            self._ensure_user_initialized(user_id)
            
            # Check for duplicate conversations (same user_input, assistant_response, and chat_id)
            existing_conversations = self.conversations.get(user_id, [])
            for existing_conv in existing_conversations:
                if (existing_conv['user_input'] == user_input and 
                    existing_conv['assistant_response'] == assistant_response and 
                    existing_conv['chat_id'] == chat_id):
                    logger.info(f"Duplicate conversation detected for user {user_id}, chat {chat_id}. Skipping.")
                    return
            
            conversation = {
                'user_input': user_input,
                'assistant_response': assistant_response,
                'chat_id': chat_id,
                'timestamp': datetime.now().isoformat(),
                'id': len(self.conversations[user_id])
            }
            
            # Store chat title
            if chat_title:
                self.chat_titles[user_id][chat_id] = chat_title
            
            # Add to conversations list
            self.conversations[user_id].append(conversation)
            
            # Create embedding for this conversation
            if self.rag_enabled and FAISS_AVAILABLE:
                conversation_text = f"User: {user_input} Assistant: {assistant_response}"
                embedding = self._get_simple_embedding(conversation_text)
                self.embeddings_list[user_id].append(embedding)
                
                # Rebuild FAISS index for this user
                self._rebuild_faiss_index(user_id)
            
            # Cleanup old conversations if needed
            self._cleanup_old_conversations(user_id)
            
            # Save to disk
            self._save_user_memory(user_id)
            
            logger.info(f"Added conversation to user {user_id} RAG memory. Total: {len(self.conversations[user_id])}")
            
        except Exception as e:
            logger.error(f"Error adding conversation for user {user_id}: {e}")
    
    def _rebuild_faiss_index(self, user_id: str):
        """Rebuild FAISS index from user's embeddings"""
        try:
            self._ensure_user_initialized(user_id)
            if not self.embeddings_list[user_id] or not FAISS_AVAILABLE:
                return
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(self.embeddings_list[user_id]).astype('float32')
            
            # Create FAISS index
            dimension = embeddings_array.shape[1]
            self.index[user_id] = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
            self.index[user_id].add(embeddings_array)
            
            logger.info(f"Rebuilt FAISS index for user {user_id} with {len(self.embeddings_list[user_id])} embeddings, dimension {dimension}")
            
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index for user {user_id}: {e}")
    
    def search_relevant_conversations(self, user_id: str, query: str, chat_id: str = None, top_k: int = 4):
        """Search for relevant conversations in user's memory using FAISS"""
        try:
            self._ensure_user_initialized(user_id)
            if not self.rag_enabled or not self.index.get(user_id) or len(self.conversations[user_id]) == 0:
                logger.info(f"RAG search skipped for user {user_id}: enabled={self.rag_enabled}, index={self.index.get(user_id) is not None}, conversations={len(self.conversations.get(user_id, []))}")
                return []
            
            # Get embedding for query
            query_embedding = self._get_simple_embedding(query)
            query_array = np.array([query_embedding]).astype('float32')
            
            # Search FAISS index
            k = min(top_k * 2, len(self.conversations[user_id]))  # Search for more than needed
            D, I = self.index[user_id].search(query_array, k)
            
            logger.info(f"FAISS search for user {user_id} query '{query}': Found {len(I[0])} results")
            
            relevant_conversations = []
            for i, (distance, idx) in enumerate(zip(D[0], I[0])):
                if idx >= len(self.conversations[user_id]):
                    continue
                
                conversation = self.conversations[user_id][idx]
                
                # Filter by chat_id if specified
                if chat_id and conversation.get('chat_id') != chat_id:
                    continue
                
                # Convert distance to similarity score (lower distance = higher similarity)
                similarity = 1.0 / (1.0 + distance)
                
                logger.info(f"User {user_id} conversation {idx}: distance={distance}, similarity={similarity}, text='{conversation['user_input'][:50]}...'")
                
                relevant_conversations.append({
                    'conversation': conversation,
                    'score': float(similarity),
                    'index': idx
                })
                
                if len(relevant_conversations) >= top_k:
                    break
            
            # Sort by similarity (highest first)
            relevant_conversations.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Returning {len(relevant_conversations)} relevant conversations for user {user_id}")
            return relevant_conversations[:top_k]
            
        except Exception as e:
            logger.error(f"Error in FAISS search for user {user_id}: {e}")
            return self._fallback_keyword_search(user_id, query, chat_id, top_k)
    
    def _fallback_keyword_search(self, user_id: str, query: str, chat_id: str = None, top_k: int = 4):
        """Fallback keyword search when FAISS fails"""
        try:
            self._ensure_user_initialized(user_id)
            user_conversations = self.conversations.get(user_id, [])
            query_words = query.lower().split()
            relevant_conversations = []
            
            for idx, conversation in enumerate(user_conversations):
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
            logger.error(f"Error in fallback search for user {user_id}: {e}")
            return []

    def get_context_for_prompt(self, user_id: str, current_query: str, chat_id: str = None, max_context_length: int = 1500) -> str:
        """Get relevant context for the current query for a specific user"""
        try:
            self._ensure_user_initialized(user_id)
            
            # Check if this might be asking about personal information
            personal_keywords = ['name', 'called', 'who am i', 'my name', 'remember', 'told you', 'what is my']
            is_personal_query = any(keyword in current_query.lower() for keyword in personal_keywords)
            
            logger.info(f"Query analysis for user {user_id}: '{current_query}' - Personal query: {is_personal_query}")
            
            # If it's a personal query, search more broadly
            top_k = 8 if is_personal_query else 4
            relevant_convos = self.search_relevant_conversations(user_id, current_query, chat_id, top_k=top_k)
            
            # If no results for personal query, try searching for any name-like words
            if is_personal_query and not relevant_convos:
                logger.info(f"No direct matches for personal query for user {user_id}, searching for proper nouns...")
                # Search for conversations containing proper nouns (potential names)
                user_conversations = self.conversations.get(user_id, [])
                all_relevant = []
                for conv in user_conversations:
                    text = f"{conv['user_input']} {conv['assistant_response']}"
                    # Look for capitalized words that might be names
                    words = text.split()
                    proper_nouns = [word for word in words if word.istitle() and word.isalpha() and len(word) > 2]
                    if proper_nouns:
                        logger.info(f"Found proper nouns: {proper_nouns} in conversation: '{conv['user_input'][:50]}...'")
                        all_relevant.append({
                            'conversation': conv,
                            'score': 0.5,  # Give it a decent score
                            'index': conv.get('id', 0)
                        })
                relevant_convos = all_relevant[:top_k]
            
            context = self._build_context_string(relevant_convos, max_context_length)
            if context:
                logger.info(f"Built context for user {user_id}: {len(context)} characters")
            else:
                logger.info(f"No context found for user {user_id}")
            
            return context
        except Exception as e:
            logger.error(f"Error getting context for user {user_id}: {e}")
            return ""
    
    def _build_context_string(self, conversations: List[Dict], max_length: int) -> str:
        """Build context string from relevant conversations"""
        try:
            if not conversations:
                return ""
            
            context_parts = []
            current_length = 0
            
            for conv_data in conversations:
                conversation = conv_data['conversation']
                user_input = conversation['user_input'][:200]  # Limit length
                assistant_response = conversation['assistant_response'][:200]  # Limit length
                
                context_part = f"Previous: User: {user_input}\nAssistant: {assistant_response}\n"
                
                if current_length + len(context_part) > max_length:
                    break
                
                context_parts.append(context_part)
                current_length += len(context_part)
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Error building context string: {e}")
            return ""

    def _save_user_memory(self, user_id: str):
        """Save user's memory to disk"""
        try:
            user_dir = os.path.join(self.users_dir, user_id)
            os.makedirs(user_dir, exist_ok=True)
            
            # Save conversations and metadata
            memory_data = {
                'conversations': self.conversations.get(user_id, []),
                'chat_titles': self.chat_titles.get(user_id, {}),
                'embeddings_list': self.embeddings_list.get(user_id, []),
                'rag_enabled': self.rag_enabled
            }
            
            memory_file = self._get_user_memory_file(user_id)
            with open(memory_file, 'wb') as f:
                pickle.dump(memory_data, f)
            
            # Save FAISS index if available
            if self.rag_enabled and self.index.get(user_id) is not None and FAISS_AVAILABLE:
                index_file = self._get_user_index_file(user_id)
                faiss.write_index(self.index[user_id], index_file)
                
            logger.info(f"Saved memory for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error saving memory for user {user_id}: {e}")
    
    def _load_user_memory(self, user_id: str):
        """Load user's memory from disk"""
        try:
            memory_file = self._get_user_memory_file(user_id)
            if os.path.exists(memory_file):
                with open(memory_file, 'rb') as f:
                    memory_data = pickle.load(f)
                    
                self.conversations[user_id] = memory_data.get('conversations', [])
                self.chat_titles[user_id] = memory_data.get('chat_titles', {})
                self.embeddings_list[user_id] = memory_data.get('embeddings_list', [])
                
                logger.info(f"Loaded {len(self.conversations[user_id])} conversations for user {user_id}")
            
            # Load FAISS index if available
            index_file = self._get_user_index_file(user_id)
            if self.rag_enabled and os.path.exists(index_file) and FAISS_AVAILABLE:
                try:
                    self.index[user_id] = faiss.read_index(index_file)
                    logger.info(f"Loaded FAISS index for user {user_id} with {self.index[user_id].ntotal} entries")
                except Exception as e:
                    logger.error(f"Error loading FAISS index for user {user_id}: {e}")
                    # Rebuild index from embeddings
                    if self.embeddings_list.get(user_id):
                        self._rebuild_faiss_index(user_id)
                    
        except Exception as e:
            logger.error(f"Error loading memory for user {user_id}: {e}")
    
    def get_chat_history_list(self, user_id: str) -> List[Dict]:
        """Get list of chat histories for a user with deduplication"""
        try:
            self._ensure_user_initialized(user_id)
            chat_history = {}
            seen_titles = set()
            
            # Group conversations by chat_id and deduplicate by title
            for conv in self.conversations.get(user_id, []):
                chat_id = conv.get('chat_id', 'default_chat')
                chat_title = self.chat_titles.get(user_id, {}).get(chat_id, f"Chat {chat_id}")
                
                # Skip if we've already seen this title (prevents duplicate titles)
                if chat_title in seen_titles:
                    continue
                    
                if chat_id not in chat_history:
                    chat_history[chat_id] = {
                        'id': chat_id,
                        'title': chat_title,
                        'timestamp': conv.get('timestamp', datetime.now().isoformat()),
                        'message_count': 0
                    }
                    seen_titles.add(chat_title)
                    
                chat_history[chat_id]['message_count'] += 2  # User + Assistant message
                
                # Update timestamp to the latest message
                if conv.get('timestamp'):
                    chat_history[chat_id]['timestamp'] = conv['timestamp']
            
            # Convert to list and sort by timestamp (newest first)
            chat_list = list(chat_history.values())
            chat_list.sort(key=lambda x: x['timestamp'], reverse=True)
            
            logger.info(f"Retrieved {len(chat_list)} unique chat histories for user {user_id}")
            return chat_list
            
        except Exception as e:
            logger.error(f"Error getting chat history for user {user_id}: {e}")
            return []
    
    def get_chat_messages(self, user_id: str, chat_id: str) -> List[Dict]:
        """Get messages for a specific chat"""
        try:
            self._ensure_user_initialized(user_id)
            messages = []
            
            # Find conversations for this chat_id
            for conv in self.conversations.get(user_id, []):
                if conv.get('chat_id') == chat_id:
                    # Add user message
                    messages.append({
                        'role': 'user',
                        'content': conv['user_input']
                    })
                    # Add assistant message
                    messages.append({
                        'role': 'assistant',
                        'content': conv['assistant_response']
                    })
            
            logger.info(f"Retrieved {len(messages)} messages for chat {chat_id}, user {user_id}")
            return messages
            
        except Exception as e:
            logger.error(f"Error getting messages for chat {chat_id}, user {user_id}: {e}")
            return []
    
    def get_chat_title(self, user_id: str, chat_id: str) -> Optional[str]:
        """Get title for a specific chat"""
        try:
            self._ensure_user_initialized(user_id)
            return self.chat_titles.get(user_id, {}).get(chat_id)
        except Exception as e:
            logger.error(f"Error getting chat title for chat {chat_id}, user {user_id}: {e}")
            return None
    
    def delete_chat(self, user_id: str, chat_id: str):
        """Delete a specific chat and its conversations"""
        try:
            self._ensure_user_initialized(user_id)
            
            # Remove conversations for this chat_id
            user_conversations = self.conversations.get(user_id, [])
            remaining_conversations = []
            remaining_embeddings = []
            
            for i, conv in enumerate(user_conversations):
                if conv.get('chat_id') != chat_id:
                    remaining_conversations.append(conv)
                    # Keep corresponding embedding if it exists
                    if i < len(self.embeddings_list.get(user_id, [])):
                        remaining_embeddings.append(self.embeddings_list[user_id][i])
            
            # Update conversations and embeddings
            self.conversations[user_id] = remaining_conversations
            self.embeddings_list[user_id] = remaining_embeddings
            
            # Remove chat title
            if user_id in self.chat_titles and chat_id in self.chat_titles[user_id]:
                del self.chat_titles[user_id][chat_id]
            
            # Rebuild FAISS index
            if self.rag_enabled and remaining_embeddings:
                self._rebuild_faiss_index(user_id)
            elif self.rag_enabled:
                # No embeddings left, reset index
                self.index[user_id] = None
            
            # Save changes
            self._save_user_memory(user_id)
            
            logger.info(f"Deleted chat {chat_id} for user {user_id}. Remaining conversations: {len(remaining_conversations)}")
            
        except Exception as e:
            logger.error(f"Error deleting chat {chat_id} for user {user_id}: {e}")

    def _cleanup_old_conversations(self, user_id: str):
        """Remove old conversations if we exceed max limit for a user"""
        try:
            user_conversations = self.conversations.get(user_id, [])
            if len(user_conversations) > self.max_conversations:
                # Remove oldest conversations
                remove_count = len(user_conversations) - self.max_conversations
                self.conversations[user_id] = user_conversations[remove_count:]
                
                # Also cleanup embeddings
                user_embeddings = self.embeddings_list.get(user_id, [])
                if len(user_embeddings) > remove_count:
                    self.embeddings_list[user_id] = user_embeddings[remove_count:]
                
                # Rebuild FAISS index
                self._rebuild_faiss_index(user_id)
                
        except Exception as e:
            logger.error(f"Error cleaning up conversations for user {user_id}: {e}")

    def remove_duplicate_conversations(self, user_id: str):
        """Remove duplicate conversations for a user"""
        try:
            self._ensure_user_initialized(user_id)
            
            conversations = self.conversations.get(user_id, [])
            if not conversations:
                return
            
            # Track unique conversations using a set of tuples
            seen_conversations = set()
            unique_conversations = []
            unique_embeddings = []
            
            for i, conv in enumerate(conversations):
                # Create a unique identifier for this conversation
                conv_key = (conv['user_input'], conv['assistant_response'], conv['chat_id'])
                
                if conv_key not in seen_conversations:
                    seen_conversations.add(conv_key)
                    unique_conversations.append(conv)
                    
                    # Keep corresponding embedding if it exists
                    if i < len(self.embeddings_list.get(user_id, [])):
                        unique_embeddings.append(self.embeddings_list[user_id][i])
            
            # Update with unique conversations
            original_count = len(conversations)
            self.conversations[user_id] = unique_conversations
            self.embeddings_list[user_id] = unique_embeddings
            
            # Rebuild FAISS index with unique conversations
            if self.rag_enabled and unique_embeddings:
                self._rebuild_faiss_index(user_id)
            
            # Save changes
            self._save_user_memory(user_id)
            
            removed_count = original_count - len(unique_conversations)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} duplicate conversations for user {user_id}. Remaining: {len(unique_conversations)}")
            
        except Exception as e:
            logger.error(f"Error removing duplicates for user {user_id}: {e}")

# Initialize per-user RAG storage
conversation_rag = ConversationRAG()

# Pydantic models for authentication
class UserRegister(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

class UserProfile(BaseModel):
    user_id: str
    username: str
    email: str
    created_at: str

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
    return {"message": "Chatbot API is running with per-user RAG-powered chat storage and authentication"}

# Authentication Endpoints
@app.post("/register", response_model=AuthResponse)
async def register(user_data: UserRegister):
    try:
        users = load_users()
        
        # Check if username already exists
        for user_id, user_info in users.items():
            if user_info["username"] == user_data.username:
                raise HTTPException(status_code=400, detail="Username already exists")
            if user_info["email"] == user_data.email:
                raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create new user
        user_id = str(uuid.uuid4())
        users[user_id] = {
            "username": user_data.username,
            "email": user_data.email,
            "password_hash": hash_password(user_data.password),
            "created_at": datetime.now().isoformat()
        }
        
        # Save users
        save_users(users)
        
        # Create JWT token
        token = create_jwt_token(user_id)
        
        # Initialize user's conversation memory
        conversation_rag._ensure_user_initialized(user_id)
        
        return AuthResponse(
            access_token=token,
            token_type="bearer",
            user={
                "user_id": user_id,
                "username": user_data.username,
                "email": user_data.email,
                "created_at": users[user_id]["created_at"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during registration: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/login", response_model=AuthResponse)
async def login(user_data: UserLogin):
    try:
        users = load_users()
        
        # Find user by username
        user_id = None
        user_info = None
        for uid, info in users.items():
            if info["username"] == user_data.username:
                user_id = uid
                user_info = info
                break
        
        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # Verify password
        if not verify_password(user_data.password, user_info["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        # Create JWT token
        token = create_jwt_token(user_id)
        
        # Ensure user's conversation memory is loaded
        conversation_rag._ensure_user_initialized(user_id)
        
        return AuthResponse(
            access_token=token,
            token_type="bearer",
            user={
                "user_id": user_id,
                "username": user_info["username"],
                "email": user_info["email"],
                "created_at": user_info["created_at"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/me", response_model=UserProfile)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    return UserProfile(
        user_id=current_user["user_id"],
        username=current_user["username"],
        email=current_user["email"],
        created_at=current_user["created_at"]
    )

@app.get("/profile", response_model=UserProfile)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Alias for /me endpoint for frontend compatibility"""
    return UserProfile(
        user_id=current_user["user_id"],
        username=current_user["username"],
        email=current_user["email"],
        created_at=current_user["created_at"]
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        # Convert Pydantic messages to dict format for Groq API
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Get the latest user message for RAG retrieval
        latest_user_message = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                latest_user_message = msg["content"]
                break
        
        # Use new ConversationRAG to get relevant context for this user
        context_from_rag = ""
        if latest_user_message:
            # Use chat_id for context if provided, otherwise search all conversations for this user
            context_chat_id = request.chat_id if request.chat_id else None
            context_from_rag = conversation_rag.get_context_for_prompt(user_id, latest_user_message, chat_id=context_chat_id)
            if context_from_rag:
                logger.info(f"Retrieved context from RAG for user {user_id}: {len(context_from_rag)} characters")
                logger.info(f"RAG Context: {context_from_rag[:500]}...")  # Log first 500 chars for debugging
            else:
                logger.info(f"No RAG context found for user {user_id} query: '{latest_user_message}'")
                # Log some debug info
                user_conversations = conversation_rag.conversations.get(user_id, [])
                logger.info(f"Total conversations for user {user_id}: {len(user_conversations)}")
                if user_conversations:
                    recent_conv = user_conversations[-1]
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
        
        # Add system message with user context
        system_content = f"You are an expert AI assistant named Yota that provides comprehensive, well-structured responses using HTML formatting. You are chatting with {current_user['username']}. For most responses, focus on using subheadings and clean structure:\n- <h3>Subheadings</h3> for main sections (use these primarily)\n- <h4>Minor headings</h4> for subsections when needed\n- <ul><li>Bullet points</li></ul> for lists\n- <ol><li>Numbered lists</li></ol> for sequential information\n- <strong>Bold text</strong> for emphasis\n- <a href='url' target='_blank'>Clickable links</a> when referencing sources\n- <p>Paragraphs</p> with proper spacing\n\nOnly use <h2> for very major topics or complex responses that need large section breaks. Keep responses engaging, informative, and visually appealing with clean HTML structure."
        context_messages.append({"role": "system", "content": system_content})
        
        # Add relevant context from RAG if available
        if context_from_rag:
            context_messages.append({"role": "system", "content": f"RELEVANT CONVERSATION CONTEXT:\n{context_from_rag}"})
        
        # Add web search results if available
        if web_search_content:
            context_messages.append({"role": "system", "content": web_search_content})
        
        # Add the full conversation history for complete context
        context_messages.extend(messages)
        
        logger.info(f"Sending {len(context_messages)} messages to Groq for user {user_id} (ConversationRAG-optimized)")
        
        # Make API call to Groq with context-optimized messages
        chat_completion = client.chat.completions.create(
            messages=context_messages,
            model=request.model,
            max_tokens=2000,
            temperature=0.9
        )
        
        response_content = chat_completion.choices[0].message.content
        
        # Store conversation in user's RAG memory if we have a valid conversation
        if latest_user_message and response_content:
            chat_id = request.chat_id or f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            conversation_rag.add_conversation(
                user_id=user_id,
                user_input=latest_user_message,
                assistant_response=response_content,
                chat_id=chat_id,
                chat_title=f"Chat {datetime.now().strftime('%m/%d %H:%M')}"
            )
        
        return ChatResponse(message=response_content)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint for user {current_user.get('user_id', 'unknown')}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@app.post("/chat-with-image", response_model=ChatResponse)
async def chat_with_image(request: ImageAnalysisRequest, current_user: dict = Depends(get_current_user)):
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
async def upload_image(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
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
async def save_chat(request: SaveChatRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        
        # Check if this chat has already been saved to prevent duplicates
        existing_conversations = conversation_rag.conversations.get(user_id, [])
        existing_chat_ids = set(conv['chat_id'] for conv in existing_conversations)
        
        # Only save if this chat_id doesn't already exist
        if request.chat_id not in existing_chat_ids:
            # Add messages to ConversationRAG storage
            for i in range(0, len(request.messages), 2):
                if i + 1 < len(request.messages):
                    user_msg = request.messages[i]
                    assistant_msg = request.messages[i + 1]
                    
                    if user_msg.role == "user" and assistant_msg.role == "assistant":
                        conversation_rag.add_conversation(
                            user_id=user_id,
                            user_input=user_msg.content,
                            assistant_response=assistant_msg.content,
                            chat_id=request.chat_id,
                            chat_title=request.title
                        )
            
            logger.info(f"Saved new chat {request.chat_id} with {len(request.messages)} messages to ConversationRAG for user {user_id}")
        else:
            logger.info(f"Chat {request.chat_id} already exists for user {user_id}, skipping duplicate save")
        
        return {"success": True, "chat_id": request.chat_id}
        
    except Exception as e:
        logger.error(f"Error saving chat for user {current_user.get('user_id', 'unknown')}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving chat: {str(e)}")

@app.get("/chat-history", response_model=ChatHistoryResponse)
async def get_chat_history(current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        chats = conversation_rag.get_chat_history_list(user_id)
        chat_items = [ChatHistoryItem(
            id=chat["id"],
            title=chat["title"],
            timestamp=chat["timestamp"],
            message_count=chat["message_count"]
        ) for chat in chats]
        
        return ChatHistoryResponse(chats=chat_items)
        
    except Exception as e:
        logger.error(f"Error fetching chat history for user {current_user.get('user_id', 'unknown')}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching chat history: {str(e)}")

@app.post("/load-chat", response_model=LoadChatResponse)
async def load_chat(request: LoadChatRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        messages_data = conversation_rag.get_chat_messages(user_id, request.chat_id)
        
        if not messages_data:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        messages = [Message(role=msg["role"], content=msg["content"]) for msg in messages_data]
        chat_title = conversation_rag.get_chat_title(user_id, request.chat_id) or "Untitled Chat"
        
        return LoadChatResponse(
            chat_id=request.chat_id,
            title=chat_title,
            messages=messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading chat for user {current_user.get('user_id', 'unknown')}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading chat: {str(e)}")

@app.delete("/delete-chat/{chat_id}")
async def delete_chat(chat_id: str, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        # Check if chat exists for this user
        chat_messages = conversation_rag.get_chat_messages(user_id, chat_id)
        if not chat_messages:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Delete from ConversationRAG storage
        conversation_rag.delete_chat(user_id, chat_id)
        
        logger.info(f"Deleted chat {chat_id} for user {user_id} from ConversationRAG storage")
        
        return {"success": True, "message": "Chat deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat for user {current_user.get('user_id', 'unknown')}: {str(e)}")
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
        gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        
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
            elif part.inline_data is not None:
                # Convert the image data to base64
                image_data = part.inline_data.data
                base64_image = base64.b64encode(image_data).decode('utf-8')
                generated_images.append(base64_image)
        
        # Create response with both text and images
        if generated_images:
            image_html = ""
            for i, img_base64 in enumerate(generated_images):
                image_html += f'<img src="data:image/png;base64,{img_base64}" alt="Generated Image {i+1}" style="max-width: 100%; border-radius: 8px; margin: 8px 0;" />'
            
            response_content = f"<h3>🎨 Generated Image</h3>"
            if generated_text:
                response_content += f"<p>{generated_text}</p>"
            response_content += image_html
        else:
            response_content = generated_text or "Image generation completed, but no image was returned."
        
        return ChatResponse(message=response_content)
        
    except Exception as e:
        logger.error(f"Error in image generation endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

@app.post("/cleanup-duplicates")
async def cleanup_duplicates(current_user: dict = Depends(get_current_user)):
    """Clean up duplicate conversations for the current user"""
    try:
        user_id = current_user["user_id"]
        conversation_rag.remove_duplicate_conversations(user_id)
        
        # Get updated chat count
        chats = conversation_rag.get_chat_history_list(user_id)
        total_conversations = len(conversation_rag.conversations.get(user_id, []))
        
        return {
            "success": True,
            "message": "Duplicate conversations cleaned up successfully",
            "total_chats": len(chats),
            "total_conversations": total_conversations
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up duplicates for user {current_user.get('user_id', 'unknown')}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cleaning up duplicates: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
