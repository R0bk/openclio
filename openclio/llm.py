"""LLM utilities for making API calls with retries and caching."""
import dotenv
dotenv.load_dotenv()

from hashlib import sha256
import json
import logging
import numpy as np
from models import MessageDict
from aiolimiter import AsyncLimiter
from diskcache import Cache
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from openai.types.chat import ChatCompletion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiter and cache
rate_limit = AsyncLimiter(30, 5)
rate_limit = AsyncLimiter(5, 5)
embed_rate_limit = AsyncLimiter(100, 60)
cache = Cache("./cache", size_limit=2**32)

# Initialize OpenAI client
client = AsyncOpenAI()
MODEL = 'bedrock.anthropic.claude-3-5-sonnet' # 'bedrock.anthropic.claude-3-haiku'
EMBEDDING_MODEL = 'azure.text-embedding-3-large'

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying LLM call after {retry_state.outcome.exception()}"
    )
)
async def llm(messages: list[MessageDict], temperature: float = 0.5, model: str = MODEL) -> str:
    """Make an LLM API call with retries and disk caching
    
    Args:
        messages: List of message dictionaries with role and content
        temperature: Sampling temperature for generation
        
    Returns:
        Generated text response
    """
    # Create cache key from inputs
    key_dict = {
        "messages": messages,
        "temperature": temperature,
        "model": model
    }
    cache_key = sha256(json.dumps(key_dict, sort_keys=True).encode()).hexdigest()
    
    # Try to get from cache first
    if cache_key in cache:
        logger.debug("Cache hit for LLM call")
        return cache[cache_key]
        
    # If not in cache, make the API call with rate limiting
    async with rate_limit:
        response: ChatCompletion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        result = response.choices[0].message.content
        
        if result is None:
            raise ValueError("Returned LLM call result is None")
        
        # Join prefilled assistant response with filled response
        if messages[-1]['role'] == 'assistant':
            result = messages[-1]['content'] + result

        # Store in cache before returning
        cache[cache_key] = result
        logger.debug("Cached LLM response")
        
        return result 
    

def split_prompt_into_conversation(prompt: str) -> list[MessageDict]:
    """Splits a prompt into user and assistant messages"""
    messages = []
    lines = prompt.strip().split('\n') + ['Assistant:']
    
    buffer = []
    for line in lines:
        if line.startswith(('Human:', 'Assistant:')) and buffer:
            role = "user" if buffer[0].startswith('Human:') else "assistant"
            content = '\n'.join([buffer[0][buffer[0].find(':')+1:]] + buffer[1:])
            messages.append({"role": role, "content": content})
            buffer = []
        buffer.append(line)
                    
    return messages

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying embedding call after {retry_state.outcome.exception()}"
    )
)
async def embed(texts: list[str], offline: bool = False) -> np.ndarray:
    """Generate embeddings for a list of texts using either Azure or local model
    
    Args:
        texts: List of texts to embed
        offline: Whether to use online embedding model (True) or local SentenceTransformer (False)
        
    Returns:
        Array of embeddings with shape (len(texts), embedding_dim)
    """
    if not texts:
        return np.array([])
        
    if offline:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode(texts)

    # Process in batches of 500 to stay within API limits
    embeddings = []
    for i in range(0, len(texts), 500):
        batch = texts[i:i + 500]
        
        # Check cache
        key = sha256(json.dumps({"texts": batch, "model": EMBEDDING_MODEL}).encode()).hexdigest()
        if key in cache:
            embeddings.append(cache[key])
            continue
            
        # Get embeddings from API
        async with embed_rate_limit:
            response = await client.embeddings.create(model=EMBEDDING_MODEL, input=batch, dimensions=1024)
            batch_emb = np.array([r.embedding for r in response.data])
            cache[key] = batch_emb
            embeddings.append(batch_emb)
    
    return np.vstack(embeddings) if embeddings else np.array([])
