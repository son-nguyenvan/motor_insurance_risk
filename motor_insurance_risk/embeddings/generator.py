# embedding_generator.py

from openai import OpenAI
from motor_insurance_risk.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL
)
from motor_insurance_risk.embeddings.utils import num_tokens_from_string

class EmbeddingGenerator:
    """Class responsible for generating embeddings using OpenAI's API"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def get_embedding(self, text):
        """
        Generate embedding for a single text input
        
        Args:
            text (str): The text to generate embedding for
            
        Returns:
            list: The embedding vector
        """
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text.replace("\n", " ")
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts):
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts (list): List of texts to generate embeddings for
            
        Returns:
            list: List of embedding vectors
        """
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=cleaned_texts
        )
        return [data.embedding for data in response.data]

    def get_token_count(self, text):
        """
        Get token count for a text
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        return num_tokens_from_string(text)