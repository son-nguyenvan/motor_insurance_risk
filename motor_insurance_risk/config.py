import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
COMPLETION_MODEL = "gpt-4"
MAX_TOKENS = 1000
TEMPERATURE = 0

# Database Configuration
DB_CONNECTION_STRING = os.getenv("TIMESCALE_CONNECTION_STRING")

# Embedding Configuration
MAX_TOKENS_PER_CHUNK = 512
EMBEDDING_DIMENSION = 1536

# Cost Configuration
EMBEDDING_COST_PER_1K_TOKENS = 0.0002

# Table Configuration
MOTOR_EMBEDDINGS_TABLE = "motor_embeddings"