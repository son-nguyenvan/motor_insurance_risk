import tiktoken
from motor_insurance_risk.config import EMBEDDING_COST_PER_1K_TOKENS

def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Calculate number of tokens in a text string"""
    if not string:
        return 0
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def get_essay_length(essay: str) -> int:
    """Calculate length of essay in words"""
    return len(essay.split())

def get_embedding_cost(num_tokens: int) -> float:
    """Calculate cost of embedding tokens"""
    return (num_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS

def calculate_total_embeddings_cost(df) -> float:
    """Calculate total cost of embedding all content in dataframe"""
    total_tokens = sum(
        num_tokens_from_string(text)
        for text in df['content']
        if text
    )
    return get_embedding_cost(total_tokens)