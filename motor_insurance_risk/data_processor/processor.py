import csv

import pandas as pd
from motor_insurance_risk.config import MAX_TOKENS_PER_CHUNK
from motor_insurance_risk.embeddings.generator import EmbeddingGenerator

class DataProcessor:
    """Class responsible for processing and chunking text data"""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()

    def process_dataframe(self, df):
        """
        Process dataframe and generate embeddings for content
        
        Args:
            df (pd.DataFrame): Input dataframe with content column
            
        Returns:
            pd.DataFrame: Processed dataframe with embeddings
        """
        chunks = []
        
        for _, row in df.iterrows():
            text = row['content']
            token_len = self.embedding_generator.get_token_count(text)
            
            if token_len <= MAX_TOKENS_PER_CHUNK:
                chunks.append(self._create_chunk_record(row, text, token_len))
            else:
                chunks.extend(self._split_and_process_text(row, text))
        
        chunk_contents = [chunk[7] for chunk in chunks]  # content is at index 7
        embeddings = self.embedding_generator.get_embeddings_batch(chunk_contents)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.append(embedding)

        return pd.DataFrame(
            chunks,
            columns=[
                'document', 'driver_id', 'vehicle_id', 'policy_id',
                'underwriting_decision', 'risk_class', 'reason_for_decline',
                'content', 'tokens', 'embeddings'
            ]
        )

    def _create_chunk_record(self, row, content, token_len):
        """Create a record for a content chunk"""
        return [
            row['document'], row['driver_id'], row['vehicle_id'],
            row['policy_id'], row['underwriting_decision'],
            row['risk_class'], row['reason_for_decline'],
            content, token_len
        ]

    def _split_and_process_text(self, row, text):
        """Split long text into smaller chunks"""
        chunks = []
        words = text.split()
        
        ideal_size = int(MAX_TOKENS_PER_CHUNK // (4/3))
        total_words = len(words)
        
        for i in range(0, total_words, ideal_size):
            chunk_words = words[i:i + ideal_size]
            chunk_text = ' '.join(chunk_words)
            token_len = self.embedding_generator.get_token_count(chunk_text)
            
            if token_len > 0:
                chunks.append(self._create_chunk_record(row, chunk_text, token_len))
        
        return chunks

    @staticmethod
    def save_to_file(df, filepath):
        """Save processed dataframe to csv file"""
        df.to_csv(filepath, index=False,quoting=csv.QUOTE_MINIMAL)