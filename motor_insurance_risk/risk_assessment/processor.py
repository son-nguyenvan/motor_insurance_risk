import json
from openai import OpenAI

from motor_insurance_risk.config import (
    OPENAI_API_KEY,
    COMPLETION_MODEL,
    TEMPERATURE,
    MAX_TOKENS
)
from motor_insurance_risk.database.connection import DatabaseConnection
from motor_insurance_risk.embeddings.generator import EmbeddingGenerator

class RiskAssessmentProcessor:
    def __init__(self):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.embedding_generator = EmbeddingGenerator()
        self.db = DatabaseConnection()

    def process_risk_assessment(self, input_text: str, top_k: int = 5) -> dict:
        """Process risk assessment request and return response"""
        try:
            # Connect to database
            self.db.connect()

            # Get embedding for input text
            query_embedding = self.embedding_generator.get_embedding(input_text)

            # Get similar documents
            related_docs = self.db.get_similar_documents(query_embedding, top_k)

            # Generate assistant response
            response = self._generate_response(input_text, related_docs, top_k)

            return json.loads(response)

        finally:
            self.db.close()

    def _generate_response(self, input_text: str, related_docs, top_k: int) -> str:
        """Generate GPT response based on input and similar documents"""
        system_message = """
        You are an AI assistant that provides risk assessments for motor insurance. 
        Your responses should follow this structure:
        1. Underwriting Decision: <decision>
        2. Risk Class: <risk_class>
        3. Reason for Decline: <reason>
        4. Vehicle information: <vehicle_information>
        Should include the price, prepared price, and any additional costs to maintain it
        Ensure that the structure is consistent and responses are concise.
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"```{input_text}```"},
            {
                "role": "assistant",
                "content": self._generate_dynamic_content(related_docs, top_k)
            }
        ]

        response = self.openai_client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        return response.to_json()

    def _generate_dynamic_content(self, related_docs, k: int) -> str:
        """Generate dynamic content based on similar cases"""
        content = "Based on similar cases, here are relevant risk assessments with additional vehicle details:\n"
        
        for i, doc in enumerate(related_docs[:k]):
            content += f"{i+1}. Case Details: {doc[0]}\n"
        
        return content