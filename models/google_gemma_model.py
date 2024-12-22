import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env file
load_dotenv()

class GemmaChecker:
    def __init__(self):
        # Retrieve API key from environment variables
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)

    def get_suggestions(self, tamil_text: str) -> str:
        """
        Generate spelling and grammar suggestions for Tamil text.
        
        Args:
            tamil_text (str): The Tamil text to analyze.
        
        Returns:
            str: Detailed corrections and suggestions in Tamil.
        """
        prompt = (
            f"As a Tamil language expert, analyze the following text for spelling and grammatical errors. "
            f"Provide detailed corrections and suggestions in Tamil:\n\n"
            f"Text to analyze: {tamil_text}\n\n"
            f"Please provide:\n"
            f"1. Corrected version of the text\n"
            f"2. Specific errors found\n"
            f"3. Explanation of corrections in Tamil"
        )
        
        try:
            # Request suggestions from the Groq API
            completion = self.client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Tamil language expert who provides detailed corrections and suggestions for Tamil text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2048
            )
            
            # Return the generated response
            return completion.choices[0].message.content
        
        except Exception as e:
            # Handle errors, including model decommission errors
            if hasattr(e, 'code') and e.code == 'model_decommissioned':
                return "Error: The model is no longer supported. Please contact the administrator to update the model."
            return f"Error getting suggestions: {str(e)}"

    def check_text(self, text: str) -> list:
        """
        Check the provided text for errors using the Groq API.
        
        Args:
            text (str): The Tamil text to check.
        
        Returns:
            list: A list of tuples containing the type of message, the content, and the original text.
        """
        try:
            suggestions = self.get_suggestions(text)
            if suggestions.startswith("Error"):
                return [("error", suggestions, text)]
            return [("info", suggestions, text)]
        
        except Exception as e:
            # Return error details if the process fails
            return [("error", f"Error checking text: {str(e)}", text)]
