from google import genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CHUNK_SIZE = 200  # Number of lines per AI review request

class AIReviewer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set!")

        self.client = genai.Client(api_key=self.api_key)

    def review_patch(self, filename, patch_text):
        """
        Reviews large patches by splitting them into chunks.
        Returns combined AI feedback as a single string.
        """
        lines = patch_text.splitlines()
        feedback_chunks = []

        # Process in chunks
        for i in range(0, len(lines), CHUNK_SIZE):
            chunk = "\n".join(lines[i:i + CHUNK_SIZE])
            prompt = f"Review the following code patch from {filename} for readability, bugs, and style:\n\n{chunk}"

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",  # Free tier
                contents=prompt,
            )

            feedback_chunks.append(response.text)

        # Combine feedback for all chunks
        return "\n---\n".join(feedback_chunks)
