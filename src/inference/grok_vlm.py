"""
Grok VLM - xAI Grok Vision API wrapper implementing VLMInterface
Uses OpenAI-compatible API endpoint
"""

import os
import time
import base64
import io
from typing import List, Optional
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env.local
env_path = Path(__file__).parent.parent.parent / '.env.local'
load_dotenv(env_path)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from src.inference.vlm_interface import VLMInterface


class GrokVLM(VLMInterface):
    """xAI Grok Vision API wrapper using OpenAI-compatible endpoint"""

    # Available Grok vision models
    AVAILABLE_MODELS = {
        "grok-4": "Grok 4",
        "grok-4-fast": "Grok 4 Fast",
        "grok-2-vision-latest": "Grok 2 Vision (Latest)",
        "grok-2-vision-1212": "Grok 2 Vision (Dec 2024)",
    }

    # xAI API base URL
    BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        model_name: str = "grok-4",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Grok VLM

        Args:
            model_name: Grok model to use (default: grok-2-vision-latest)
            api_key: API key (defaults to XAI_API_KEY or GROK_API_KEY env var)
            max_retries: Maximum number of retries on API errors
            retry_delay: Initial delay between retries (exponential backoff)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.model_name_id = model_name
        self.display_name = self.AVAILABLE_MODELS.get(model_name, model_name)
        self.api_key = api_key or os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._loaded = False
        self._client = None

        if not self.api_key:
            raise ValueError("XAI_API_KEY or GROK_API_KEY not found in environment variables or .env.local")

    def load(self) -> None:
        """Initialize the OpenAI-compatible client for xAI"""
        if self._loaded:
            return

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.BASE_URL
        )

        self._loaded = True

    def unload(self) -> None:
        """Release resources"""
        self._client = None
        self._loaded = False

    def _image_to_base64_url(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()

        # Create data URL
        base64_data = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"

    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        Answer a single question about an image

        Args:
            image: PIL Image
            question: Question text

        Returns:
            Answer text
        """
        if not self._loaded:
            self.load()

        # Prepare the image as base64 URL
        image_url = self._image_to_base64_url(image)

        # Create the message with medical context
        messages = [
            {
                "role": "system",
                "content": "You are a medical imaging expert analyzing fetal ultrasound images. Provide clear, professional medical responses."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name_id,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.4
                )

                # Extract text from response
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                else:
                    return "No response generated."

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                continue

        raise RuntimeError(f"Grok API failed after {self.max_retries} attempts: {last_error}")

    def answer_batch(self, image: Image.Image, questions: List[str]) -> List[str]:
        """
        Answer multiple questions about an image

        Args:
            image: PIL Image
            questions: List of question texts

        Returns:
            List of answer texts
        """
        return [self.answer_question(image, q) for q in questions]

    @property
    def model_name(self) -> str:
        return self.display_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def create_grok_vlm(model: str = "grok-4") -> GrokVLM:
    """
    Factory function to create a Grok VLM instance

    Args:
        model: Model name to use

    Returns:
        GrokVLM instance
    """
    return GrokVLM(model_name=model)
