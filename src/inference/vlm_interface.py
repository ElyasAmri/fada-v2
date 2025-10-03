"""
VLM Interface - Modular abstraction for local and API-based VLM inference
Allows easy swapping between local GPU inference and cloud API endpoints
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


class VLMInterface(ABC):
    """Abstract base class for VLM inference"""

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload the model from memory"""
        pass

    @abstractmethod
    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        Answer a single question about an image

        Args:
            image: PIL Image
            question: Question text

        Returns:
            Answer text
        """
        pass

    @abstractmethod
    def answer_batch(self, image: Image.Image, questions: List[str]) -> List[str]:
        """
        Answer multiple questions about an image

        Args:
            image: PIL Image
            questions: List of question texts

        Returns:
            List of answer texts
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model name"""
        pass

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether model is currently loaded"""
        pass


class LocalVLM(VLMInterface):
    """Local GPU-based VLM inference using HuggingFace models"""

    def __init__(self, model_id: str, display_name: str, use_4bit: bool = True):
        """
        Initialize local VLM

        Args:
            model_id: HuggingFace model ID (e.g., "openbmb/MiniCPM-V-2_6")
            display_name: Display name for UI (e.g., "MiniCPM-V-2.6")
            use_4bit: Whether to use 4-bit quantization
        """
        self.model_id = model_id
        self.display_name = display_name
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load model with quantization"""
        if self._loaded:
            return

        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        else:
            quantization_config = None

        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            _attn_implementation="eager"
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        self._loaded = True

    def unload(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False

    def answer_question(self, image: Image.Image, question: str) -> str:
        """Answer single question using model's chat interface"""
        if not self._loaded:
            raise RuntimeError(f"Model {self.display_name} is not loaded")

        # Prepare message based on model type
        if "minicpm" in self.model_id.lower():
            # MiniCPM-V format
            msgs = [{'role': 'user', 'content': [image, question]}]
            with torch.no_grad():
                response = self.model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=self.tokenizer
                )
        elif "qwen2-vl" in self.model_id.lower():
            # Qwen2-VL format
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]}]
            with torch.no_grad():
                response = self.model.chat(
                    messages=msgs,
                    tokenizer=self.tokenizer
                )
        elif "internvl" in self.model_id.lower():
            # InternVL format
            pixel_values = self.model.load_image(image).to(self.model.device)
            with torch.no_grad():
                response = self.model.chat(
                    tokenizer=self.tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=dict(max_new_tokens=512)
                )
        else:
            raise NotImplementedError(f"Model type {self.model_id} not yet supported")

        return response

    def answer_batch(self, image: Image.Image, questions: List[str]) -> List[str]:
        """Answer multiple questions sequentially"""
        return [self.answer_question(image, q) for q in questions]

    @property
    def model_name(self) -> str:
        return self.display_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class APIVLM(VLMInterface):
    """API-based VLM inference (future implementation for cloud deployment)"""

    def __init__(self, api_endpoint: str, api_key: str, model_name: str):
        """
        Initialize API-based VLM

        Args:
            api_endpoint: API endpoint URL
            api_key: API authentication key
            model_name: Model identifier
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.display_name = model_name
        self._loaded = False

    def load(self) -> None:
        """No-op for API (always available)"""
        self._loaded = True

    def unload(self) -> None:
        """No-op for API"""
        self._loaded = False

    def answer_question(self, image: Image.Image, question: str) -> str:
        """Send request to API endpoint"""
        import requests
        import io
        import base64

        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Send API request
        response = requests.post(
            self.api_endpoint,
            headers={'Authorization': f'Bearer {self.api_key}'},
            json={
                'model': self.display_name,
                'image': img_base64,
                'question': question
            }
        )
        response.raise_for_status()

        return response.json()['answer']

    def answer_batch(self, image: Image.Image, questions: List[str]) -> List[str]:
        """Batch API request (if supported by endpoint)"""
        import requests
        import io
        import base64

        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        # Send batch API request
        response = requests.post(
            f"{self.api_endpoint}/batch",
            headers={'Authorization': f'Bearer {self.api_key}'},
            json={
                'model': self.display_name,
                'image': img_base64,
                'questions': questions
            }
        )
        response.raise_for_status()

        return response.json()['answers']

    @property
    def model_name(self) -> str:
        return self.display_name

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class VLMManager:
    """Manager for multiple VLM models"""

    def __init__(self):
        self.models: Dict[str, VLMInterface] = {}

    def register_model(self, key: str, model: VLMInterface):
        """Register a VLM model"""
        self.models[key] = model

    def get_model(self, key: str) -> VLMInterface:
        """Get a registered model"""
        return self.models.get(key)

    def get_all_models(self) -> List[str]:
        """Get list of registered model keys"""
        return list(self.models.keys())

    def unload_all(self):
        """Unload all models from memory"""
        for model in self.models.values():
            if model.is_loaded:
                model.unload()


# Preset configurations for top models
def create_top_vlms(use_api: bool = False, api_endpoint: str = None, api_key: str = None) -> VLMManager:
    """
    Create VLM manager with top-3 models

    Args:
        use_api: Whether to use API endpoints instead of local inference
        api_endpoint: API endpoint URL (if use_api=True)
        api_key: API key (if use_api=True)

    Returns:
        VLMManager with registered models
    """
    manager = VLMManager()

    if use_api:
        # API-based models
        manager.register_model("minicpm", APIVLM(
            api_endpoint=f"{api_endpoint}/minicpm",
            api_key=api_key,
            model_name="MiniCPM-V-2.6"
        ))
        manager.register_model("qwen2vl", APIVLM(
            api_endpoint=f"{api_endpoint}/qwen2vl",
            api_key=api_key,
            model_name="Qwen2-VL-2B"
        ))
        manager.register_model("internvl2", APIVLM(
            api_endpoint=f"{api_endpoint}/internvl2",
            api_key=api_key,
            model_name="InternVL2-4B"
        ))
    else:
        # Local GPU models
        manager.register_model("minicpm", LocalVLM(
            model_id="openbmb/MiniCPM-V-2_6",
            display_name="MiniCPM-V-2.6",
            use_4bit=True
        ))
        manager.register_model("qwen2vl", LocalVLM(
            model_id="Qwen/Qwen2-VL-2B-Instruct",
            display_name="Qwen2-VL-2B",
            use_4bit=True
        ))
        manager.register_model("internvl2", LocalVLM(
            model_id="OpenGVLab/InternVL2-4B",
            display_name="InternVL2-4B",
            use_4bit=True
        ))

    return manager
