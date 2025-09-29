"""
OpenAI API Integration for Enhanced Response Generation
Provides more natural and contextual responses using GPT models
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import openai
from openai import OpenAI
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env.local
env_path = Path(__file__).parent.parent.parent / '.env.local'
load_dotenv(env_path)

# Import model configuration
from .model_config import (
    get_model_config,
    validate_model,
    DEFAULT_MODEL,
    MEDICAL_TEMPERATURE,
    MEDICAL_MAX_TOKENS
)

logger = logging.getLogger(__name__)


@dataclass
class AnalysisContext:
    """Context for generating AI responses"""
    predicted_class: str
    confidence: float
    class_description: str
    visible_structures: List[str]
    clinical_purpose: str
    measurements: List[str]
    top_3_predictions: Optional[List[Tuple[str, float]]] = None
    additional_findings: Optional[Dict] = None


class OpenAIResponseGenerator:
    """Generate sophisticated responses using OpenAI GPT models"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = MEDICAL_TEMPERATURE,
        max_tokens: int = MEDICAL_MAX_TOKENS
    ):
        """
        Initialize OpenAI response generator

        Args:
            api_key: OpenAI API key (uses env var if not provided)
            model: GPT model to use (defaults to gpt-4o-mini)
            temperature: Response randomness (0-1, lower for medical)
            max_tokens: Maximum response length
        """
        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

        # Get model configuration from environment or use defaults
        self.model = os.getenv("MODEL_NAME", model)
        self.temperature = float(os.getenv("TEMPERATURE", str(temperature)))
        self.max_tokens = int(os.getenv("MAX_TOKENS", str(max_tokens)))

        # Validate model
        if not validate_model(self.model):
            logger.warning(f"Unknown model {self.model}, falling back to {DEFAULT_MODEL}")
            self.model = DEFAULT_MODEL

        # Get model configuration for logging
        model_config = get_model_config(self.model)
        if self.client and model_config:
            logger.info(f"OpenAI configured with {model_config.display_name}")
            logger.info(f"  Context window: {model_config.context_window:,} tokens")
            logger.info(f"  Optimized for: {model_config.recommended_for}")

        # System prompt for medical accuracy and appropriate tone
        self.system_prompt = """You are an AI assistant helping interpret fetal ultrasound images for educational purposes.
        Your responses should be:
        1. Medically accurate but accessible to non-medical users
        2. Clear about the limitations of AI analysis
        3. Encouraging users to consult healthcare professionals
        4. Professional yet friendly in tone
        5. Focused on explaining what the image shows and its clinical relevance

        Important guidelines:
        - Never diagnose conditions or abnormalities
        - Always include appropriate disclaimers
        - Use medical terms but explain them simply
        - Be helpful but not prescriptive
        - Acknowledge uncertainty when confidence is low"""

    def is_available(self) -> bool:
        """Check if OpenAI API is configured and available"""
        return self.client is not None

    def generate_response(
        self,
        context: AnalysisContext,
        user_question: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate natural language response using GPT

        Args:
            context: Analysis context with classification results
            user_question: Optional specific question from user

        Returns:
            Generated response or None if API unavailable
        """
        if not self.is_available():
            logger.warning("OpenAI API not available")
            return None

        try:
            # Build the prompt
            prompt = self._build_prompt(context, user_question)

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    def _build_prompt(
        self,
        context: AnalysisContext,
        user_question: Optional[str] = None
    ) -> str:
        """Build prompt for GPT model"""

        prompt_parts = [
            f"I've analyzed a fetal ultrasound image with the following results:",
            f"- Detected view: {context.predicted_class} ({context.class_description})",
            f"- Confidence level: {context.confidence:.1f}%",
            f"- Clinical purpose: {context.clinical_purpose}"
        ]

        if context.visible_structures:
            prompt_parts.append(f"- Typical structures visible: {', '.join(context.visible_structures)}")

        if context.measurements:
            prompt_parts.append(f"- Common measurements: {', '.join(context.measurements)}")

        # Add alternatives if confidence is low
        if context.confidence < 70 and context.top_3_predictions:
            alternatives = [f"{cls} ({conf:.1f}%)" for cls, conf in context.top_3_predictions[1:3]]
            prompt_parts.append(f"- Alternative possibilities: {', '.join(alternatives)}")

        prompt_parts.append("")  # Empty line

        if user_question:
            prompt_parts.append(f"The user asks: '{user_question}'")
            prompt_parts.append("Please provide a helpful response addressing their question while explaining what this ultrasound view shows.")
        else:
            prompt_parts.append("Please provide a clear, informative explanation of what this ultrasound view shows and its clinical significance.")

        # Add confidence-based instruction
        if context.confidence < 60:
            prompt_parts.append("\nNote: Given the low confidence, emphasize uncertainty and recommend professional verification.")
        elif context.confidence < 80:
            prompt_parts.append("\nNote: Given moderate confidence, acknowledge some uncertainty while being informative.")

        return "\n".join(prompt_parts)

    def generate_conversational_response(
        self,
        context: AnalysisContext,
        conversation_history: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Generate response considering conversation history

        Args:
            context: Current analysis context
            conversation_history: Previous messages in conversation

        Returns:
            Generated response or None if API unavailable
        """
        if not self.is_available():
            return None

        try:
            # Build messages including history
            messages = [{"role": "system", "content": self.system_prompt}]

            # Add conversation history
            for msg in conversation_history[-5:]:  # Keep last 5 exchanges
                messages.append(msg)

            # Add current analysis
            current_prompt = self._build_prompt(context)
            messages.append({"role": "user", "content": current_prompt})

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    def answer_followup_question(
        self,
        question: str,
        context: AnalysisContext,
        conversation_history: List[Dict[str, str]]
    ) -> Optional[str]:
        """
        Answer follow-up questions about the ultrasound

        Args:
            question: User's follow-up question
            context: Original analysis context
            conversation_history: Previous messages

        Returns:
            Answer to the question or None if API unavailable
        """
        if not self.is_available():
            return None

        try:
            # Build focused prompt for Q&A
            qa_prompt = f"""Based on the ultrasound analysis showing a {context.class_description} view
            with {context.confidence:.1f}% confidence, the user asks: "{question}"

            Context: This view is used for {context.clinical_purpose}.
            Visible structures typically include: {', '.join(context.visible_structures) if context.visible_structures else 'standard anatomical features'}.

            Please provide a clear, helpful answer while maintaining medical accuracy and appropriate disclaimers."""

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": qa_prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    def generate_educational_content(
        self,
        predicted_class: str,
        class_description: str
    ) -> Optional[str]:
        """
        Generate educational content about a specific ultrasound view

        Args:
            predicted_class: The anatomical view class
            class_description: Description of the view

        Returns:
            Educational content or None if API unavailable
        """
        if not self.is_available():
            return None

        try:
            prompt = f"""Please create educational content about the {class_description} ultrasound view.

            Include:
            1. What this view shows and why it's important
            2. What healthcare providers look for in this view
            3. When during pregnancy this scan is typically performed
            4. What measurements or assessments are made
            5. Simple explanation of the anatomy visible

            Keep the explanation accessible to expecting parents while being medically accurate.
            Format with clear sections and bullet points where appropriate."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,  # Lower temperature for educational content
                max_tokens=600
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None


class HybridResponseGenerator:
    """Combines template-based and OpenAI responses"""

    def __init__(
        self,
        use_openai: bool = True,
        openai_api_key: Optional[str] = None,
        fallback_to_templates: bool = True
    ):
        """
        Initialize hybrid response generator

        Args:
            use_openai: Whether to attempt OpenAI responses
            openai_api_key: OpenAI API key
            fallback_to_templates: Use templates if OpenAI fails
        """
        # Import template generator
        from .response_generator import ResponseGenerator

        self.template_generator = ResponseGenerator()
        self.openai_generator = None

        if use_openai:
            self.openai_generator = OpenAIResponseGenerator(api_key=openai_api_key)
            if self.openai_generator.is_available():
                logger.info("OpenAI integration enabled")
            else:
                logger.warning("OpenAI API not available, using template responses only")

        self.fallback_to_templates = fallback_to_templates

    def generate_response(
        self,
        predicted_class: str,
        confidence: float,
        top_3_predictions: Optional[List[Tuple[str, float]]] = None,
        user_question: Optional[str] = None,
        use_ai: bool = True
    ) -> str:
        """
        Generate response using best available method

        Args:
            predicted_class: Predicted anatomical view
            confidence: Prediction confidence
            top_3_predictions: Top predictions
            user_question: Optional user question
            use_ai: Whether to try AI response

        Returns:
            Generated response string
        """
        # Try OpenAI first if available and requested
        if use_ai and self.openai_generator and self.openai_generator.is_available():
            # Get class info for context
            class_info = self.template_generator.class_descriptions.get(predicted_class, {})

            context = AnalysisContext(
                predicted_class=predicted_class,
                confidence=confidence,
                class_description=class_info.get('name', predicted_class),
                visible_structures=class_info.get('structures', []),
                clinical_purpose=class_info.get('clinical_purpose', 'medical assessment'),
                measurements=class_info.get('measurements', []),
                top_3_predictions=top_3_predictions
            )

            ai_response = self.openai_generator.generate_response(context, user_question)

            if ai_response:
                return ai_response

        # Fallback to template response
        if self.fallback_to_templates:
            return self.template_generator.generate_response(
                predicted_class=predicted_class,
                confidence=confidence,
                top_3_predictions=top_3_predictions,
                include_details=True
            )

        return "Unable to generate response at this time."


# Example usage
if __name__ == "__main__":
    # Test hybrid generator
    print("TESTING HYBRID RESPONSE GENERATOR")
    print("=" * 50)

    generator = HybridResponseGenerator(
        use_openai=True,  # Will use templates if API key not set
        fallback_to_templates=True
    )

    # Test with high confidence
    response = generator.generate_response(
        predicted_class="Trans-thalamic",
        confidence=92.5,
        top_3_predictions=[
            ("Trans-thalamic", 92.5),
            ("Trans-ventricular", 5.2),
            ("Trans-cerebellum", 2.3)
        ],
        user_question="What measurements can be taken from this view?",
        use_ai=False  # Force template for testing
    )

    print("Template-based response:")
    print(response)

    # If OpenAI is configured, test AI response
    if generator.openai_generator and generator.openai_generator.is_available():
        print("\n" + "=" * 50)
        print("AI-enhanced response:")
        response = generator.generate_response(
            predicted_class="Trans-thalamic",
            confidence=92.5,
            top_3_predictions=[
                ("Trans-thalamic", 92.5),
                ("Trans-ventricular", 5.2),
                ("Trans-cerebellum", 2.3)
            ],
            user_question="What measurements can be taken from this view?",
            use_ai=True
        )
        print(response)