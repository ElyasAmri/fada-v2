"""
Question Loader - Dynamically load questions from Excel annotation files
Allows updating questions without code changes
"""

from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import re


class QuestionLoader:
    """Load and manage VQA questions from Excel annotation files"""

    def __init__(self, data_dir: str = "data/Fetal Ultrasound"):
        """
        Initialize question loader

        Args:
            data_dir: Directory containing Excel annotation files
        """
        self.data_dir = Path(data_dir)
        self._questions_cache: Optional[List[str]] = None
        self._category_files: Dict[str, Path] = {}
        self._scan_annotation_files()

    def _scan_annotation_files(self):
        """Scan for Excel annotation files"""
        for excel_file in self.data_dir.glob("*_image_list.xlsx"):
            # Extract category name (e.g., "Abodomen" from "Abodomen_image_list.xlsx")
            category = excel_file.stem.replace("_image_list", "")
            self._category_files[category] = excel_file

    def _extract_questions(self, df: pd.DataFrame) -> List[str]:
        """
        Extract question texts from DataFrame columns

        Args:
            df: DataFrame from Excel file

        Returns:
            List of question texts (without column prefixes like "Q1:\n")
        """
        questions = []

        for col in df.columns:
            # Match columns like "Q1:\nAnatomical Structures..."
            if col.startswith("Q") and ":" in col:
                # Extract the question text after "Q#:\n"
                match = re.match(r'Q\d+:\s*\n?(.*)', col)
                if match:
                    question_text = match.group(1).strip()
                    questions.append(question_text)

        return questions

    def get_questions(self, force_reload: bool = False) -> List[str]:
        """
        Get the standardized 8 questions

        Args:
            force_reload: Force reload from Excel files

        Returns:
            List of 8 question texts
        """
        if self._questions_cache is not None and not force_reload:
            return self._questions_cache

        # Load from first available annotation file
        if not self._category_files:
            # Fallback to hardcoded questions if no Excel files found
            return self._get_default_questions()

        # Get first Excel file
        excel_file = list(self._category_files.values())[0]

        try:
            df = pd.read_excel(excel_file)
            questions = self._extract_questions(df)

            if len(questions) != 8:
                print(f"Warning: Found {len(questions)} questions, expected 8. Using defaults.")
                return self._get_default_questions()

            self._questions_cache = questions
            return questions

        except Exception as e:
            print(f"Error loading questions from {excel_file}: {e}")
            return self._get_default_questions()

    def _get_default_questions(self) -> List[str]:
        """
        Get default fallback questions

        Returns:
            List of 8 default questions
        """
        return [
            "Anatomical Structures Identification: Identify and describe all anatomical structures visible in the image.",
            "Fetal Orientation: Determine the orientation of the fetus based on the image (e.g., head up/down, front/back view).",
            "Plane Evaluation: Assess if the image is taken at a standard diagnostic plane and describe its diagnostic relevance.",
            "Biometric Measurements: Identify any measurable biometric parameters (e.g., femur length, head circumference) from the image.",
            "Gestational Age: Estimate the gestational age of the fetus based on the visible features.",
            "Image Quality: Assess the quality of the ultrasound image, mentioning any factors that might affect its interpretation (e.g., clarity, artifacts).",
            "Normality / Abnormality: Determine whether the observed structures appear normal or identify any visible abnormalities or concerns.",
            "Clinical Recommendations: Provide any relevant clinical recommendations or suggested next steps based on your interpretation."
        ]

    def get_question_short_names(self) -> List[str]:
        """
        Get short names for questions (for UI display)

        Returns:
            List of 8 short question names
        """
        return [
            "Q1: Anatomical Structures",
            "Q2: Fetal Orientation",
            "Q3: Plane Evaluation",
            "Q4: Biometric Measurements",
            "Q5: Gestational Age",
            "Q6: Image Quality",
            "Q7: Normality/Abnormality",
            "Q8: Clinical Recommendations"
        ]

    def get_categories(self) -> List[str]:
        """
        Get list of available categories

        Returns:
            List of category names
        """
        return list(self._category_files.keys())

    def get_category_images(self, category: str) -> List[Path]:
        """
        Get list of images for a category

        Args:
            category: Category name (e.g., "Abodomen")

        Returns:
            List of image file paths
        """
        category_dir = self.data_dir / category
        if not category_dir.exists():
            return []

        # Get PNG images
        images = list(category_dir.glob("*.png"))
        images.sort()
        return images

    def reload(self):
        """Force reload questions from Excel files"""
        self._questions_cache = None
        self._category_files = {}
        self._scan_annotation_files()


# Singleton instance for easy access
_question_loader = None


def get_question_loader(data_dir: str = "data/Fetal Ultrasound") -> QuestionLoader:
    """
    Get singleton QuestionLoader instance

    Args:
        data_dir: Directory containing Excel annotation files

    Returns:
        QuestionLoader instance
    """
    global _question_loader
    if _question_loader is None:
        _question_loader = QuestionLoader(data_dir)
    return _question_loader
