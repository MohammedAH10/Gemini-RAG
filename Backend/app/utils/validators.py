"""
Text Validation utilities for RAG processing
"""

import re
import unicodedata
from typing import Optional, Tuple

from loguru import logger


class TextValidator:
    """Validate and clean text contents"""


    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 10_000_000

    MIN_WORD_COUNT = 3

    @staticmethod
    def is_valid_text(text: str) -> Tuple[bool, Optional[str]]:
        """
        Check if text is valid for processing.
            Args:
                text: Text to validate

            Returns:
                Tuple of (is_valid, error_message)
        """

        if not text:
            return False, " Text is empty"

        if not isinstance(text, str):
            return False, "Text must be a string"

        # check length
        text_length = len(text)

        if text_length < TextValidator.MIN_TEXT_LENGTH:
            return False, f"Text too short (minimum {TextValidator.MIN_TEXT_LENGTH})"

        if text_length > TextValidator.MAX_TEXT_LENGTH:
            return False, f"Text too long (maximum {TextValidator.MAX_TEXT_LENGTH})"

        # check word count
        word_count = len(text.split())
        if word_count < TextValidator.MIN_WORD_COUNT:
            return False, f"Text has few words (minimum {TextValidator.MIN_WORD_COUNT})"

        # checking for whitespaces
        non_whitespace = len((re.sub(r"\s", "", text)))
        if non_whitespace < TextValidator.MIN_TEXT_LENGTH:
            return False, "Text contins mostly whitespaces"

        return True, None

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """

        if not text:
            return ""

        # Remove null bytes
        text = text.replace("\x00", "")

        # Normalize whitespace (but preserve single newlines)
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)

        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Remove leading and trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Unicode characters.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """

        # Normalize to NFC form (canonical composition)
        text = unicodedata.normalize("NFC", text)

        return text

    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """
        remove excessive whitespace while preserving struture

        Args:
            Text: text to process

        Returns:
            text with normalized whitespace
        """

        # Replace tabs with spaces
        text = text.replace("\t", " ")

        # Replace multiple spaces with single space
        text = re.sub(r" {2,}", " ", text)

        # Preserve paragraph breaks (double newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @staticmethod
    def validate_chunk_size(chunk_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate chunk size parameter.

        Args:
            chunk_size: Chunk size in tokens

        Returns:
            Tuple of (is_valid, error_message)
        """
        if chunk_size < 50:
            return False, "Chunk size must be at least 50 tokens"

        if chunk_size > 4000:
            return False, "Chunk size must not exceed 4000 tokens"

        return True, None

    @staticmethod
    def validate_chunk_overlap(
        chunk_overlap: int, chunk_size: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate chunk overlap parameter.

        Args:
            chunk_overlap: Overlap size in tokens
            chunk_size: Chunk size in tokens

        Returns:
            Tuple of (is_valid, error_message)
        """
        if chunk_overlap < 0:
            return False, "Chunk overlap must be non-negative"

        if chunk_overlap >= chunk_size:
            return False, "Chunk overlap must be less than Chunk Size"

        # Overlap should be reasonable (not too long)
        if chunk_overlap > chunk_size * 0.5:
            return False, "Chunk overlap should not exceed 50% of Chunk size"

        return True, None


def validate_text(text: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate text.

    Args:
        text: Text to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    return TextValidator.is_valid_text(text)


def clean_text(text: str) -> str:
    """
    Convenience function to clean text.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    return TextValidator.clean_text(text)
