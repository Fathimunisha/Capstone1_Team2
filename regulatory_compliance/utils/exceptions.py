from fastapi import Request
from fastapi.responses import JSONResponse


class InvalidFileException(Exception):
    """
    Raised when uploaded file is not a PDF.
    """

    def __init__(self, message: str):
        self.message = message


class DocumentNotFoundException(Exception):
    """
    Raised when requested document is unavailable.
    """

    def __init__(self, message: str):
        self.message = message


class DatabaseException(Exception):
    """
    Raised when database operation fails.
    """

    def __init__(self, message: str):
        self.message = message


class LLMException(Exception):
    """
    Raised when LLM fails.
    """

    def __init__(self, message: str):
        self.message = message
