"""
Pytest configuration and fixtures for all tests.
"""

import os
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Set up test environment variables.
    This runs once before all tests.
    """
    # Set test environment variables
    os.environ["SECRET_KEY"] = "test_secret_key_minimum_32_characters_long_for_testing"
    os.environ["GEMINI_API_KEY"] = "AIzaSyTest_Key_For_Testing_Purposes_Only"
    os.environ["SUPABASE_URL"] = "https://test-project.supabase.co"
    os.environ["SUPABASE_KEY"] = "test_supabase_anon_key_for_testing"
    os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_service_role_key_for_testing"

    # Optional: Set other test-specific settings
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DEBUG"] = "false"

    yield

    # Cleanup after all tests (optional)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Fixture to provide environment variables for individual tests.
    Use this when you need to override env vars for specific tests.
    """
    monkeypatch.setenv("SECRET_KEY", "test_secret_key_minimum_32_characters_long")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey123456789")
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test_key_123456")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test_service_key_123456")

    return monkeypatch
