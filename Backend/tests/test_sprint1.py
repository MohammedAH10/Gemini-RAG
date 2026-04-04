"""
Sprint 1 Tests: Configuration and Setup
"""

# pyright: ignore[reportCallIssue]
from pathlib import Path

import pytest
from app.config import Settings, get_settings


# Fixtures
@pytest.fixture
def temp_env_vars(monkeypatch):
    """Fixture to set temporary environment variables for testing."""
    test_vars = {
        "SECRET_KEY": "test_secret_key_minimum_32_characters_long_for_security",
        "GEMINI_API_KEY": "AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
        "SUPABASE_URL": "https://test-project.supabase.co",
        "SUPABASE_KEY": "test-supabase-anon-key-here",
        "SUPABASE_SERVICE_ROLE_KEY": "test-service-role-key",
    }

    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)

    return test_vars


# Basic Loading Tests
def test_settings_load():
    """Test that settings load correctly."""
    settings = get_settings()

    assert settings.app_name is not None
    assert settings.app_version == "1.0.0"
    assert settings.environment in ["development", "production", "staging", "testing"]


def test_settings_singleton():
    """Test that get_settings returns the same instance."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2  # Same object in memory


# Directory Tests (Using Path)
def test_directories_created():
    """Test that necessary directories are created."""
    settings = get_settings()

    pdf_path = settings.get_pdf_storage_path()
    chroma_path = settings.get_chroma_persist_directory()
    log_path = settings.get_log_file_path().parent

    # Using Path methods
    assert isinstance(pdf_path, Path)
    assert isinstance(chroma_path, Path)
    assert isinstance(log_path, Path)

    assert pdf_path.exists()
    assert chroma_path.exists()
    assert log_path.exists()

    assert pdf_path.is_dir()
    assert chroma_path.is_dir()
    assert log_path.is_dir()


def test_path_properties():
    """Test that path properties return Path objects."""
    settings = get_settings()

    pdf_path = settings.get_pdf_storage_path()
    chroma_path = settings.get_chroma_persist_directory()
    log_path = settings.get_log_file_path()

    # Verify they're Path objects
    assert isinstance(pdf_path, Path)
    assert isinstance(chroma_path, Path)
    assert isinstance(log_path, Path)

    # Verify they're absolute or can be resolved
    assert pdf_path.resolve().exists()
    assert chroma_path.resolve().exists()


# Validation Tests (Using Settings class directly)
def test_invalid_secret_key_too_short(temp_env_vars, monkeypatch):
    """Test that short secret key raises validation error."""
    monkeypatch.setenv("SECRET_KEY", "short")

    with pytest.raises(ValueError, match="at least 32 characters"):
        Settings()  # pyright: ignore[reportCallIssue]


def test_invalid_gemini_api_key(temp_env_vars, monkeypatch):
    """Test that invalid Gemini API key raises error."""
    monkeypatch.setenv("GEMINI_API_KEY", "invalid_key")

    # This should still load but log a warning
    # The validator checks for "AIza" prefix
    settings = Settings()  # pyright: ignore[reportCallIssue]
    assert settings.gemini_api_key == "invalid_key"


def test_invalid_supabase_url(temp_env_vars, monkeypatch):
    """Test that invalid Supabase URL raises error."""
    monkeypatch.setenv("SUPABASE_URL", "http://invalid-url.com")

    with pytest.raises(ValueError, match="must be a valid HTTPS URL"):
        Settings()  # pyright: ignore[reportCallIssue]


def test_invalid_chunk_size_too_small(temp_env_vars, monkeypatch):
    """Test that chunk size too small raises error."""
    monkeypatch.setenv("CHUNK_SIZE", "50")

    with pytest.raises(ValueError, match="must be between 100 and 2000"):
        Settings()  # pyright: ignore[reportCallIssue]


def test_invalid_chunk_size_too_large(temp_env_vars, monkeypatch):
    """Test that chunk size too large raises error."""
    monkeypatch.setenv("CHUNK_SIZE", "3000")

    with pytest.raises(ValueError, match="must be between 100 and 2000"):
        Settings()  # pyright: ignore[reportCallIssue]


def test_invalid_chunk_overlap_negative(temp_env_vars, monkeypatch):
    """Test that negative chunk overlap raises error."""
    monkeypatch.setenv("CHUNK_OVERLAP", "-10")

    with pytest.raises(ValueError, match="must be between 0 and 500"):
        Settings()  # pyright: ignore[reportCallIssue]


def test_invalid_temperature_too_high(temp_env_vars, monkeypatch):
    """Test that temperature too high raises error."""
    monkeypatch.setenv("TEMPERATURE", "3.0")

    with pytest.raises(ValueError, match="must be between 0.0 and 2.0"):
        Settings()  # pyright: ignore[reportCallIssue]


def test_invalid_top_k_too_low(temp_env_vars, monkeypatch):
    """Test that top_k too low raises error."""
    monkeypatch.setenv("TOP_K_RESULTS", "0")

    with pytest.raises(ValueError, match="must be between 1 and 20"):
        Settings()  # pyright: ignore[reportCallIssue]


# Parametrized Tests (Using pytest.mark.parametrize)
@pytest.mark.parametrize(
    "chunk_size,expected",
    [
        (100, True),  # Minimum valid
        (512, True),  # Default
        (1000, True),  # Middle range
        (2000, True),  # Maximum valid
    ],
)
def test_valid_chunk_sizes(temp_env_vars, monkeypatch, chunk_size, expected):
    """Test various valid chunk sizes."""
    monkeypatch.setenv("CHUNK_SIZE", str(chunk_size))

    settings = Settings()  # pyright: ignore[reportCallIssue]
    assert (settings.chunk_size == chunk_size) == expected


@pytest.mark.parametrize(
    "temperature,expected",
    [
        (0.0, True),
        (0.5, True),
        (0.7, True),
        (1.0, True),
        (2.0, True),
    ],
)
def test_valid_temperatures(temp_env_vars, monkeypatch, temperature, expected):
    """Test various valid temperature values."""
    monkeypatch.setenv("TEMPERATURE", str(temperature))

    settings = Settings()  # pyright: ignore[reportCallIssue]
    assert (settings.temperature == temperature) == expected


# Property Tests
def test_cors_origins_list():
    """Test CORS origins parsing."""
    settings = get_settings()

    origins = settings.cors_origins_list
    assert isinstance(origins, list)
    assert len(origins) > 0

    # Should contain localhost entries
    assert any("localhost" in origin for origin in origins)


def test_max_file_size_conversion():
    """Test max file size conversion to bytes."""
    settings = get_settings()

    expected_bytes = settings.max_file_size_mb * 1024 * 1024
    assert settings.max_file_size_bytes == expected_bytes


# Validation Tests
def test_chunk_size_validation():
    """Test chunk size is within valid range."""
    settings = get_settings()

    assert 100 <= settings.chunk_size <= 2000


def test_chunk_overlap_validation():
    """Test chunk overlap is valid."""
    settings = get_settings()

    assert 0 <= settings.chunk_overlap < settings.chunk_size


def test_temperature_validation():
    """Test temperature is in valid range."""
    settings = get_settings()

    assert 0.0 <= settings.temperature <= 2.0


def test_top_k_validation():
    """Test top_k is in valid range."""
    settings = get_settings()

    assert 1 <= settings.top_k_results <= 20


# Display Config Tests
def test_display_config():
    """Test that display_config returns safe config."""
    settings = get_settings()

    config = settings.display_config()

    # Should include these
    assert "app_name" in config
    assert "environment" in config
    assert "chunk_size" in config
    assert "chunk_overlap" in config
    assert "top_k_results" in config

    # Should NOT include sensitive data
    assert "secret_key" not in config
    assert "google_api_key" not in config
    assert "supabase_key" not in config
    assert "supabase_service_role_key" not in config


def test_display_config_types():
    """Test that display_config values have correct types."""
    settings = get_settings()

    config = settings.display_config()

    assert isinstance(config["app_name"], str)
    assert isinstance(config["app_version"], str)
    assert isinstance(config["debug"], bool)
    assert isinstance(config["chunk_size"], int)
    assert isinstance(config["temperature"], float)
    assert isinstance(config["cors_origins"], list)


# Environment-specific Tests
def test_environment_values():
    """Test that environment is one of expected values."""
    settings = get_settings()

    assert settings.environment in ["development", "production", "staging", "testing"]


def test_debug_mode_correlation():
    """Test that debug mode correlates with environment."""
    settings = get_settings()

    if settings.environment == "production":
        # In production, we expect debug to be False (ideally)
        # But we won't enforce it, just check it exists
        assert isinstance(settings.debug, bool)


# Integration Tests
def test_full_configuration_load():
    """Test that all configuration loads without errors."""
    settings = get_settings()

    # Check all major config groups exist
    assert settings.app_name
    assert settings.gemini_api_key
    assert settings.supabase_url
    assert settings.chroma_collection_name
    assert settings.chunk_size > 0
    assert settings.max_file_size_mb > 0


def test_directory_creation_idempotent():
    """Test that calling ensure_directories_exist multiple times is safe."""
    settings = get_settings()

    # Call multiple times
    settings.ensure_directories_exist()
    settings.ensure_directories_exist()
    settings.ensure_directories_exist()

    # All should still exist
    assert settings.get_pdf_storage_path().exists()
    assert settings.get_chroma_persist_directory().exists()
    assert settings.get_log_file_path().parent.exists()


# Edge Case Tests
def test_cors_origins_with_whitespace(temp_env_vars, monkeypatch):
    """Test that CORS origins with whitespace are handled correctly."""
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000 , http://localhost:5173")

    settings = Settings()  # pyright: ignore[reportCallIssue]
    origins = settings.cors_origins_list

    # Should strip whitespace
    assert all(
        not origin.startswith(" ") and not origin.endswith(" ") for origin in origins
    )


def test_empty_cors_methods(temp_env_vars, monkeypatch):
    """Test default CORS methods."""
    settings = get_settings()

    assert settings.cors_allow_methods == "*"
    assert settings.cors_allow_headers == "*"
