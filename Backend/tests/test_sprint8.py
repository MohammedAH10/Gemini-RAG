"""
Sprint 8 Tests: Supabase Authentication & Rate Limiting
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock

# Set environment variables
os.environ["SECRET_KEY"] = "test_secret_key_minimum_32_characters_long"
os.environ["GEMINI_API_KEY"] = "AIzaSyTestKey123456789"
os.environ["SUPABASE_URL"] = "https://test.supabase.co"
os.environ["SUPABASE_KEY"] = "test_key_123456"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "test_service_key_123456"

from app.services.supabase_service import SupabaseService
from app.utils.auth import RateLimiter


@pytest.fixture
def mock_supabase_client():
    """Create mock Supabase client."""
    client = Mock()
    
    # Mock auth responses
    mock_user = Mock()
    mock_user.id = "test_user_123"
    mock_user.email = "test@example.com"
    mock_user.created_at = "2024-01-01T00:00:00Z"
    mock_user.user_metadata = {}
    
    mock_session = Mock()
    mock_session.access_token = "test_access_token"
    mock_session.refresh_token = "test_refresh_token"
    mock_session.expires_at = 3600
    mock_session.expires_in = 3600
    
    mock_response = Mock()
    mock_response.user = mock_user
    mock_response.session = mock_session
    
    client.auth.sign_up.return_value = mock_response
    client.auth.sign_in_with_password.return_value = mock_response
    client.auth.get_user.return_value = mock_response
    client.auth.refresh_session.return_value = mock_response
    
    return client


# Supabase Service Tests
def test_supabase_service_initialization():
    """Test Supabase service initialization."""
    with patch('app.services.supabase_service.create_client'):
        service = SupabaseService()
        assert service.client is not None


def test_sign_up_success(mock_supabase_client):
    """Test successful user registration."""
    with patch('app.services.supabase_service.create_client', return_value=mock_supabase_client):
        service = SupabaseService()
        
        success, user_data, error = service.sign_up(
            email="test@example.com",
            password="password123"
        )
        
        assert success
        assert user_data is not None
        assert user_data["email"] == "test@example.com"
        assert error is None


def test_sign_up_failure():
    """Test failed user registration."""
    mock_client = Mock()
    mock_client.auth.sign_up.side_effect = Exception("Email already exists")
    
    with patch('app.services.supabase_service.create_client', return_value=mock_client):
        service = SupabaseService()
        
        success, user_data, error = service.sign_up(
            email="test@example.com",
            password="password123"
        )
        
        assert not success
        assert user_data is None
        assert error is not None


def test_sign_in_success(mock_supabase_client):
    """Test successful user sign in."""
    with patch('app.services.supabase_service.create_client', return_value=mock_supabase_client):
        service = SupabaseService()
        
        success, session_data, error = service.sign_in(
            email="test@example.com",
            password="password123"
        )
        
        assert success
        assert session_data is not None
        assert "access_token" in session_data
        assert error is None


def test_sign_in_invalid_credentials():
    """Test sign in with invalid credentials."""
    mock_client = Mock()
    mock_client.auth.sign_in_with_password.side_effect = Exception("Invalid credentials")
    
    with patch('app.services.supabase_service.create_client', return_value=mock_client):
        service = SupabaseService()
        
        success, session_data, error = service.sign_in(
            email="test@example.com",
            password="wrong_password"
        )
        
        assert not success
        assert session_data is None
        assert error is not None


def test_verify_token_valid(mock_supabase_client):
    """Test token verification with valid token."""
    with patch('app.services.supabase_service.create_client', return_value=mock_supabase_client):
        service = SupabaseService()
        
        is_valid, user_id, error = service.verify_token("valid_token")
        
        assert is_valid
        assert user_id == "test_user_123"
        assert error is None


def test_verify_token_invalid():
    """Test token verification with invalid token."""
    mock_client = Mock()
    mock_client.auth.get_user.side_effect = Exception("Invalid token")
    
    with patch('app.services.supabase_service.create_client', return_value=mock_client):
        service = SupabaseService()
        
        is_valid, user_id, error = service.verify_token("invalid_token")
        
        assert not is_valid
        assert user_id is None
        assert error is not None


def test_refresh_session(mock_supabase_client):
    """Test session refresh."""
    with patch('app.services.supabase_service.create_client', return_value=mock_supabase_client):
        service = SupabaseService()
        
        success, session_data, error = service.refresh_session("refresh_token")
        
        assert success
        assert session_data is not None
        assert "access_token" in session_data


# Rate Limiter Tests
def test_rate_limiter_initialization():
    """Test rate limiter initialization."""
    limiter = RateLimiter()
    assert limiter is not None
    assert "query" in limiter.limits


def test_rate_limit_allows_requests():
    """Test that rate limiter allows requests under limit."""
    limiter = RateLimiter()
    
    for i in range(5):
        is_allowed, error = limiter.is_allowed("user_123", "query")
        assert is_allowed
        assert error is None


def test_rate_limit_blocks_excess_requests():
    """Test that rate limiter blocks requests over limit."""
    limiter = RateLimiter()
    
    # Set very low limit for testing
    limiter.limits["test"] = {"requests": 3, "window": 60}
    
    # First 3 requests should succeed
    for i in range(3):
        is_allowed, error = limiter.is_allowed("user_123", "test")
        assert is_allowed
    
    # 4th request should fail
    is_allowed, error = limiter.is_allowed("user_123", "test")
    assert not is_allowed
    assert "Rate limit exceeded" in error


def test_rate_limit_different_users():
    """Test that rate limits are per-user."""
    limiter = RateLimiter()
    
    # User 1 makes requests
    for i in range(5):
        is_allowed, _ = limiter.is_allowed("user_1", "query")
        assert is_allowed
    
    # User 2 should have separate limit
    is_allowed, _ = limiter.is_allowed("user_2", "query")
    assert is_allowed


def test_rate_limit_different_endpoints():
    """Test that different endpoints have different limits."""
    limiter = RateLimiter()
    
    # Make requests to query endpoint
    for i in range(5):
        is_allowed, _ = limiter.is_allowed("user_123", "query")
        assert is_allowed
    
    # Upload endpoint should have separate limit
    is_allowed, _ = limiter.is_allowed("user_123", "upload")
    assert is_allowed


def test_rate_limit_usage_stats():
    """Test getting rate limit usage statistics."""
    limiter = RateLimiter()
    
    # Make some requests
    for i in range(5):
        limiter.is_allowed("user_123", "query")
    
    # Get usage stats
    usage = limiter.get_usage("user_123", "query")
    
    assert usage["current_requests"] == 5
    assert usage["max_requests"] == 20  # Default query limit
    assert usage["remaining"] == 15


def test_rate_limit_window_expiry():
    """Test that old requests expire from window."""
    import time
    
    limiter = RateLimiter()
    
    # Set short window for testing
    limiter.limits["test"] = {"requests": 2, "window": 1}  # 1 second window
    
    # Make 2 requests
    limiter.is_allowed("user_123", "test")
    limiter.is_allowed("user_123", "test")
    
    # Should be at limit
    is_allowed, _ = limiter.is_allowed("user_123", "test")
    assert not is_allowed
    
    # Wait for window to expire
    time.sleep(1.1)
    
    # Should be allowed again
    is_allowed, _ = limiter.is_allowed("user_123", "test")
    assert is_allowed


# Integration Tests
def test_sign_out(mock_supabase_client):
    """Test user sign out."""
    with patch('app.services.supabase_service.create_client', return_value=mock_supabase_client):
        service = SupabaseService()
        
        success, error = service.sign_out("access_token")
        
        # Sign out should succeed (or at least not fail)
        assert error is None or success


def test_password_reset(mock_supabase_client):
    """Test password reset email."""
    with patch('app.services.supabase_service.create_client', return_value=mock_supabase_client):
        service = SupabaseService()
        
        success, error = service.reset_password_email("test@example.com")
        
        # Should succeed (or at least not crash)
        assert error is None or success