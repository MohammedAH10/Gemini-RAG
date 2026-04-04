"""
Supabase service for authentication and database operations.
"""
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import secrets
import base64
import hashlib

from supabase import create_client, Client
from loguru import logger

from app.config import get_settings

settings = get_settings()


class SupabaseService:
    """
    Service for Supabase operations.
    Handles authentication, user management, and database operations.
    """
    
    def __init__(self):
        """Initialize Supabase client."""
        self.client: Client = create_client(
            settings.supabase_url,
            settings.supabase_key
        )
        
        # Admin client for privileged operations
        if settings.supabase_service_role_key:
            self.admin_client: Client = create_client(
                settings.supabase_url,
                settings.supabase_service_role_key
            )
        else:
            self.admin_client = None
        
        logger.info("SupabaseService initialized")
    
    # Authentication Methods
    
    def sign_up(self, email: str, password: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Register a new user.

        Args:
            email: User email
            password: User password
            metadata: Optional user metadata

        Returns:
            Tuple of (success, user_data, error_message)
        """
        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": metadata or {},
                    "email_redirect_to": settings.oauth_redirect_url or "http://localhost:8000/api/v1/auth/google/callback"
                }
            })

            if response.user:
                logger.info(f"User registered: {email}")
                
                # Check if email confirmation is required
                try:
                    has_identities = response.user.identities and len(response.user.identities) > 0
                    email_not_confirmed = not response.user.email_confirmed_at
                    
                    if has_identities and email_not_confirmed:
                        logger.info(f"User registered, email confirmation required: {email}")
                        return True, {
                            "user_id": response.user.id,
                            "email": response.user.email,
                            "access_token": response.session.access_token if response.session else None,
                            "refresh_token": response.session.refresh_token if response.session else None,
                            "created_at": datetime.now().isoformat(),
                            "email_confirmed": False,
                            "message": "Registration successful. Please check your email to confirm your account."
                        }, None
                except (AttributeError, TypeError):
                    # If we can't check identities, assume email confirmation is not required
                    logger.warning(f"Could not check email confirmation status for {email}")
                
                # User is already confirmed or email confirmation is disabled
                return True, {
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "access_token": response.session.access_token if response.session else None,
                    "refresh_token": response.session.refresh_token if response.session else None,
                    "created_at": datetime.now().isoformat(),
                    "email_confirmed": True
                }, None
            else:
                return False, None, "Registration failed"

        except Exception as e:
            error_msg = str(e)
            # Handle specific error types
            if "User already registered" in error_msg or "already been registered" in error_msg:
                error_msg = "An account with this email already exists"
            elif "Password should be at least" in error_msg:
                error_msg = "Password must be at least 8 characters long"
            elif "Invalid email" in error_msg:
                error_msg = "Please provide a valid email address"
            elif "Unable to validate email address" in error_msg:
                error_msg = "Email validation failed. Please check your email format"
            
            logger.error(f"Sign up failed for {email}: {error_msg}")
            return False, None, error_msg
    
    def sign_in(self, email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Sign in user with email and password.

        Args:
            email: User email
            password: User password

        Returns:
            Tuple of (success, session_data, error_message)
        """
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            if response.user and response.session:
                logger.info(f"User signed in: {email}")
                return True, {
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                    "expires_at": response.session.expires_at,
                    "expires_in": response.session.expires_in
                }, None
            else:
                return False, None, "Invalid credentials"

        except Exception as e:
            error_msg = str(e)
            # Handle specific error types
            if "Invalid login credentials" in error_msg:
                error_msg = "Invalid email or password"
            elif "Email not confirmed" in error_msg:
                error_msg = "Please verify your email address before signing in"
            elif "Too many requests" in error_msg:
                error_msg = "Too many login attempts. Please try again later"
            
            logger.error(f"Sign in failed for {email}: {error_msg}")
            return False, None, error_msg
    
    def sign_out(self, access_token: str) -> Tuple[bool, Optional[str]]:
        """
        Sign out user.

        Args:
            access_token: User's access token

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Sign out with the specific scope and token
            self.client.auth.sign_out({
                "scope": "local",
                "access_token": access_token
            })
            logger.info("User signed out")
            return True, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Sign out failed: {error_msg}")
            return False, error_msg
    
    def refresh_session(self, refresh_token: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Refresh user session.

        Args:
            refresh_token: Refresh token

        Returns:
            Tuple of (success, new_session_data, error_message)
        """
        try:
            # Set the session with refresh token first
            self.client.auth.set_session(refresh_token)
            
            # Then refresh it
            response = self.client.auth.refresh_session()

            if response.session:
                logger.info("Session refreshed")
                return True, {
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                    "expires_at": response.session.expires_at,
                    "expires_in": response.session.expires_in
                }, None
            else:
                return False, None, "Failed to refresh session"

        except Exception as e:
            error_msg = str(e)
            # Handle specific error types
            if "refresh_token_not_found" in error_msg or "Invalid refresh token" in error_msg:
                error_msg = "Invalid or expired refresh token. Please log in again"
            elif "Token has expired" in error_msg:
                error_msg = "Session expired. Please log in again"
            
            logger.error(f"Session refresh failed: {error_msg}")
            return False, None, error_msg
    
    def get_user(self, access_token: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Get user information from access token.

        Args:
            access_token: User's access token

        Returns:
            Tuple of (success, user_data, error_message)
        """
        try:
            # Get user directly from access token
            response = self.client.auth.get_user(access_token)

            if response.user:
                return True, {
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "created_at": response.user.created_at,
                    "metadata": response.user.user_metadata
                }, None
            else:
                return False, None, "User not found"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Get user failed: {error_msg}")
            return False, None, error_msg
    
    def verify_token(self, access_token: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Verify access token and extract user ID.
        
        Args:
            access_token: Access token to verify
            
        Returns:
            Tuple of (is_valid, user_id, error_message)
        """
        try:
            response = self.client.auth.get_user(access_token)
            
            if response.user:
                return True, response.user.id, None
            else:
                return False, None, "Invalid token"
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Token verification failed: {error_msg}")
            return False, None, error_msg
    
    # User Management
    
    def update_user_metadata(self, access_token: str, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Update user metadata.
        
        Args:
            access_token: User's access token
            metadata: Metadata to update
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            self.client.auth.update_user({
                "data": metadata
            })
            logger.info("User metadata updated")
            return True, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Metadata update failed: {error_msg}")
            return False, error_msg
    
    def reset_password_email(self, email: str) -> Tuple[bool, Optional[str]]:
        """
        Send password reset email.
        
        Args:
            email: User email
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            self.client.auth.reset_password_email(email)
            logger.info(f"Password reset email sent to {email}")
            return True, None
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Password reset failed: {error_msg}")
            return False, error_msg
    
    def get_google_oauth_url(self, redirect_to: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Get Google OAuth URL for sign in.

        Args:
            redirect_to: Optional redirect URL after authentication

        Returns:
            Tuple of (success, oauth_url, error_message)
        """
        try:
            # Use provided redirect_to, or fall back to config, or use default
            redirect_url = redirect_to or settings.oauth_redirect_url or "http://localhost:8000/api/v1/auth/google/callback"

            response = self.client.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {
                    "redirect_to": redirect_url
                }
            })

            if response.url:
                logger.info("Generated Google OAuth URL")
                return True, response.url, None
            else:
                return False, None, "Failed to generate OAuth URL"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Google OAuth URL generation failed: {error_msg}")
            return False, None, error_msg

    # def _generate_pkce_pair(self) -> Tuple[str, str]:
    #     """
    #     Generate a PKCE code_verifier and code_challenge pair.

    #     Returns:
    #         Tuple of (code_verifier, code_challenge)
    #     """
    #     # Generate a 32-byte random code_verifier, base64url-encoded
    #     code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b'=').decode('ascii')

    #     # code_challenge = BASE64URL(SHA256(code_verifier))
    #     code_challenge = (
    #         base64.urlsafe_b64encode(
    #             hashlib.sha256(code_verifier.encode('ascii')).digest()
    #         ).rstrip(b'=').decode('ascii')
    #     )

    #     return code_verifier, code_challenge

    # def get_google_oauth_url_with_pkce(self, redirect_to: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    #     """
    #     Get Google OAuth URL with manually generated PKCE parameters.

    #     We generate our own code_verifier/code_challenge pair instead of relying
    #     on the Supabase SDK's internal storage (which is unreliable server-side).

    #     Returns:
    #         Tuple of (success, oauth_url, error_message, code_verifier)
    #     """
    #     try:
    #         redirect_url = redirect_to or settings.oauth_redirect_url or "http://localhost:8000/api/v1/auth/google/callback"

    #         # Generate our own PKCE pair
    #         code_verifier, code_challenge = self._generate_pkce_pair()

    #         # Build the Supabase GoTrue authorize URL manually with our PKCE params
    #         from urllib.parse import urlencode

    #         base_url = f"{settings.supabase_url}/auth/v1/authorize"
    #         params = {
    #             "provider": "google",
    #             "redirect_to": redirect_url,
    #             "code_challenge": code_challenge,
    #             "code_challenge_method": "s256",
    #         }
    #         oauth_url = f"{base_url}?{urlencode(params)}"

    #         logger.info("Generated Google OAuth URL with PKCE (manually generated)")
    #         return True, oauth_url, None, code_verifier

    #     except Exception as e:
    #         error_msg = str(e)
    #         logger.error(f"Google OAuth URL generation failed: {error_msg}")
    #         return False, None, error_msg, None

    # def exchange_code_for_session_with_pkce(self, code: str, redirect_url: str, code_verifier: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    #     """
    #     Exchange OAuth code for session using our manually stored PKCE code_verifier.

    #     Calls the GoTrue /token endpoint directly with the code_verifier as a
    #     request parameter, bypassing the SDK's unreliable internal storage.

    #     Args:
    #         code: OAuth authorization code
    #         redirect_url: The redirect URL used in the OAuth flow
    #         code_verifier: PKCE code verifier (persisted via cookie)

    #     Returns:
    #         Tuple of (success, session_data, error_message)
    #     """
    #     try:
    #         import httpx

    #         token_url = f"{settings.supabase_url}/auth/v1/token"
    #         headers = {
    #             "Content-Type": "application/json",
    #             "apikey": settings.supabase_key,
    #         }
    #         payload = {
    #             "grant_type": "pkce",
    #             "code": code,
    #             "code_verifier": code_verifier,
    #             "redirect_to": redirect_url,
    #         }

    #         with httpx.Client() as client:
    #             response = client.post(token_url, json=payload, headers=headers)

    #         if response.status_code not in (200, 201):
    #             try:
    #                 error_detail = response.json().get("msg", response.text)
    #             except Exception:
    #                 error_detail = response.text
    #             logger.error(f"Token exchange failed: {response.status_code} - {error_detail}")
    #             return False, None, f"Token exchange failed: {error_detail}"

    #         session_data = response.json()
    #         access_token = session_data.get("access_token")
    #         refresh_token = session_data.get("refresh_token")
    #         expires_in = session_data.get("expires_in")
    #         expires_at = session_data.get("expires_at")
    #         user = session_data.get("user", {})

    #         user_id = user.get("id") or session_data.get("user_id")
    #         email = user.get("email") or session_data.get("email")

    #         if access_token and user_id:
    #             logger.info(f"OAuth session created for {email}")
    #             return True, {
    #                 "user_id": user_id,
    #                 "email": email,
    #                 "access_token": access_token,
    #                 "refresh_token": refresh_token,
    #                 "expires_at": expires_at,
    #                 "expires_in": expires_in,
    #                 "provider": "google"
    #             }, None
    #         else:
    #             return False, None, "Failed to create session: missing access_token or user_id"

    #     except Exception as e:
    #         error_msg = str(e)
    #         logger.error(f"OAuth code exchange failed: {error_msg}")
    #         return False, None, error_msg
        
        
    def exchange_code_for_session(self, code: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Exchange OAuth code for session.

        Args:
            code: OAuth authorization code
            redirect_url: The redirect URL used in the OAuth flow

        Returns:
            Tuple of (success, session_data, error_message)
        """
        try:
            # USe exchange_code_for_session method with auth_code grant type
            response = self.client.auth.exchange_code_for_session({
                "auth_code": code,
                # "redirect_to": redirect_url
            })

            if response.user and response.session:
                logger.info(f"OAuth session created for {response.user.email}")
                return True, {
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                    "expires_at": response.session.expires_at,
                    "expires_in": response.session.expires_in,
                    "provider": "google"
                }, None
            else:
                return False, None, "Failed to create session"

        except Exception as e:
            error_msg = str(e)
            logger.error(f"OAuth code exchange failed: {error_msg}")
            return False, None, error_msg

# Global instance
_supabase_service: Optional[SupabaseService] = None


def get_supabase_service() -> SupabaseService:
    """
    Get or create global Supabase service instance.
    
    Returns:
        SupabaseService instance
    """
    global _supabase_service
    
    if _supabase_service is None:
        _supabase_service = SupabaseService()
    
    return _supabase_service
