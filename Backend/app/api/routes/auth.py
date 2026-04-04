"""
API routes for authentication.
"""
from fastapi import APIRouter, HTTPException, status, Depends, Query, Request
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials

from loguru import logger

from typing import Optional

from app.services.supabase_service import get_supabase_service, SupabaseService
from app.utils.auth import get_current_user, rate_limiter, security
from app.models.schemas import (
    UserSignUpRequest,
    UserSignInRequest,
    AuthResponse,
    RefreshTokenRequest,
    UserInfoResponse,
    RateLimitInfo,
    PasswordResetRequest
)

router = APIRouter()

# Cookie name for PKCE code_verifier persistence
PKCE_COOKIE_NAME = "pkce_code_verifier"


@router.post(
    "/signup",
    response_model=AuthResponse,
    summary="Register new user",
    description="Create a new user account"
)
async def sign_up(
    request: UserSignUpRequest,
    supabase: SupabaseService = Depends(get_supabase_service)
):
    """
    Register a new user.

    Args:
        request: Sign up request with email and password
        supabase: Supabase service

    Returns:
        Authentication response with tokens
    """
    logger.info(f"Sign up request for {request.email}")

    success, user_data, error = supabase.sign_up(
        email=request.email,
        password=request.password,
        metadata=request.metadata
    )

    if success and user_data:
        # Check if email confirmation is required
        if user_data.get("email_confirmed") == False:
            return AuthResponse(
                success=True,
                user_id=user_data["user_id"],
                email=user_data["email"],
                message=user_data.get("message", "Registration successful. Please check your email to confirm your account.")
            )
        
        return AuthResponse(
            success=True,
            user_id=user_data["user_id"],
            email=user_data["email"],
            access_token=user_data.get("access_token"),
            refresh_token=user_data.get("refresh_token"),
            message="User registered successfully"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Registration failed"
        )


@router.post(
    "/signin",
    response_model=AuthResponse,
    summary="Sign in user",
    description="Authenticate user and get access token"
)
async def sign_in(
    request: UserSignInRequest,
    supabase: SupabaseService = Depends(get_supabase_service)
):
    """
    Sign in user with email and password.
    
    Args:
        request: Sign in request
        supabase: Supabase service
        
    Returns:
        Authentication response with tokens
    """
    logger.info(f"Sign in request for {request.email}")
    
    success, session_data, error = supabase.sign_in(
        email=request.email,
        password=request.password
    )
    
    if success and session_data:
        return AuthResponse(
            success=True,
            user_id=session_data["user_id"],
            email=session_data["email"],
            access_token=session_data["access_token"],
            refresh_token=session_data["refresh_token"],
            expires_in=session_data.get("expires_in"),
            message="Signed in successfully"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error or "Invalid credentials"
        )


@router.post(
    "/signout",
    summary="Sign out user",
    description="Sign out current user"
)
async def sign_out(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: SupabaseService = Depends(get_supabase_service)
):
    """
    Sign out current user.

    Args:
        credentials: HTTP authorization credentials
        supabase: Supabase service

    Returns:
        Success message
    """
    logger.info("Sign out request")

    access_token = credentials.credentials
    success, error = supabase.sign_out(access_token)

    if success:
        return {"success": True, "message": "Signed out successfully"}
    else:
        return {"success": True, "message": "Sign out completed"}


@router.post(
    "/refresh",
    response_model=AuthResponse,
    summary="Refresh access token",
    description="Get new access token using refresh token"
)
async def refresh_token(
    request: RefreshTokenRequest,
    supabase: SupabaseService = Depends(get_supabase_service)
):
    """
    Refresh access token.
    
    Args:
        request: Refresh token request
        supabase: Supabase service
        
    Returns:
        New access token
    """
    logger.info("Token refresh request")
    
    success, session_data, error = supabase.refresh_session(request.refresh_token)
    
    if success and session_data:
        return AuthResponse(
            success=True,
            access_token=session_data["access_token"],
            refresh_token=session_data["refresh_token"],
            expires_in=session_data.get("expires_in"),
            message="Token refreshed successfully"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error or "Failed to refresh token"
        )


@router.get(
    "/me",
    response_model=UserInfoResponse,
    summary="Get current user info",
    description="Get information about the authenticated user"
)
async def get_user_info(
    user_id: str = Depends(get_current_user),
    supabase: SupabaseService = Depends(get_supabase_service)
):
    """
    Get current user information.
    
    Args:
        user_id: Current user ID
        supabase: Supabase service
        
    Returns:
        User information
    """
    # For now, return basic info from token
    # In production, fetch from database
    return UserInfoResponse(
        user_id=user_id,
        email="",  # Would be fetched from DB
        created_at="",
        metadata={}
    )


@router.get(
    "/rate-limit",
    response_model=RateLimitInfo,
    summary="Get rate limit info",
    description="Get current rate limit usage for authenticated user"
)
async def get_rate_limit_info(
    user_id: str = Depends(get_current_user),
    endpoint_type: str = "default"
):
    """
    Get rate limit information for current user.
    
    Args:
        user_id: Current user ID
        endpoint_type: Type of endpoint to check
        
    Returns:
        Rate limit information
    """
    usage = rate_limiter.get_usage(user_id, endpoint_type)
    
    return RateLimitInfo(**usage)


@router.post(
    "/reset-password",
    summary="Request password reset",
    description="Send password reset email"
)
async def reset_password(
    request: PasswordResetRequest,
    supabase: SupabaseService = Depends(get_supabase_service)
):
    """
    Request password reset email.
    
    Args:
        request: Password reset request
        supabase: Supabase service
        
    Returns:
        Success message
    """
    logger.info(f"Password reset request for {request.email}")
    
    success, error = supabase.reset_password_email(request.email)
    
    # Always return success for security (don't reveal if email exists)
    return {
        "success": True,
        "message": "If the email exists, a reset link has been sent"
    }


@router.get(
    "/verify",
    summary="Verify token",
    description="Verify that the current token is valid"
)
async def verify_token(user_id: str = Depends(get_current_user)):
    """
    Verify current token.
    
    Args:
        user_id: Current user ID from token
        
    Returns:
        Verification result
    """
    return {
        "valid": True,
        "user_id": user_id,
        "message": "Token is valid"
    }
    

@router.get(
    "/google",
    summary="Google OAuth sign in",
    description="Redirect to Google for OAuth authentication"
)
async def google_auth(
redirect_to: Optional[str] = Query(None, description="Redirect URL after auth"),
    supabase: SupabaseService = Depends(get_supabase_service)
):
    """
    Initiate Google OAuth flow.

    Args:
        redirect_to: Optional redirect URL after authentication
        supabase: Supabase service

    Returns:
        Redirect to Google OAuth
    """
    logger.info("Google OAuth request")

    # success, oauth_url, error, code_verifier = supabase.get_google_oauth_url_with_pkce(redirect_to)
    success, oauth_url, error = supabase.get_google_oauth_url(redirect_to)
    
    if success and oauth_url:
        return RedirectResponse(url=oauth_url)
        # # Persist PKCE code_verifier in a cookie so it survives reloads and redirects
        # if code_verifier:
        #     response.set_cookie(
        #         key=PKCE_COOKIE_NAME,
        #         value=code_verifier,
        #         httponly=True,
        #         samesite="lax",
        #         max_age=600,  # 10 minutes
        #         path="/",
        #     )
        #     logger.info("PKCE code_verifier stored in cookie")
        # return response
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error or "Failed to initiate Google OAuth"
        )


@router.get(
    "/google/callback",
    summary="Google OAuth callback",
    description="Handle Google OAuth callback and redirect to frontend dashboard"
)
async def google_callback(
    request: Request,
    code: str = Query(..., description="OAuth authorization code"),
    redirect_url: Optional[str] = Query(None, description="The redirect URL used in the OAuth flow"),
    supabase: SupabaseService = Depends(get_supabase_service)
):
    """
    Handle Google OAuth callback.

    After exchanging the code for a session, redirects the user back to
    the frontend dashboard with tokens embedded as query parameters.

    Args:
        code: OAuth authorization code
        redirect_url: The redirect URL used in the OAuth flow
        supabase: Supabase service

    Returns:
        Redirect to frontend dashboard with tokens
    """
    logger.info("Google OAuth callback with code: {code[:10]} ....")

    # # Use provided redirect_url or default
    # callback_url = redirect_url or "http://localhost:8000/api/v1/auth/google/callback"

    # # Read PKCE code_verifier from cookie (set during /google request)
    # code_verifier = request.cookies.get(PKCE_COOKIE_NAME)
    # if not code_verifier:
    #     logger.error("PKCE code_verifier not found in cookie")
    #     raise HTTPException(
    #         status_code=status.HTTP_400_BAD_REQUEST,
    #         detail="Missing PKCE code_verifier. Please restart the Google OAuth flow."
    #     )

    # success, session_data, error = supabase.exchange_code_for_session_with_pkce(code, callback_url, code_verifier)
    success, session_data, error = supabase.exchange_code_for_session(code)


    # Build frontend dashboard URL with tokens as query params
    # Determine frontend base URL from the Referer header or use default
    # referer = request.headers.get("referer", "")
    frontend_base = "http://localhost:5173" #update this later for production
    # if "localhost:3000" in referer:
    #     frontend_base = "http://localhost:3000"

    if success and session_data:
        from urllib.parse import urlencode

        # redirect to frontend with tokens as url params
        params = {
            "access_token": session_data["access_token"],
            "refresh_token": session_data["refresh_token"],
            "user_id": session_data["user_id"],
            "email": session_data["email"],
            "provider": "google",
        }
        callback_url = f"{frontend_base}/auth/google/callback?{urlencode(params)}"
        logger.info(f"Google OAuth success, redirecting to: {callback_url}")
        return RedirectResponse(url=callback_url, status_code=302)
    else:
        # redirect to login with error
        error_url = f"{frontend_base}/login?error={error or 'Google authentication failed'}"
        logger.info(f"Google OAuth failed, redirecting to: {error}")
        return RedirectResponse(url=error_url, status_code=302)


@router.post(
    "/google/mobile",
    response_model=AuthResponse,
    summary="Google OAuth for mobile",
    description="Sign in with Google ID token (for mobile apps)"
)
async def google_mobile_auth(
    id_token: str = Query(..., description="Google ID token"),
    supabase: SupabaseService = Depends(get_supabase_service)
):
    """
    Sign in with Google ID token (for mobile/SPAs).

    Args:
        id_token: Google ID token
        supabase: Supabase service

    Returns:
        Authentication response with tokens
    """
    logger.info("Google mobile auth request")

    try:
        response = supabase.client.auth.sign_in_with_id_token({
            "provider": "google",
            "token": id_token
        })

        if response.user and response.session:
            return AuthResponse(
                success=True,
                user_id=response.user.id,
                email=response.user.email,
                access_token=response.session.access_token,
                refresh_token=response.session.refresh_token,
                message="Google authentication successful"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Google authentication failed"
            )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Google mobile auth failed: {error_msg}")
        
        # Handle specific error types
        if "invalid_token" in error_msg.lower() or "invalid credential" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google ID token"
            )
        elif "expired" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Google ID token has expired"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Google authentication failed: {error_msg}"
            )
        
    