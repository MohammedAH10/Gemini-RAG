# Google Authentication Setup

This document explains how to configure and use Google authentication for signup and login.

## Features Added

- ✅ Google sign-up button on the Signup page
- ✅ Google login button on the Login page  
- ✅ OAuth callback handler (`/auth/google/callback`)
- ✅ Redirect-based OAuth flow
- ✅ Session management with localStorage

## Backend Requirements

Your backend should have the following endpoints:

1. **`/auth/google`** - Initiates Google OAuth flow
   - Accepts `redirect_uri` query parameter
   - Redirects to Google's consent page
   - After consent, redirects back to `redirect_uri` with `code` parameter

2. **`/auth/google/callback`** (POST) - Exchanges authorization code for token
   - Request body: `{ "code": "<authorization_code>" }`
   - Response: `{ "access_token": "...", "user": { ... } }`

## Frontend Configuration

Create a `.env` file (copy from `.env.example`):

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
VITE_FRONTEND_URL=http://localhost:5173
```

Make sure `VITE_FRONTEND_URL` matches your frontend's URL exactly.

## How It Works

1. User clicks "Continue with Google"
2. Frontend redirects to backend's `/auth/google` endpoint
3. Backend redirects to Google's OAuth consent page
4. User authorizes the application
5. Google redirects back to `/auth/google/callback` with authorization code
6. Frontend sends code to backend's `/auth/google/callback` endpoint
7. Backend exchanges code for access token and user info
8. Frontend stores token and redirects to dashboard

## Files Modified

- `src/components/auth/SignupForm.jsx` - Added Google signup button
- `src/components/auth/LoginForm.jsx` - Added Google login button
- `src/store/authStore.js` - Added `googleAuth` function
- `src/services/auth.js` - Updated `googleAuth` and added `handleGoogleCallback`
- `src/App.jsx` - Added `/auth/google/callback` route
- `src/pages/GoogleCallback.jsx` - New OAuth callback handler
- `.env.example` - Added `VITE_FRONTEND_URL`

## Testing

1. Ensure your backend is running with Google OAuth configured
2. Start the frontend: `npm run dev`
3. Navigate to `/signup` or `/login`
4. Click "Continue with Google"
5. Complete the OAuth flow

## Troubleshooting

- **"Authentication cancelled"** - User closed the auth window or denied access
- **"Invalid authentication response"** - Missing code parameter in callback
- **"Google authentication failed"** - Backend error (check network tab for details)

Make sure your backend's Google OAuth credentials (Client ID, Client Secret) are properly configured.
