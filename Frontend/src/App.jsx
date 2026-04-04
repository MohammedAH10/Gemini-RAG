import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import { useAuthStore } from './store/authStore'
import Login from './pages/Login'
import Signup from './pages/Signup'
import GoogleCallback from './pages/GoogleCallback'
import AuthCallback from './pages/AuthCallback'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import Query from './pages/Query'
import Profile from './pages/Profile'
import MainLayout from './components/layout/MainLayout'
import AuthLayout from './components/layout/AuthLayout'
import ProtectedRoute from './components/auth/ProtectedRoute'

function App() {
  const { isAuthenticated, initializeAuth } = useAuthStore()

  // Re-initialize auth state from localStorage on app mount
  useEffect(() => {
    initializeAuth()
  }, [initializeAuth])

  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: 'rgba(15, 23, 42, 0.9)',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(0, 243, 255, 0.3)',
            color: '#fff',
          },
        }}
      />
      <Routes>
        {/* Landing / Home — show Landing if not authenticated, redirect to Dashboard if authenticated */}
        <Route
          path="/"
          element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Landing />}
        />

        {/* Auth Routes */}
        <Route element={<AuthLayout />}>
          <Route
            path="/login"
            element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Login />}
          />
          <Route
            path="/signup"
            element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Signup />}
          />
          <Route path="/auth/google/callback" element={<GoogleCallback />} />
        </Route>

        {/* Protected Routes */}
        <Route element={<ProtectedRoute />}>
          <Route element={<MainLayout />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/documents" element={<Documents />} />
            <Route path="/query" element={<Query />} />
            <Route path="/profile" element={<Profile />} />
          </Route>
        </Route>

        {/* Catch all — redirect to home */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  )
}

export default App
