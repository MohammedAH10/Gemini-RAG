import { Navigate } from 'react-router-dom';
import useAuthStore from '../context/authStore';

export default function ProtectedRoute({ children }) {
  const { isAuthenticated } = useAuthStore();

  if (!isAuthenticated) {
    return <Navigate to="/auth/signin" replace />;
  }

  return children;
}
