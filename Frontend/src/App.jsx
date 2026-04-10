import { Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';

import Landing from './pages/Landing';
import SignUp from './pages/SignUp';
import SignIn from './pages/SignIn';
import GoogleCallback from './pages/GoogleCallback';
import Dashboard from './pages/Dashboard';
import Chat from './pages/Chat';
import Documents from './pages/Documents';
import DocumentViewer from './pages/DocumentViewer';
import DocumentUpload from './pages/DocumentUpload';
import Explore from './pages/Explore';
import VectorStore from './pages/VectorStore';
import Settings from './pages/Settings';
import History from './pages/History';
import Profile from './pages/Profile';
import Health from './pages/Health';
import NotFound from './pages/NotFound';

function App() {
  return (
    <>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'rgba(15, 33, 64, 0.95)',
            border: '1px solid rgba(79, 172, 254, 0.3)',
            color: '#e8f0fe',
            borderRadius: 'var(--radius-md)',
          },
        }}
      />
      <Routes>
        {/* Public Routes */}
        <Route path="/" element={<Landing />} />
        <Route path="/auth/signup" element={<SignUp />} />
        <Route path="/auth/signin" element={<SignIn />} />
        <Route path="/auth/google/callback" element={<GoogleCallback />} />
        <Route path="/health" element={<Health />} />

        {/* Protected Routes */}
        <Route
          path="/"
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="chat" element={<Chat />} />
          <Route path="documents" element={<Documents />} />
          <Route path="documents/:id" element={<DocumentViewer />} />
          <Route path="documents/upload" element={<DocumentUpload />} />
          <Route path="explore" element={<Explore />} />
          <Route path="vector-store" element={<VectorStore />} />
          <Route path="settings" element={<Settings />} />
          <Route path="history" element={<History />} />
          <Route path="profile" element={<Profile />} />
        </Route>

        {/* 404 */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  );
}

export default App;
