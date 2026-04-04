import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiUser, FiMail, FiSettings, FiShield, FiActivity } from 'react-icons/fi';
import { useAuthStore } from '../store/authStore';
import { useQueryStore } from '../store/queryStore';
import { useDocumentStore } from '../store/documentStore';
import toast from 'react-hot-toast';

export default function Profile() {
  const { user, updateUser } = useAuthStore();
  const { stats: queryStats, fetchStats } = useQueryStore();
  const { documents, fetchDocuments } = useDocumentStore();
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchStats();
    fetchDocuments(1, 1);
  }, [fetchStats, fetchDocuments]);

  const handleUpdateProfile = async (e) => {
    e.preventDefault();
    setLoading(true);
    try {
      // Update profile logic here
      toast.success('Profile updated');
    } catch (error) {
      toast.error('Update failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl font-bold gradient-text mb-2">Profile</h1>
        <p className="text-white/60">Manage your account settings</p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Profile Card */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card lg:col-span-1"
        >
          <div className="text-center">
            <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-gradient-to-br from-neon-blue via-neon-purple to-neon-pink flex items-center justify-center text-3xl font-bold neon-glow">
              {user?.email?.[0]?.toUpperCase() || 'U'}
            </div>
            <h2 className="text-2xl font-bold mb-2">{user?.email || 'User'}</h2>
            <div className="flex items-center justify-center gap-2 text-white/60 mb-6">
              <FiMail className="text-sm" />
              <span>{user?.email}</span>
            </div>

            <div className="space-y-3 text-left">
              <div className="flex items-center gap-3 p-3 glass rounded-lg">
                <FiShield className="text-neon-blue" />
                <div>
                  <p className="text-sm font-medium">Account Status</p>
                  <p className="text-xs text-white/60">Active</p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-3 glass rounded-lg">
                <FiActivity className="text-neon-green" />
                <div>
                  <p className="text-sm font-medium">Member Since</p>
                  <p className="text-xs text-white/60">
                    {user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'Today'}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Stats & Settings */}
        <div className="lg:col-span-2 space-y-6">
          {/* Usage Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card"
          >
            <h3 className="text-xl font-bold mb-4 gradient-text flex items-center gap-2">
              <FiActivity className="text-neon-blue" />
              Usage Statistics
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 glass rounded-lg text-center">
                <p className="text-3xl font-bold text-neon-blue mb-1">{documents.length}</p>
                <p className="text-sm text-white/60">Documents</p>
              </div>
              <div className="p-4 glass rounded-lg text-center">
                <p className="text-3xl font-bold text-neon-purple mb-1">
                  {queryStats?.total_queries || 0}
                </p>
                <p className="text-sm text-white/60">Queries</p>
              </div>
              <div className="p-4 glass rounded-lg text-center">
                <p className="text-3xl font-bold text-neon-pink mb-1">
                  {queryStats?.total_chunks || 0}
                </p>
                <p className="text-sm text-white/60">Chunks</p>
              </div>
              <div className="p-4 glass rounded-lg text-center">
                <p className="text-3xl font-bold text-neon-green mb-1">
                  {queryStats?.avg_similarity?.toFixed(2) || '0.00'}
                </p>
                <p className="text-sm text-white/60">Avg Similarity</p>
              </div>
            </div>
          </motion.div>

          {/* Settings Form */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card"
          >
            <h3 className="text-xl font-bold mb-4 gradient-text flex items-center gap-2">
              <FiSettings className="text-neon-purple" />
              Settings
            </h3>
            <form onSubmit={handleUpdateProfile} className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2 text-white/80">Email</label>
                <div className="relative">
                  <FiMail className="absolute left-4 top-1/2 transform -translate-y-1/2 text-white/40" />
                  <input
                    type="email"
                    defaultValue={user?.email}
                    className="input-field pl-12"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-white/80">Current Password</label>
                <input
                  type="password"
                  className="input-field"
                  placeholder="Enter current password"
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-2 text-white/80">New Password</label>
                <input
                  type="password"
                  className="input-field"
                  placeholder="Enter new password"
                />
              </div>

              <button type="submit" className="btn-primary w-full disabled:opacity-50" disabled={loading}>
                {loading ? (
                  <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin mx-auto" />
                ) : (
                  'Update Profile'
                )}
              </button>
            </form>
          </motion.div>

          {/* API Info */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="card"
          >
            <h3 className="text-xl font-bold mb-4 gradient-text flex items-center gap-2">
              <FiShield className="text-neon-green" />
              API Access
            </h3>
            <div className="p-4 glass rounded-lg font-mono text-sm break-all">
              <p className="text-white/60 mb-2">Your API Token:</p>
              <code className="text-neon-blue">
                {localStorage.getItem('access_token')?.substring(0, 50)}...
              </code>
            </div>
            <p className="text-xs text-white/40 mt-2">
              Use this token to authenticate API requests
            </p>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
