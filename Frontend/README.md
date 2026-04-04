# N.O.V.A.R - Network of Vectorized Archive Retrieval

A futuristic, responsive React frontend for your RAG (Retrieval-Augmented Generation) document system built with FastAPI.

## ✨ Features

- 🎨 **Futuristic UI Design** - Glass morphism, neon effects, smooth animations
- 📱 **Fully Responsive** - Works seamlessly on desktop, tablet, and mobile
- 🔐 **Authentication** - Login/Signup with JWT token management
- 📄 **Document Management** - Upload, view, search, and delete documents
- 💬 **AI Query Interface** - Ask questions with real-time responses and source citations
- 📊 **Dashboard Analytics** - Track usage statistics and recent activity
- 👤 **User Profile** - Manage account settings and view API token
- 🎭 **Smooth Animations** - Powered by Framer Motion
- 🔔 **Toast Notifications** - Real-time feedback on user actions
- 🌙 **Dark Theme** - Easy on the eyes with a cyberpunk aesthetic

## 🚀 Tech Stack

- **React 18** - UI library
- **Vite** - Build tool and dev server
- **TailwindCSS** - Utility-first CSS framework
- **Framer Motion** - Animation library
- **Zustand** - State management
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **React Markdown** - Markdown rendering
- **React Icons** - Icon library
- **React Hot Toast** - Notifications

## 📦 Installation

1. **Install dependencies:**
```bash
npm install
```

2. **Configure environment:**
```bash
cp .env.example .env
```

Edit `.env` with your backend API URL:
```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

## 🏃 Running the App

### Development Mode
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Production Build
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── auth/
│   │   │   ├── LoginForm.jsx          # Login form component
│   │   │   ├── SignupForm.jsx         # Registration form component
│   │   │   └── ProtectedRoute.jsx     # Route protection wrapper
│   │   ├── documents/
│   │   │   └── (future enhancements)
│   │   ├── query/
│   │   │   ├── QueryInput.jsx         # Query input with options
│   │   │   └── QueryResponse.jsx      # Response display with sources
│   │   ├── common/
│   │   │   └── (future enhancements)
│   │   └── layout/
│   │       ├── MainLayout.jsx         # Main app layout with sidebar
│   │       └── AuthLayout.jsx         # Auth pages layout with background
│   ├── pages/
│   │   ├── Login.jsx                  # Login page
│   │   ├── Signup.jsx                 # Signup page
│   │   ├── Dashboard.jsx              # Main dashboard with stats
│   │   ├── Documents.jsx              # Document management
│   │   ├── Query.jsx                  # Query interface
│   │   └── Profile.jsx                # User profile & settings
│   ├── services/
│   │   ├── api.js                     # Axios instance with interceptors
│   │   ├── auth.js                    # Authentication API calls
│   │   ├── documents.js               # Document API calls
│   │   └── query.js                   # Query API calls
│   ├── store/
│   │   ├── authStore.js               # Authentication state
│   │   ├── documentStore.js           # Document state
│   │   └── queryStore.js              # Query state
│   ├── App.jsx                        # Main app component with routes
│   ├── main.jsx                       # Entry point
│   └── index.css                      # Global styles with Tailwind
├── public/
├── .env                               # Environment variables
├── .env.example                       # Environment template
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js
```

## 🎨 Design System

### Colors
- **Primary**: Neon Blue (`#00f3ff`)
- **Secondary**: Neon Purple (`#b967ff`)
- **Accent**: Neon Pink (`#ff6bcb`)
- **Success**: Neon Green (`#00ff9d`)
- **Background**: Dark slate (`#020617` to `#0f172a`)

### Components
- **Glass Cards**: Semi-transparent with backdrop blur
- **Neon Borders**: Glowing borders on hover
- **Gradient Text**: Eye-catching text effects
- **Animated Backgrounds**: Floating orbs with blur effects

## 🔌 API Integration

The frontend connects to your FastAPI backend through the following endpoints:

### Authentication
- `POST /api/v1/auth/signup` - Register new user
- `POST /api/v1/auth/signin` - Login
- `POST /api/v1/auth/signout` - Logout
- `GET /api/v1/auth/me` - Get current user

### Documents
- `POST /api/v1/documents/upload` - Upload document
- `GET /api/v1/documents` - List documents (paginated)
- `GET /api/v1/documents/{id}` - Get document details
- `DELETE /api/v1/documents/{id}` - Delete document

### Query
- `POST /api/v1/query/ask` - Ask question (main RAG endpoint)
- `GET /api/v1/query/stats` - Get query statistics

## 🌟 Key Features

### 1. Dashboard
- Quick query interface
- Recent documents list
- Usage statistics cards
- Real-time metrics

### 2. Document Management
- Drag & drop file upload
- Document search and filtering
- Pagination support
- Bulk operations ready

### 3. Query Interface
- Rich text input with keyboard shortcuts
- Configurable result count
- Response time tracking
- Source citations with expand/collapse
- Query history

### 4. Profile Page
- User information display
- Usage statistics
- API token display
- Account settings

## 🔒 Security Features

- JWT token-based authentication
- Automatic token refresh
- Secure token storage in localStorage
- Protected routes
- Automatic logout on 401 errors
- CORS handling via Vite proxy

## 🎯 Future Enhancements

- [ ] Google OAuth integration
- [ ] Document preview
- [ ] Advanced query filters
- [ ] Batch query processing
- [ ] Export query results
- [ ] Real-time collaboration
- [ ] Document sharing
- [ ] Analytics dashboard
- [ ] Mobile app (React Native)

## 🐛 Troubleshooting

### Backend Connection Issues
Make sure your FastAPI backend is running on `http://localhost:8000`

```bash
# In your backend directory
uvicorn app.main:app --reload
```

### CORS Errors
The Vite dev server includes a proxy configuration. For production, ensure your backend has proper CORS headers configured.

### Build Errors
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

## 📝 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_BASE_URL` | Backend API URL | `http://localhost:8000/api/v1` |

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the N.O.V.A.R system.

## 🙏 Acknowledgments

- Built with React and Vite
- Styled with TailwindCSS
- Animated with Framer Motion
- Icons from React Icons

---

**Made with ❤️ for N.O.V.A.R - Network of Vectorized Archive Retrieval**
