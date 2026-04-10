# Gemini RAG Frontend

A futuristic React-based frontend for the Gemini RAG (Retrieval-Augmented Generation) system.

## Tech Stack

- **React 18** — UI library
- **Vite** — Build tool & dev server
- **React Router v6** — Client-side routing
- **Zustand** — State management
- **Axios** — HTTP client
- **Framer Motion** — Animations
- **React Icons** — Icon library
- **React Hot Toast** — Notifications

## Theme

- Deep navy blue (`#0a1628`, `#0f2140`)
- Electric light blue (`#4facfe`, `#00f2fe`)
- Deep purple (`#7c3aed`)
- Light pink (`#f9a8d4`, `#f472b6`)

## Getting Started

```bash
# Install dependencies
npm install

# Copy env file
cp .env.example .env

# Start dev server
npm run dev
```

## Project Structure

```
Frontend/
├── src/
│   ├── components/     # Reusable UI components
│   │   ├── Layout.jsx
│   │   ├── Sidebar.jsx
│   │   ├── Header.jsx
│   │   └── ProtectedRoute.jsx
│   ├── pages/          # Page components
│   │   ├── Landing.jsx
│   │   ├── SignUp.jsx
│   │   ├── SignIn.jsx
│   │   ├── GoogleCallback.jsx
│   │   ├── Dashboard.jsx
│   │   ├── Chat.jsx
│   │   ├── Documents.jsx
│   │   ├── DocumentViewer.jsx
│   │   ├── DocumentUpload.jsx
│   │   ├── Explore.jsx
│   │   ├── VectorStore.jsx
│   │   ├── Settings.jsx
│   │   ├── History.jsx
│   │   ├── Profile.jsx
│   │   ├── Health.jsx
│   │   └── NotFound.jsx
│   ├── services/       # API services
│   │   └── api.js
│   ├── context/        # State management
│   │   └── authStore.js
│   ├── styles/         # CSS modules & global styles
│   │   ├── index.css
│   │   ├── theme.js
│   │   └── *.module.css
│   ├── App.jsx
│   └── main.jsx
├── public/
├── package.json
├── vite.config.js
└── index.html
```

## Routes

| Path | Page | Auth |
|------|------|------|
| `/` | Landing | Public |
| `/auth/signup` | Sign Up | Public |
| `/auth/signin` | Sign In | Public |
| `/auth/google/callback` | OAuth Callback | Public |
| `/dashboard` | Dashboard | Protected |
| `/chat` | Chat | Protected |
| `/documents` | Documents List | Protected |
| `/documents/:id` | Document Viewer | Protected |
| `/documents/upload` | Upload | Protected |
| `/explore` | Explore/Search | Protected |
| `/vector-store` | Vector Store Manager | Protected |
| `/settings` | Settings | Protected |
| `/history` | History | Protected |
| `/profile` | Profile | Protected |
| `/health` | System Health | Public |

## API Integration

API endpoints are stubbed out with `TODO` comments. Wire them up by updating `src/services/api.js` and the respective page components.
