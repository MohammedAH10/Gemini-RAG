# Quick Start Guide - N.O.V.A.R

## 🎉 Your Frontend is Ready!

The development server is running at: **http://localhost:3000**

## 📋 Getting Started

### 1. Start Your Backend
Make sure your FastAPI backend is running:
```bash
# In your backend directory
cd /path/to/your/backend
uvicorn app.main:app --reload --port 8000
```

### 2. Open the Frontend
Navigate to: **http://localhost:3000**

### 3. Create an Account
- Click "Sign up" on the login page
- Enter your email and password (min 8 characters)
- You'll be automatically logged in and redirected to the dashboard

### 4. Explore Features

#### Dashboard (`/dashboard`)
- View quick statistics
- Try the quick query feature
- See recent documents

#### Documents (`/documents`)
- Upload documents via drag & drop or file browser
- View all your uploaded documents
- Search and filter documents
- Delete documents you no longer need

#### Query (`/query`)
- Ask questions about your documents
- View AI-generated answers with source citations
- See response times and retrieval stats
- Browse query history

#### Profile (`/profile`)
- View your account information
- Check usage statistics
- Access your API token
- Update your settings

## 🎨 Features Overview

### Futuristic Design
- **Glass Morphism**: Semi-transparent cards with blur effects
- **Neon Effects**: Glowing borders and text effects
- **Smooth Animations**: Powered by Framer Motion
- **Dark Theme**: Cyberpunk-inspired color scheme

### Responsive Layout
- **Desktop**: Full sidebar navigation with collapsible menu
- **Tablet**: Optimized spacing and grid layouts
- **Mobile**: Hamburger menu with slide-out navigation

### Interactive Elements
- **Drag & Drop**: Easy file upload
- **Hover Effects**: Visual feedback on all interactive elements
- **Toast Notifications**: Real-time action feedback
- **Loading States**: Spinners and disabled states during async operations

## 🔧 Configuration

### Change API URL
Edit `.env` file:
```env
VITE_API_BASE_URL=http://your-backend-url:8000/api/v1
```

### Production Build
```bash
npm run build
```

The built files will be in the `dist/` folder, ready to deploy.

## 🌐 Deployment Options

### Static Hosting (Netlify, Vercel, etc.)
```bash
npm run build
# Upload dist/ folder to your hosting provider
```

### Docker
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
FROM nginx:alpine
COPY --from=0 /app/dist /usr/share/nginx/html
EXPOSE 80
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/dist;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🐛 Common Issues

### 1. Backend Connection Error
**Problem**: Can't connect to backend
**Solution**: 
- Ensure backend is running on port 8000
- Check `.env` file has correct API URL
- Verify CORS is enabled on backend

### 2. Login Not Working
**Problem**: Can't login or signup
**Solution**:
- Check browser console for errors
- Verify backend auth endpoints work
- Ensure password meets requirements (8+ characters)

### 3. File Upload Fails
**Problem**: Can't upload documents
**Solution**:
- Check file format (PDF, TXT, DOCX, EPUB)
- Verify backend upload endpoint
- Check file size limits

## 📱 Mobile Usage

The app is fully responsive and works on:
- iOS Safari
- Android Chrome
- Tablet browsers
- Desktop browsers

## 🎯 Tips for Best Experience

1. **Use Chrome or Firefox** for best performance
2. **Keep backend running** while using the frontend
3. **Upload diverse documents** for better query results
4. **Check query history** to see past questions
5. **Use the dashboard** for quick queries

## 🚀 Next Steps

1. Upload your first document
2. Try asking a question
3. Explore the profile page
4. Test on mobile devices
5. Customize the theme in `tailwind.config.js`

---

**Enjoy your futuristic RAG interface!** 🎨✨
