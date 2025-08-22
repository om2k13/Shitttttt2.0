# Frontend - Code Review Agent

This is the React frontend for the Code Review Agent system.

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Backend Connection

The frontend expects a backend server running at `http://localhost:8000`. The frontend will only display real, live data from the backend.

## Running the System

To use the full functionality:

1. **Start the backend server:**
   ```bash
   cd ../backend
   python -m uvicorn app.main:app --reload
   ```

2. **Start the frontend:**
   ```bash
   npm run dev
   ```

## Real Data Only

The frontend is designed to work exclusively with real backend data:

- **No Mock Data** - All data comes directly from the backend
- **Live Updates** - Real-time data refresh and status monitoring
- **Error Handling** - Proper error handling for API failures
- **Data Validation** - Type-safe data handling with TypeScript

## Development

### Project Structure

```
src/
├── components/          # Reusable UI components
├── pages/              # Page components
├── lib/                # Utilities and API client
├── types/              # TypeScript type definitions
└── App.tsx            # Main application component
```

### Key Components

- **Dashboard** - Main overview with quick actions
- **CodeReview** - Code Review Agent interface
- **UserManagement** - User and organization management
- **WorkspaceManagement** - Repository workspace management
- **BackendStatus** - Backend connection status indicator

### API Integration

The frontend uses:
- **Axios** for HTTP requests
- **React Query** for data fetching and caching
- **Real-time data** from backend APIs

### Styling

- **Tailwind CSS** for styling
- **Lucide React** for icons
- **Custom components** for consistent UI

## Troubleshooting

### Common Issues

1. **Backend Connection Errors**
   - Check if backend server is running
   - Verify backend is on port 8000
   - Check backend logs for errors

2. **Proxy Errors in Vite**
   - These indicate backend connection issues
   - Ensure backend server is running
   - Check network connectivity

3. **TypeScript Errors**
   - Run `npm run build` to check for type issues
   - Ensure all dependencies are installed

### Getting Help

- Check the backend logs for server issues
- Verify network connectivity
- Ensure ports 3000 (frontend) and 8000 (backend) are available

## Build for Production

```bash
# Build the application
npm run build

# Preview production build
npm run preview
```

The built files will be in the `dist/` directory.
