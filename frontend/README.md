# Satellite Change Detection — Frontend

React + Vite dashboard for the AI-Powered Satellite Change Detection System.

## Features

- Upload a before/after satellite image pair and run the change-detection pipeline
- Staged pipeline progress display
- Change overlay / semantic segmentation visualization
- Appeared/removed object analysis with correct classes
- Automated JSON/PDF report downloads
- Analysis history dashboard

## Setup

```bash
npm install
npm run dev
```

Frontend: http://localhost:5173

The API base URL defaults to `http://127.0.0.1:8000`. Override it via the
`VITE_API_URL` environment variable if the backend is hosted elsewhere.
