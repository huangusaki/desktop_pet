import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { disableAnimationsForPyQt } from './utils/platformDetector'

// Detect PyQt environment and disable animations if needed
disableAnimationsForPyQt();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
