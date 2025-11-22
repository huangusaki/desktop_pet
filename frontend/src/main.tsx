import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
// import { disableAnimationsForPyQt } from './utils/platformDetector' // 已移除：不再禁用PyQt中的动画

// Detect PyQt environment and disable animations if needed
// disableAnimationsForPyQt(); // 已禁用：允许在PyQt中显示所有动画效果

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
