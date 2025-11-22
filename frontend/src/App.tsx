import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ChatWindow } from './components/ChatWindow';
import { ConfigPage } from './components/ConfigPage';
import { FriendsPage } from './components/FriendsPage';
import { useWebSocket } from './hooks/useWebSocket';
import './index.css';

function App() {
  // Initialize WebSocket at app level - persists across route changes
  useWebSocket();

  return (
    <Router>
      <Routes>
        <Route path="/" element={<ChatWindow />} />
        <Route path="/config" element={<ConfigPage />} />
        <Route path="/friends" element={<FriendsPage />} />
      </Routes>
    </Router>
  );
}

export default App;
