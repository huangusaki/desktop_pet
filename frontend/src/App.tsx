import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ChatWindow } from './components/ChatWindow';
import { ConfigPage } from './components/ConfigPage';
import './index.css';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<ChatWindow />} />
        <Route path="/config" element={<ConfigPage />} />
      </Routes>
    </Router>
  );
}

export default App;
