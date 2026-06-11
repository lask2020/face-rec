import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Persons from './pages/Persons';
import PersonDetail from './pages/PersonDetail';
import Cameras from './pages/Cameras';
import Detections from './pages/Detections';
import Workers from './pages/Workers';
import { useWebSocket } from './hooks/useWebSocket';

export default function App() {
  const { events, connected } = useWebSocket();

  return (
    <Layout wsConnected={connected}>
      <Routes>
        <Route path="/" element={<Dashboard events={events} />} />
        <Route path="/persons" element={<Persons />} />
        <Route path="/persons/:id" element={<PersonDetail />} />
        <Route path="/cameras" element={<Cameras />} />
        <Route path="/detections" element={<Detections events={events} />} />
        <Route path="/workers" element={<Workers />} />
      </Routes>
    </Layout>
  );
}

