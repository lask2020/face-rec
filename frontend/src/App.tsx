import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Persons from './pages/Persons';
import PersonDetail from './pages/PersonDetail';
import Cameras from './pages/Cameras';
import Detections from './pages/Detections';
import LicensePlates from './pages/LicensePlates';
import Workers from './pages/Workers';
import Signage from './pages/Signage';
import ErrorBoundary from './components/ErrorBoundary';
import { useWebSocket } from './hooks/useWebSocket';

export default function App() {
  const { events, plateEvents, connected } = useWebSocket();

  return (
    <ErrorBoundary name="App Routing">
      <Routes>
        <Route path="/signage" element={<Signage events={events} />} />
        <Route
          path="*"
          element={
            <Layout wsConnected={connected}>
              <Routes>
                <Route path="/" element={<Dashboard events={events} />} />
                <Route path="/persons" element={<Persons />} />
                <Route path="/persons/:id" element={<PersonDetail />} />
                <Route path="/cameras" element={<Cameras />} />
                <Route path="/detections" element={<Detections events={events} />} />
                <Route path="/license-plates" element={<LicensePlates events={plateEvents} />} />
                <Route path="/workers" element={<Workers />} />
              </Routes>
            </Layout>
          }
        />
      </Routes>
    </ErrorBoundary>
  );
}
