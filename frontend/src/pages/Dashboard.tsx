import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { Camera, StatsOverview, DetectionEvent } from '../api/client';
import StatsCard from '../components/StatsCard';
import CameraFeed from '../components/CameraFeed';
import DetectionCard from '../components/DetectionCard';
import LoadingSpinner from '../components/LoadingSpinner';

interface DashboardProps {
  events: DetectionEvent[];
}

export default function Dashboard({ events }: DashboardProps) {
  const [stats, setStats] = useState<StatsOverview | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
    // Refresh stats every 30 seconds
    const timer = setInterval(loadData, 30000);
    return () => clearInterval(timer);
  }, []);

  async function loadData() {
    try {
      const [statsData, camerasData] = await Promise.all([
        api.getOverview(),
        api.listCameras(),
      ]);
      setStats(statsData);
      setCameras(camerasData.items);
    } catch (err) {
      console.error('Failed to load dashboard data:', err);
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <LoadingSpinner />;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-subtitle">Real-time face recognition monitoring</p>
        </div>
        <span className="live-indicator">LIVE</span>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <StatsCard
          icon="📹"
          label="Cameras Online"
          value={stats ? `${stats.active_cameras}/${stats.total_cameras}` : '0/0'}
          color="blue"
        />
        <StatsCard
          icon="👤"
          label="Persons Registered"
          value={stats?.total_persons ?? 0}
          color="green"
        />
        <StatsCard
          icon="🔍"
          label="Detections Today"
          value={stats?.total_detections_today ?? 0}
          color="amber"
        />
        <StatsCard
          icon="⚡"
          label="Live Events"
          value={events.length}
          color="purple"
        />
      </div>

      <div className="two-col">
        {/* Camera Grid */}
        <div>
          <h2 className="section-title">📹 Camera Feeds</h2>
          {cameras.length === 0 ? (
            <div className="card">
              <div className="empty-state">
                <div className="empty-state-icon">📷</div>
                <div className="empty-state-title">No cameras configured</div>
                <div className="empty-state-text">
                  Go to the Cameras page to add your first camera.
                </div>
              </div>
            </div>
          ) : (
            <div className="camera-grid">
              {cameras.map((cam) => (
                <div key={cam.id} className="camera-card animate-in">
                  <CameraFeed cameraId={cam.id} isActive={cam.is_active} />
                  <div className="camera-info">
                    <div>
                      <div className="camera-name">{cam.name}</div>
                      <div className="camera-location">{cam.location || 'No location set'}</div>
                    </div>
                    <span className={`camera-badge ${cam.is_active ? 'online' : 'offline'}`}>
                      {cam.is_active ? 'Online' : 'Offline'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Recent Detections */}
        <div>
          <h2 className="section-title">⚡ Recent Detections</h2>
          <div className="detection-list">
            {events.length === 0 ? (
              <div className="card">
                <div className="empty-state">
                  <div className="empty-state-icon">🔍</div>
                  <div className="empty-state-title">No detections yet</div>
                  <div className="empty-state-text">
                    Detection events will appear here in real-time when cameras are processing.
                  </div>
                </div>
              </div>
            ) : (
              events.slice(0, 20).map((event, i) => (
                <DetectionCard key={`${event.timestamp}-${i}`} event={event} />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
