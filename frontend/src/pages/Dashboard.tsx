import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import type { Camera, StatsOverview, DetectionEvent, Person, Detection } from '../api/client';
import StatsCard from '../components/StatsCard';
import CameraFeed from '../components/CameraFeed';
import ErrorBoundary from '../components/ErrorBoundary';
import DetectionCard from '../components/DetectionCard';
import LoadingSpinner from '../components/LoadingSpinner';
import Modal from '../components/Modal';

interface DashboardProps {
  events: DetectionEvent[];
}

function detectionToEvent(d: Detection): DetectionEvent {
  return {
    type: 'detection',
    person_id: d.person_id,
    person_name: d.person_name,
    camera_id: d.camera_id,
    camera_name: d.camera_name,
    confidence: d.confidence,
    snapshot_url: d.snapshot_url,
    face_crop_url: d.face_crop_url,
    restored_face_url: d.restored_face_url,
    timestamp: d.detected_at,
  };
}

export default function Dashboard({ events }: DashboardProps) {
  const [stats, setStats] = useState<StatsOverview | null>(null);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [recentDetections, setRecentDetections] = useState<DetectionEvent[]>([]);
  const [selectedEvent, setSelectedEvent] = useState<DetectionEvent | null>(null);
  const [selectedPerson, setSelectedPerson] = useState<Person | null>(null);
  const [loadingPerson, setLoadingPerson] = useState(false);
  const lastEventTimeRef = useRef('');

  useEffect(() => {
    loadData();
    // Refresh stats every 30 seconds
    const timer = setInterval(loadData, 30000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (selectedEvent && selectedEvent.person_id) {
      setLoadingPerson(true);
      api.getPerson(selectedEvent.person_id)
        .then((p) => setSelectedPerson(p))
        .catch((err) => {
          console.error('Failed to load person details:', err);
          setSelectedPerson(null);
        })
        .finally(() => setLoadingPerson(false));
    } else {
      setSelectedPerson(null);
    }
  }, [selectedEvent]);

  // Load initial detections from DB so the list is populated on first render,
  // not just after the first WS event arrives this session.
  useEffect(() => {
    api.listDetections({ limit: 20 })
      .then(data => setRecentDetections(data.items.map(detectionToEvent)))
      .catch(() => {});
  }, []);

  // Prepend live WS events to the local list.
  useEffect(() => {
    if (events.length === 0) return;
    const latest = events[0];
    if (latest.timestamp === lastEventTimeRef.current) return;
    lastEventTimeRef.current = latest.timestamp;
    setRecentDetections(prev => {
      const exists = prev.some(
        e => e.timestamp === latest.timestamp && e.camera_id === latest.camera_id
      );
      if (exists) return prev;
      return [latest, ...prev].slice(0, 20);
    });
  }, [events]);

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
                  <ErrorBoundary name={`Camera Feed (${cam.name})`}>
                    <CameraFeed cameraId={cam.id} isActive={cam.is_active} />
                  </ErrorBoundary>
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
            {recentDetections.length === 0 ? (
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
              recentDetections.map((event, i) => (
                <DetectionCard
                  key={`${event.timestamp}-${event.camera_id}-${i}`}
                  event={event}
                  onClick={() => setSelectedEvent(event)}
                />
              ))
            )}
          </div>
        </div>
      </div>

      {/* Snapshot Modal */}
      {selectedEvent && (
        <Modal 
          title="Detection Details" 
          onClose={() => setSelectedEvent(null)}
          size="lg"
        >
          <div className="detection-view-layout">
            {/* Left Column: Snapshot image */}
            <div className="snapshot-side">
              <span className="person-side-label">Captured Frame</span>
              {selectedEvent.snapshot_url && (
                <div className="snapshot-modal">
                  <img src={selectedEvent.snapshot_url} alt="Detection snapshot" />
                </div>
              )}
            </div>

            {/* Right Column: Person Info & Registered Face */}
            <div className="person-side">
              {/* Face Verification Section */}
              <div className="person-side-field">
                <span className="person-side-label">Face Verification</span>
                <div style={{ display: 'flex', gap: '16px', marginTop: '8px', marginBottom: '16px' }}>
                  {/* Detected Face */}
                  {selectedEvent.snapshot_url && (
                    <div>
                      <span className="person-side-label" style={{ fontSize: '10px', opacity: 0.8, marginBottom: '4px', display: 'block' }}>Detected</span>
                      <div className="person-side-face-item" style={{ width: '80px', height: '80px' }}>
                        <img 
                          src={selectedEvent.face_crop_url || selectedEvent.snapshot_url.replace("cam_", "crop_cam_").replace(".jpg", "_0.jpg")} 
                          onError={(e) => {
                            (e.target as HTMLImageElement).src = selectedEvent.snapshot_url || '';
                          }}
                          alt="Detected face" 
                          style={{ cursor: 'pointer' }} 
                          onClick={(e) => window.open((e.target as HTMLImageElement).src, '_blank')}
                        />
                      </div>
                    </div>
                  )}

                  {/* Restored Face */}
                  {selectedEvent.restored_face_url && (
                    <div>
                      <span className="person-side-label" style={{ fontSize: '10px', opacity: 0.8, marginBottom: '4px', display: 'block' }}>Restored ✨</span>
                      <div className="person-side-face-item" style={{ width: '80px', height: '80px', border: '1px solid var(--primary-light)' }}>
                        <img src={selectedEvent.restored_face_url} alt="Restored face" style={{ cursor: 'pointer' }} onClick={(e) => window.open((e.target as HTMLImageElement).src, '_blank')} />
                      </div>
                    </div>
                  )}

                  {/* Registered Face */}
                  {selectedEvent.person_id && (
                    <div>
                      <span className="person-side-label" style={{ fontSize: '10px', opacity: 0.8, marginBottom: '4px', display: 'block' }}>Registered</span>
                      <div className="person-side-face-item" style={{ width: '80px', height: '80px' }}>
                        {loadingPerson ? (
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>...</div>
                        ) : selectedPerson && selectedPerson.faces && selectedPerson.faces.length > 0 ? (
                          <img src={selectedPerson.faces[0].image_url} alt="Registered face" style={{ cursor: 'pointer' }} onClick={(e) => window.open((e.target as HTMLImageElement).src, '_blank')} />
                        ) : (
                          <div className="person-side-avatar-placeholder" style={{ width: '100%', height: '100%', margin: 0, borderRadius: 'var(--radius-md)', fontSize: '20px' }}>?</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {selectedEvent.person_id ? (
                <>
                  <div className="person-side-title">👤 Registered Person</div>
                  <div className="person-side-info" style={{ marginTop: '8px' }}>
                    <div className="person-side-field">
                      <span className="person-side-label">Name</span>
                      <span className="person-side-value">{selectedPerson?.name || selectedEvent.person_name}</span>
                    </div>
                    <div className="person-side-field">
                      <span className="person-side-label">Department</span>
                      <span className="person-side-value">{selectedPerson?.department || 'No department'}</span>
                    </div>
                    {selectedPerson?.notes && (
                      <div className="person-side-field">
                        <span className="person-side-label">Notes</span>
                        <span className="person-side-value" style={{ fontSize: '13px', lineHeight: 1.4 }}>
                          {selectedPerson.notes}
                        </span>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <>
                  <div className="person-side-title">❓ Unknown Person</div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '10px' }}>
                    <span style={{ color: 'var(--text-muted)', fontSize: '13px', textAlign: 'left' }}>
                      This face was not recognized as any registered person in the database.
                    </span>
                  </div>
                </>
              )}

              {/* General Metadata */}
              <div className="person-side-info" style={{ marginTop: 'auto', paddingTop: '16px', borderTop: '1px solid var(--border-subtle)' }}>
                <div className="person-side-field">
                  <span className="person-side-label">Camera</span>
                  <span className="person-side-value">📹 {selectedEvent.camera_name}</span>
                </div>
                <div className="person-side-field">
                  <span className="person-side-label">Detection Time</span>
                  <span className="person-side-value" style={{ fontFamily: 'var(--font-mono)', fontSize: '13px' }}>
                    {new Date(selectedEvent.timestamp).toLocaleString()}
                  </span>
                </div>
                <div className="person-side-field">
                  <span className="person-side-label">Confidence Score</span>
                  <span className={`detection-confidence ${
                    selectedEvent.confidence >= 0.7 ? 'high' : selectedEvent.confidence >= 0.4 ? 'medium' : 'low'
                  }`} style={{ alignSelf: 'flex-start' }}>
                    {(selectedEvent.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}
