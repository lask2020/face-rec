import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { Detection, Person, Camera, DetectionEvent } from '../api/client';
import LoadingSpinner from '../components/LoadingSpinner';
import Modal from '../components/Modal';

interface DetectionsProps {
  events?: DetectionEvent[];
}

export default function Detections({ events = [] }: DetectionsProps) {
  const [detections, setDetections] = useState<Detection[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [lastProcessedEventTime, setLastProcessedEventTime] = useState<string>('');

  // Filters
  const [personFilter, setPersonFilter] = useState<string>('');
  const [cameraFilter, setCameraFilter] = useState<string>('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');

  // Live WebSocket updates
  useEffect(() => {
    if (!events || events.length === 0) return;
    const latestEvent = events[0];

    // Avoid double processing the same event
    if (latestEvent.timestamp === lastProcessedEventTime) return;
    setLastProcessedEventTime(latestEvent.timestamp);

    // Apply active filters to the incoming live event
    if (personFilter) {
      if (personFilter === 'null') {
        if (latestEvent.person_id !== null) return;
      } else if (String(latestEvent.person_id) !== personFilter) {
        return;
      }
    }
    if (cameraFilter && String(latestEvent.camera_id) !== cameraFilter) {
      return;
    }

    // Increment overall total count
    setTotal((t) => t + 1);

    // Only inject in current view if we are on page 1
    if (page !== 1) return;

    // Build Detection object from event
    const newDetection: Detection = {
      id: Math.floor(Date.now() + Math.random() * 1000),
      person_id: latestEvent.person_id,
      person_name: latestEvent.person_name,
      camera_id: latestEvent.camera_id,
      camera_name: latestEvent.camera_name,
      confidence: latestEvent.confidence,
      snapshot_url: latestEvent.snapshot_url,
      face_crop_url: latestEvent.snapshot_url?.replace("cam_", "crop_cam_").replace(".jpg", "_0.jpg") || null,
      detected_at: latestEvent.timestamp,
    };

    setDetections((prev) => {
      const exists = prev.some(
        (d) =>
          d.camera_id === newDetection.camera_id &&
          new Date(d.detected_at).getTime() === new Date(newDetection.detected_at).getTime()
      );
      if (exists) return prev;
      return [newDetection, ...prev].slice(0, limit);
    });
  }, [events, page, personFilter, cameraFilter, lastProcessedEventTime]);


  // Data for filter dropdowns
  const [persons, setPersons] = useState<Person[]>([]);
  const [cameras, setCameras] = useState<Camera[]>([]);

  // Snapshot modal & Person details
  const [selectedDetection, setSelectedDetection] = useState<Detection | null>(null);
  const [selectedPerson, setSelectedPerson] = useState<Person | null>(null);
  const [loadingPerson, setLoadingPerson] = useState(false);

  useEffect(() => {
    if (selectedDetection && selectedDetection.person_id) {
      setLoadingPerson(true);
      api.getPerson(selectedDetection.person_id)
        .then((p) => setSelectedPerson(p))
        .catch((err) => {
          console.error('Failed to load person:', err);
          setSelectedPerson(null);
        })
        .finally(() => setLoadingPerson(false));
    } else {
      setSelectedPerson(null);
    }
  }, [selectedDetection]);

  const limit = 20;

  useEffect(() => {
    // Load filter data once
    api.listPersons({ limit: 100 }).then((d) => setPersons(d.items)).catch(() => {});
    api.listCameras().then((d) => setCameras(d.items)).catch(() => {});
  }, []);

  useEffect(() => {
    loadDetections();
  }, [page, personFilter, cameraFilter, dateFrom, dateTo]);

  async function loadDetections() {
    setLoading(true);
    try {
      const data = await api.listDetections({
        person_id: personFilter ? parseInt(personFilter) : undefined,
        camera_id: cameraFilter ? parseInt(cameraFilter) : undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        page,
        limit,
      });
      setDetections(data.items);
      setTotal(data.total);
    } catch (err) {
      console.error('Failed to load detections:', err);
    } finally {
      setLoading(false);
    }
  }

  function clearFilters() {
    setPersonFilter('');
    setCameraFilter('');
    setDateFrom('');
    setDateTo('');
    setPage(1);
  }

  const totalPages = Math.ceil(total / limit);
  const hasFilters = !!(personFilter || cameraFilter || dateFrom || dateTo);

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Detection Logs</h1>
          <p className="page-subtitle">
            Browse face detection history · {total} total records
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="filters-bar">
        <select
          className="form-select"
          value={personFilter}
          onChange={(e) => { setPersonFilter(e.target.value); setPage(1); }}
        >
          <option value="">All Persons</option>
          {persons.map((p) => (
            <option key={p.id} value={p.id}>{p.name}</option>
          ))}
        </select>

        <select
          className="form-select"
          value={cameraFilter}
          onChange={(e) => { setCameraFilter(e.target.value); setPage(1); }}
        >
          <option value="">All Cameras</option>
          {cameras.map((c) => (
            <option key={c.id} value={c.id}>{c.name}</option>
          ))}
        </select>

        <input
          type="date"
          className="form-input"
          value={dateFrom}
          onChange={(e) => { setDateFrom(e.target.value); setPage(1); }}
          style={{ width: 'auto' }}
          placeholder="From date"
        />

        <input
          type="date"
          className="form-input"
          value={dateTo}
          onChange={(e) => { setDateTo(e.target.value); setPage(1); }}
          style={{ width: 'auto' }}
          placeholder="To date"
        />

        {hasFilters && (
          <button className="btn btn-ghost btn-sm" onClick={clearFilters}>
            ✕ Clear
          </button>
        )}
      </div>

      {loading ? (
        <LoadingSpinner />
      ) : detections.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">📋</div>
            <div className="empty-state-title">No detections found</div>
            <div className="empty-state-text">
              {hasFilters
                ? 'Try adjusting your filters to see more results.'
                : 'Detection logs will appear here when cameras detect faces.'}
            </div>
          </div>
        </div>
      ) : (
        <>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Person</th>
                  <th>Camera</th>
                  <th>Time</th>
                  <th>Confidence</th>
                  <th>Snapshot</th>
                </tr>
              </thead>
              <tbody>
                {detections.map((d, i) => (
                  <tr key={d.id}>
                    <td style={{ color: 'var(--text-muted)' }}>
                      {(page - 1) * limit + i + 1}
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <div
                          className={`detection-avatar ${d.person_id ? 'known' : 'unknown'}`}
                          style={{ width: '32px', height: '32px', fontSize: '13px' }}
                        >
                          {d.person_name.charAt(0).toUpperCase()}
                        </div>
                        <span style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                          {d.person_name}
                        </span>
                      </div>
                    </td>
                    <td>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        📹 {d.camera_name}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: '13px' }}>
                        {new Date(d.detected_at).toLocaleString()}
                      </span>
                    </td>
                    <td>
                      <span
                        className={`detection-confidence ${
                          d.confidence >= 0.7 ? 'high' : d.confidence >= 0.4 ? 'medium' : 'low'
                        }`}
                      >
                        {(d.confidence * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td>
                      {d.snapshot_url ? (
                        <button
                          className="btn btn-ghost btn-sm"
                          onClick={() => setSelectedDetection(d)}
                        >
                          🖼️ View
                        </button>
                      ) : (
                        <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="pagination">
              <button
                className="pagination-btn"
                disabled={page <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                ← Prev
              </button>

              {Array.from({ length: Math.min(totalPages, 7) }, (_, i) => {
                let pageNum: number;
                if (totalPages <= 7) {
                  pageNum = i + 1;
                } else if (page <= 4) {
                  pageNum = i + 1;
                } else if (page >= totalPages - 3) {
                  pageNum = totalPages - 6 + i;
                } else {
                  pageNum = page - 3 + i;
                }
                return (
                  <button
                    key={pageNum}
                    className={`pagination-btn ${page === pageNum ? 'active' : ''}`}
                    onClick={() => setPage(pageNum)}
                  >
                    {pageNum}
                  </button>
                );
              })}

              <button
                className="pagination-btn"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}

      {/* Snapshot Modal */}
      {selectedDetection && (
        <Modal 
          title="Detection Details" 
          onClose={() => setSelectedDetection(null)}
          size="lg"
        >
          <div className="detection-view-layout">
            {/* Left Column: Snapshot image */}
            <div className="snapshot-side">
              <span className="person-side-label">Captured Frame</span>
              {selectedDetection.snapshot_url && (
                <div className="snapshot-modal">
                  <img src={selectedDetection.snapshot_url} alt="Detection snapshot" />
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
                  {selectedDetection.face_crop_url && (
                    <div>
                      <span className="person-side-label" style={{ fontSize: '10px', opacity: 0.8, marginBottom: '4px', display: 'block' }}>Detected</span>
                      <div className="person-side-face-item" style={{ width: '80px', height: '80px' }}>
                        <img src={selectedDetection.face_crop_url} alt="Detected face" />
                      </div>
                    </div>
                  )}

                  {/* Registered Face (if known) */}
                  {selectedDetection.person_id && (
                    <div>
                      <span className="person-side-label" style={{ fontSize: '10px', opacity: 0.8, marginBottom: '4px', display: 'block' }}>Registered</span>
                      <div className="person-side-face-item" style={{ width: '80px', height: '80px' }}>
                        {loadingPerson ? (
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)' }}>...</div>
                        ) : selectedPerson && selectedPerson.faces && selectedPerson.faces.length > 0 ? (
                          <img src={selectedPerson.faces[0].image_url} alt="Registered face" />
                        ) : (
                          <div className="person-side-avatar-placeholder" style={{ width: '100%', height: '100%', margin: 0, borderRadius: 'var(--radius-md)', fontSize: '20px' }}>?</div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {selectedDetection.person_id ? (
                <>
                  <div className="person-side-title">👤 Registered Person</div>
                  
                  {loadingPerson ? (
                    <LoadingSpinner />
                  ) : (
                    <>
                      {/* Person Details */}
                      <div className="person-side-info" style={{ marginTop: '8px' }}>
                        <div className="person-side-field">
                          <span className="person-side-label">Name</span>
                          <span className="person-side-value">{selectedPerson?.name}</span>
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
                  )}
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
                  <span className="person-side-value">📹 {selectedDetection.camera_name}</span>
                </div>
                <div className="person-side-field">
                  <span className="person-side-label">Detection Time</span>
                  <span className="person-side-value" style={{ fontFamily: 'var(--font-mono)', fontSize: '13px' }}>
                    {new Date(selectedDetection.detected_at).toLocaleString()}
                  </span>
                </div>
                <div className="person-side-field">
                  <span className="person-side-label">Confidence Score</span>
                  <span className={`detection-confidence ${
                    selectedDetection.confidence >= 0.7 ? 'high' : selectedDetection.confidence >= 0.4 ? 'medium' : 'low'
                  }`} style={{ alignSelf: 'flex-start' }}>
                    {(selectedDetection.confidence * 100).toFixed(0)}%
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
