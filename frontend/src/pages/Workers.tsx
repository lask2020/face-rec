import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { WorkerInfo } from '../api/client';
import StatsCard from '../components/StatsCard';
import LoadingSpinner from '../components/LoadingSpinner';

const ROLE_LABELS: Record<string, string> = {
  inference: 'Inference',
  training: 'Training',
  both: 'Both',
};

const ROLE_COLORS: Record<string, string> = {
  inference: 'var(--accent-blue)',
  training: 'var(--accent-purple, #8b5cf6)',
  both: 'var(--accent-green)',
};

export default function Workers() {
  const [workers, setWorkers] = useState<WorkerInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
    const timer = setInterval(loadData, 5000);
    return () => clearInterval(timer);
  }, []);

  async function loadData() {
    try {
      const data = await api.listWorkers();
      setWorkers(data.workers || []);
      setError(null);
    } catch (err) {
      console.error('Failed to load workers:', err);
      setError('Connection to Control Plane lost');
    } finally {
      setLoading(false);
    }
  }

  async function handleTogglePause(name: string) {
    try {
      await api.toggleWorkerPause(name);
      loadData();
    } catch (err) {
      console.error('Failed to toggle worker pause:', err);
      alert(err instanceof Error ? err.message : 'Failed to toggle worker pause');
    }
  }

  const onlineWorkers = workers.filter((w) => w.is_online);
  const totalAssignedCameras = workers.reduce((sum, w) => sum + (w.cameras?.length || 0), 0);

  if (loading) return <LoadingSpinner />;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">AI Inference Workers</h1>
          <p className="page-subtitle">Stateless Python AI worker nodes running InsightFace</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {error && <span className="camera-badge offline">{error}</span>}
          {!error && (
            <span className="live-indicator" style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span className="status-dot" style={{ margin: 0, width: '6px', height: '6px' }} />
              LIVE
            </span>
          )}
        </div>
      </div>

      {/* Stats */}
      <div className="stats-grid">
        <StatsCard icon="🤖" label="Active AI Workers" value={onlineWorkers.length} color="blue" />
        <StatsCard icon="📹" label="Assigned Cameras" value={totalAssignedCameras} color="green" />
        <StatsCard
          icon="⚡"
          label="Fleet Status"
          value={onlineWorkers.length > 0 ? 'Healthy' : 'No Workers'}
          color={onlineWorkers.length > 0 ? 'green' : 'amber'}
        />
      </div>

      <h2 className="section-title" style={{ marginBottom: '16px' }}>🤖 Connected Worker Fleet</h2>

      {workers.length === 0 ? (
        <div className="card animate-in">
          <div className="empty-state">
            <div className="empty-state-icon">⚠️</div>
            <div className="empty-state-title">No AI Workers Connected</div>
            <div className="empty-state-text" style={{ maxWidth: '500px', margin: '8px auto' }}>
              The system requires at least one Python AI worker to perform face detection and recognition.
            </div>
            <div style={{ marginTop: '20px', fontFamily: 'monospace', background: 'rgba(0,0,0,0.3)', padding: '12px 20px', borderRadius: '6px', display: 'inline-block', fontSize: '13px', color: '#8b5cf6' }}>
              docker compose up -d --scale ai-worker=2
            </div>
          </div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))', gap: '20px' }}>
          {workers.map((worker) => {
            const borderColor = !worker.is_online
              ? 'var(--border-subtle)'
              : worker.is_paused
              ? 'var(--text-muted, #6b7280)'
              : 'var(--accent-blue)';

            return (
              <div
                key={worker.name}
                className="card animate-in"
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '16px',
                  borderLeft: `3px solid ${borderColor}`,
                  opacity: worker.is_online ? 1 : 0.6,
                }}
              >
                {/* Header row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ minWidth: 0 }}>
                    {/* Worker name */}
                    <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {worker.display_name || worker.name}
                    </div>
                    {/* Persistent name (if display_name differs) */}
                    {worker.display_name && worker.display_name !== worker.name && (
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
                        {worker.name}
                      </div>
                    )}
                    {/* Role + max cameras badges */}
                    <div style={{ display: 'flex', gap: '6px', marginTop: '6px', flexWrap: 'wrap' }}>
                      <span style={{
                        fontSize: '10px', padding: '1px 7px', borderRadius: '10px', fontWeight: 600,
                        background: `${ROLE_COLORS[worker.role] ?? 'var(--accent-blue)'}22`,
                        color: ROLE_COLORS[worker.role] ?? 'var(--accent-blue)',
                        border: `1px solid ${ROLE_COLORS[worker.role] ?? 'var(--accent-blue)'}44`,
                      }}>
                        {ROLE_LABELS[worker.role] ?? worker.role}
                      </span>
                      {worker.max_cameras > 0 && (
                        <span style={{
                          fontSize: '10px', padding: '1px 7px', borderRadius: '10px', fontWeight: 600,
                          background: 'rgba(255,255,255,0.05)',
                          color: 'var(--text-secondary)',
                          border: '1px solid var(--border-subtle)',
                        }}>
                          max {worker.max_cameras} cam{worker.max_cameras > 1 ? 's' : ''}
                        </span>
                      )}
                    </div>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '6px', flexShrink: 0, marginLeft: '8px' }}>
                    <span className={!worker.is_online ? 'camera-badge offline' : worker.is_paused ? 'camera-badge offline' : 'camera-badge online'}>
                      {!worker.is_online ? 'Offline' : worker.is_paused ? 'Paused' : 'Online'}
                    </span>
                    {worker.is_online && (
                      <button
                        className="btn btn-sm"
                        style={{
                          fontSize: '11px',
                          padding: '3px 8px',
                          height: 'auto',
                          background: worker.is_paused ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
                          color: worker.is_paused ? 'var(--accent-green, #10b981)' : 'var(--accent-red, #ef4444)',
                          border: worker.is_paused ? '1px solid rgba(16,185,129,0.3)' : '1px solid rgba(239,68,68,0.3)',
                        }}
                        onClick={() => handleTogglePause(worker.name)}
                      >
                        {worker.is_paused ? '▶ Resume' : '⏸ Pause'}
                      </button>
                    )}
                  </div>
                </div>

                {/* Stats row — only when online */}
                {worker.is_online && (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '12px', background: 'rgba(255,255,255,0.02)', padding: '10px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                    <div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Uptime</div>
                      <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--accent-blue)', marginTop: '2px' }}>{worker.uptime}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Avg Latency</div>
                      <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--accent-green)', marginTop: '2px' }}>
                        {worker.avg_process_ms > 0 ? `${worker.avg_process_ms.toFixed(0)} ms` : '0 ms'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Connected At</div>
                      <div style={{ fontSize: '12px', fontWeight: '500', marginTop: '2px' }}>
                        {worker.connected_at
                          ? new Date(worker.connected_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                          : '—'}
                      </div>
                    </div>
                  </div>
                )}

                {/* Camera list */}
                <div>
                  <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)', marginBottom: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>📹 Assigned Streams</span>
                    <span style={{ background: 'rgba(59,130,246,0.1)', color: 'var(--accent-blue)', fontSize: '11px', padding: '2px 8px', borderRadius: '10px' }}>
                      {worker.cameras?.length || 0} Camera(s)
                    </span>
                  </div>

                  {(!worker.cameras || worker.cameras.length === 0) ? (
                    <div style={{ fontSize: '13px', color: 'var(--text-muted)', fontStyle: 'italic', padding: '8px 12px', background: 'rgba(255,255,255,0.01)', borderRadius: 'var(--radius-sm)', border: '1px dashed var(--border-subtle)' }}>
                      {worker.is_online ? 'Idle (Waiting for camera assignments)' : 'Offline'}
                    </div>
                  ) : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                      {worker.cameras.map((cam) => (
                        <div key={cam.id} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 12px', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                          <span style={{ fontSize: '14px' }}>📹</span>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {cam.name}
                            </div>
                            <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>ID: {cam.id}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
