import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { WorkerInfo } from '../api/client';
import StatsCard from '../components/StatsCard';
import LoadingSpinner from '../components/LoadingSpinner';

export default function Workers() {
  const [workers, setWorkers] = useState<WorkerInfo[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
    // Poll active workers list every 5 seconds
    const timer = setInterval(loadData, 5000);
    return () => clearInterval(timer);
  }, []);

  async function loadData() {
    try {
      const data = await api.listWorkers();
      setWorkers(data.workers || []);
      setTotal(data.total || 0);
      setError(null);
    } catch (err) {
      console.error('Failed to load workers stats:', err);
      setError('Connection to Control Plane lost');
    } finally {
      setLoading(false);
    }
  }

  // Calculate unique assigned cameras count across all workers
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

      {/* Stats Summary */}
      <div className="stats-grid">
        <StatsCard
          icon="🤖"
          label="Active AI Workers"
          value={total}
          color="blue"
        />
        <StatsCard
          icon="📹"
          label="Assigned Cameras"
          value={totalAssignedCameras}
          color="green"
        />
        <StatsCard
          icon="⚡"
          label="Fleet Status"
          value={total > 0 ? 'Healthy' : 'No Workers'}
          color={total > 0 ? 'green' : 'amber'}
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
          {workers.map((worker) => (
            <div key={worker.id} className="card animate-in" style={{ display: 'flex', flexDirection: 'column', gap: '16px', borderLeft: '3px solid var(--accent-blue)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Worker ID</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '13px', fontWeight: 'bold', color: 'var(--text-primary)', marginTop: '2px', wordBreak: 'break-all' }}>
                    {worker.id}
                  </div>
                </div>
                <span className="camera-badge online" style={{ flexShrink: 0 }}>Online</span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', background: 'rgba(255,255,255,0.02)', padding: '10px 14px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Uptime</div>
                  <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--accent-blue)', marginTop: '2px' }}>{worker.uptime}</div>
                </div>
                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Connected At</div>
                  <div style={{ fontSize: '12px', fontWeight: '500', marginTop: '2px' }}>
                    {new Date(worker.connected_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </div>
                </div>
              </div>

              <div>
                <div style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)', marginBottom: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span>📹 Assigned Streams</span>
                  <span style={{ background: 'rgba(59,130,246,0.1)', color: 'var(--accent-blue)', fontSize: '11px', padding: '2px 8px', borderRadius: '10px' }}>
                    {worker.cameras?.length || 0} Camera(s)
                  </span>
                </div>

                {(!worker.cameras || worker.cameras.length === 0) ? (
                  <div style={{ fontSize: '13px', color: 'var(--text-muted)', fontStyle: 'italic', padding: '8px 12px', background: 'rgba(255,255,255,0.01)', borderRadius: 'var(--radius-sm)', border: '1px dashed var(--border-subtle)' }}>
                    Idle (Waiting for camera assignments)
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {worker.cameras.map((cam) => (
                      <div key={cam.id} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 12px', background: 'rgba(255,255,255,0.03)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', transition: 'background var(--transition-fast)' }}>
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
          ))}
        </div>
      )}
    </div>
  );
}
