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

interface EditState {
  display_name: string;
  role: string;
  max_cameras: number;
}

function WorkerCard({ worker, onRefresh }: { worker: WorkerInfo; onRefresh: () => void }) {
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form, setForm] = useState<EditState>({
    display_name: worker.display_name || '',
    role: worker.role || 'both',
    max_cameras: worker.max_cameras ?? 0,
  });

  // Sync form when worker data refreshes (and not editing)
  useEffect(() => {
    if (!editing) {
      setForm({
        display_name: worker.display_name || '',
        role: worker.role || 'both',
        max_cameras: worker.max_cameras ?? 0,
      });
    }
  }, [worker.display_name, worker.role, worker.max_cameras, editing]);

  async function handleTogglePause() {
    try {
      await api.toggleWorkerPause(worker.name);
      onRefresh();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to toggle pause');
    }
  }

  async function handleSave() {
    setSaving(true);
    try {
      await api.updateWorkerConfig(worker.name, {
        display_name: form.display_name,
        role: form.role,
        max_cameras: form.max_cameras,
      });
      setEditing(false);
      onRefresh();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to save config');
    } finally {
      setSaving(false);
    }
  }

  const borderColor = !worker.is_online
    ? 'var(--border-subtle)'
    : worker.is_paused
    ? 'var(--text-muted, #6b7280)'
    : 'var(--accent-blue)';

  return (
    <div
      className="card animate-in"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        borderLeft: `3px solid ${borderColor}`,
        opacity: worker.is_online ? 1 : 0.6,
      }}
    >
      {/* ── Header row ── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text-primary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {worker.display_name || worker.name}
          </div>
          {worker.display_name && worker.display_name !== worker.name && (
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
              {worker.name}
            </div>
          )}
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
          <div style={{ display: 'flex', gap: '4px' }}>
            {worker.is_online && (
              <button
                className="btn btn-sm"
                style={{
                  fontSize: '11px', padding: '3px 8px', height: 'auto',
                  background: worker.is_paused ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
                  color: worker.is_paused ? 'var(--accent-green, #10b981)' : 'var(--accent-red, #ef4444)',
                  border: worker.is_paused ? '1px solid rgba(16,185,129,0.3)' : '1px solid rgba(239,68,68,0.3)',
                }}
                onClick={handleTogglePause}
              >
                {worker.is_paused ? '▶ Resume' : '⏸ Pause'}
              </button>
            )}
            <button
              className="btn btn-sm"
              style={{
                fontSize: '11px', padding: '3px 8px', height: 'auto',
                background: editing ? 'rgba(255,255,255,0.08)' : 'transparent',
                color: 'var(--text-muted)',
                border: '1px solid var(--border-subtle)',
              }}
              onClick={() => setEditing((v) => !v)}
              title="Edit worker config"
            >
              ⚙
            </button>
          </div>
        </div>
      </div>

      {/* ── Inline config editor ── */}
      {editing && (
        <div style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)', padding: '14px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
          <div style={{ fontSize: '11px', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Worker Config
          </div>

          {/* Display Name */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Display Name</label>
            <input
              className="form-input"
              style={{ fontSize: '13px', padding: '6px 10px' }}
              placeholder={worker.name}
              value={form.display_name}
              onChange={(e) => setForm((f) => ({ ...f, display_name: e.target.value }))}
            />
          </div>

          {/* Role */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Role</label>
            <div style={{ display: 'flex', gap: '6px' }}>
              {(['inference', 'training', 'both'] as const).map((r) => (
                <button
                  key={r}
                  onClick={() => setForm((f) => ({ ...f, role: r }))}
                  style={{
                    flex: 1, padding: '6px 0', borderRadius: 'var(--radius-sm)', fontSize: '12px', fontWeight: 600, cursor: 'pointer',
                    border: form.role === r ? `1px solid ${ROLE_COLORS[r]}` : '1px solid var(--border-subtle)',
                    background: form.role === r ? `${ROLE_COLORS[r]}22` : 'transparent',
                    color: form.role === r ? ROLE_COLORS[r] : 'var(--text-muted)',
                    transition: 'all 0.15s',
                  }}
                >
                  {ROLE_LABELS[r]}
                </button>
              ))}
            </div>
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
              {form.role === 'inference' && 'รับ camera frames เท่านั้น — ไม่รับ finetune tasks'}
              {form.role === 'training' && 'รับ finetune tasks เท่านั้น — ไม่รับ camera frames'}
              {form.role === 'both' && 'รับทั้ง camera frames และ finetune tasks'}
            </div>
          </div>

          {/* Max Cameras */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <label style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Max Cameras <span style={{ opacity: 0.6 }}>(0 = ไม่จำกัด)</span></label>
            <input
              className="form-input"
              type="number"
              min={0}
              max={64}
              style={{ fontSize: '13px', padding: '6px 10px', width: '100px' }}
              value={form.max_cameras}
              onChange={(e) => setForm((f) => ({ ...f, max_cameras: Math.max(0, parseInt(e.target.value) || 0) }))}
            />
          </div>

          {/* Actions */}
          <div style={{ display: 'flex', gap: '8px', justifyContent: 'space-between', marginTop: '4px' }}>
            {/* Delete — only when offline */}
            {!worker.is_online && (
              <button
                className="btn btn-sm"
                style={{ fontSize: '12px', color: 'var(--accent-red, #ef4444)', border: '1px solid rgba(239,68,68,0.3)', background: 'rgba(239,68,68,0.07)' }}
                onClick={async () => {
                  if (!confirm(`ลบ worker "${worker.display_name || worker.name}" ออกจากระบบ?`)) return;
                  try {
                    await api.deleteWorker(worker.name);
                    onRefresh();
                  } catch (err) {
                    alert(err instanceof Error ? err.message : 'Failed to delete worker');
                  }
                }}
              >
                🗑 Delete
              </button>
            )}
            <div style={{ display: 'flex', gap: '8px', marginLeft: 'auto' }}>
              <button
                className="btn btn-sm"
                style={{ fontSize: '12px', color: 'var(--text-muted)', border: '1px solid var(--border-subtle)', background: 'transparent' }}
                onClick={() => setEditing(false)}
              >
                Cancel
              </button>
              <button
                className="btn btn-sm"
                style={{ fontSize: '12px', background: 'var(--accent-blue)', color: 'white', border: 'none', padding: '4px 16px', opacity: saving ? 0.6 : 1 }}
                onClick={handleSave}
                disabled={saving}
              >
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Live stats (online only) ── */}
      {worker.is_online && !editing && (
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

      {/* ── Assigned cameras ── */}
      {!editing && (
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
      )}
    </div>
  );
}

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
          {workers.map((worker) => (
            <WorkerCard key={worker.name} worker={worker} onRefresh={loadData} />
          ))}
        </div>
      )}
    </div>
  );
}
