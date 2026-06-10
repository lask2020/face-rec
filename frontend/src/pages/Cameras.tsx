import { useEffect, useState } from 'react';
import { api } from '../api/client';
import type { Camera, SSCameraInfo, SSConnectRequest } from '../api/client';
import CameraFeed from '../components/CameraFeed';
import Modal from '../components/Modal';
import LoadingSpinner from '../components/LoadingSpinner';

export default function Cameras() {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [editingCamera, setEditingCamera] = useState<Camera | null>(null);

  // Manual camera form
  const [formName, setFormName] = useState('');
  const [formUrl, setFormUrl] = useState('');
  const [formLocation, setFormLocation] = useState('');
  const [formFps, setFormFps] = useState(2);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState('');

  // Surveillance Station modal
  const [showSSModal, setShowSSModal] = useState(false);
  const [ssStep, setSSStep] = useState<'connect' | 'select'>('connect');
  const [ssUrl, setSSUrl] = useState('');
  const [ssUser, setSSUser] = useState('');
  const [ssPass, setSSPass] = useState('');
  const [ssVerifySSL, setSSVerifySSL] = useState(false);
  const [ssConnecting, setSSConnecting] = useState(false);
  const [ssError, setSSError] = useState('');
  const [ssCameras, setSSCameras] = useState<SSCameraInfo[]>([]);
  const [ssSelected, setSSSelected] = useState<Set<number>>(new Set());
  const [ssImporting, setSSImporting] = useState(false);
  const [ssSearchTerm, setSSSearchTerm] = useState('');

  useEffect(() => {
    loadCameras();
  }, []);

  async function loadCameras() {
    try {
      const data = await api.listCameras();
      setCameras(data.items);
    } catch (err) {
      console.error('Failed to load cameras:', err);
    } finally {
      setLoading(false);
    }
  }

  // ─── Manual Camera CRUD ────────────────────────────────────────────────

  function openAddModal() {
    setEditingCamera(null);
    setFormName('');
    setFormUrl('');
    setFormLocation('');
    setFormFps(2);
    setFormError('');
    setShowModal(true);
  }

  function openEditModal(camera: Camera) {
    setEditingCamera(camera);
    setFormName(camera.name);
    setFormUrl(camera.url);
    setFormLocation(camera.location);
    setFormFps(camera.fps_process);
    setFormError('');
    setShowModal(true);
  }

  async function handleSubmit() {
    if (!formName.trim() || !formUrl.trim()) {
      setFormError('Name and URL are required');
      return;
    }
    setSubmitting(true);
    setFormError('');
    try {
      if (editingCamera) {
        await api.updateCamera(editingCamera.id, {
          name: formName.trim(),
          url: formUrl.trim(),
          location: formLocation.trim(),
          fps_process: formFps,
        });
      } else {
        await api.createCamera({
          name: formName.trim(),
          url: formUrl.trim(),
          location: formLocation.trim(),
          fps_process: formFps,
        });
      }
      setShowModal(false);
      loadCameras();
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to save camera');
    } finally {
      setSubmitting(false);
    }
  }

  async function handleToggle(camera: Camera) {
    try {
      if (camera.is_active) {
        await api.stopCamera(camera.id);
      } else {
        await api.startCamera(camera.id);
      }
      loadCameras();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to toggle camera');
    }
  }

  async function handleDelete(camera: Camera) {
    if (!confirm(`Delete camera "${camera.name}"? This action cannot be undone.`)) return;
    try {
      await api.deleteCamera(camera.id);
      loadCameras();
    } catch (err) {
      alert('Failed to delete camera');
    }
  }

  // ─── Surveillance Station Integration ──────────────────────────────────

  function openSSModal() {
    setSSStep('connect');
    setSSError('');
    setSSCameras([]);
    setSSSelected(new Set());
    setSSSearchTerm('');
    setShowSSModal(true);
  }

  function getSSCredentials(): SSConnectRequest {
    return {
      base_url: ssUrl.trim(),
      username: ssUser.trim(),
      password: ssPass,
      verify_ssl: ssVerifySSL,
    };
  }

  async function handleSSConnect() {
    if (!ssUrl.trim() || !ssUser.trim() || !ssPass) {
      setSSError('All fields are required');
      return;
    }
    setSSConnecting(true);
    setSSError('');
    try {
      // Test connection first
      await api.ssTestConnection(getSSCredentials());

      // List cameras
      const result = await api.ssListCameras(getSSCredentials());
      setSSCameras(result.cameras);

      // Pre-select cameras that aren't imported yet
      const autoSelect = new Set<number>();
      result.cameras.forEach((cam) => {
        if (!cam.already_imported && cam.enabled) {
          autoSelect.add(cam.ss_id);
        }
      });
      setSSSelected(autoSelect);

      setSSStep('select');
    } catch (err) {
      setSSError(err instanceof Error ? err.message : 'Connection failed');
    } finally {
      setSSConnecting(false);
    }
  }

  function toggleSSCamera(ssId: number) {
    setSSSelected((prev) => {
      const next = new Set(prev);
      if (next.has(ssId)) {
        next.delete(ssId);
      } else {
        next.add(ssId);
      }
      return next;
    });
  }

  function selectAllSS() {
    const filteredCameras = ssCameras.filter(cam => 
      cam.name.toLowerCase().includes(ssSearchTerm.toLowerCase()) || 
      cam.host.toLowerCase().includes(ssSearchTerm.toLowerCase()) ||
      cam.vendor.toLowerCase().includes(ssSearchTerm.toLowerCase())
    );
    const allIds = filteredCameras.filter((c) => !c.already_imported).map((c) => c.ss_id);
    setSSSelected(new Set([...ssSelected, ...allIds]));
  }

  function deselectAllSS() {
    setSSSelected(new Set());
  }

  async function handleSSImport() {
    if (ssSelected.size === 0) return;
    setSSImporting(true);
    setSSError('');
    try {
      const result = await api.ssImportCameras({
        ...getSSCredentials(),
        camera_ids: Array.from(ssSelected),
      });

      const messages: string[] = [];
      if (result.imported > 0) messages.push(`✅ Imported ${result.imported} camera(s)`);
      if (result.skipped > 0) messages.push(`⏭️ Skipped ${result.skipped} (already imported)`);
      if (result.errors.length > 0) messages.push(`❌ Errors: ${result.errors.join(', ')}`);

      alert(messages.join('\n'));
      setShowSSModal(false);
      loadCameras();
    } catch (err) {
      setSSError(err instanceof Error ? err.message : 'Import failed');
    } finally {
      setSSImporting(false);
    }
  }

  // ─── Render ────────────────────────────────────────────────────────────

  if (loading) return <LoadingSpinner />;

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Camera Management</h1>
          <p className="page-subtitle">
            Configure and manage CCTV camera streams · {cameras.length} camera{cameras.length !== 1 ? 's' : ''}
          </p>
        </div>
        <div style={{ display: 'flex', gap: '10px' }}>
          <button className="btn btn-success" onClick={openSSModal}>
            🔗 Import from Surveillance Station
          </button>
          <button className="btn btn-primary" onClick={openAddModal}>
            ➕ Add Camera
          </button>
        </div>
      </div>

      {cameras.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">📹</div>
            <div className="empty-state-title">No cameras configured</div>
            <div className="empty-state-text">
              Add cameras manually or import from Synology Surveillance Station.
              <br />
              <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '8px', display: 'block' }}>
                Use "0" for built-in webcam, or enter an RTSP URL for IP cameras.
              </span>
            </div>
            <div style={{ marginTop: '20px', display: 'flex', gap: '10px' }}>
              <button className="btn btn-success" onClick={openSSModal}>
                🔗 Import from Surveillance Station
              </button>
              <button className="btn btn-primary" onClick={openAddModal}>
                ➕ Add Manually
              </button>
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
                  <div className="camera-location">{cam.location || cam.url}</div>
                </div>
                <div className="camera-actions">
                  <button
                    className={`btn btn-sm ${cam.is_active ? 'btn-danger' : 'btn-success'}`}
                    onClick={() => handleToggle(cam)}
                  >
                    {cam.is_active ? '⏹ Stop' : '▶ Start'}
                  </button>
                  <button className="btn btn-ghost btn-sm" onClick={() => openEditModal(cam)}>
                    ✏️
                  </button>
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={() => handleDelete(cam)}
                    style={{ color: 'var(--accent-red)' }}
                  >
                    🗑️
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ─── Add/Edit Camera Modal ──────────────────────────────────────── */}
      {showModal && (
        <Modal
          title={editingCamera ? 'Edit Camera' : 'Add New Camera'}
          onClose={() => setShowModal(false)}
          footer={
            <>
              <button className="btn btn-ghost" onClick={() => setShowModal(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={handleSubmit} disabled={submitting}>
                {submitting ? 'Saving...' : editingCamera ? '✓ Update' : '✓ Add Camera'}
              </button>
            </>
          }
        >
          {formError && (
            <div style={{
              padding: '10px 14px',
              background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--accent-red)',
              fontSize: '13px',
              marginBottom: '16px',
            }}>
              {formError}
            </div>
          )}
          <div className="form-group">
            <label className="form-label">Camera Name *</label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g. Front Door, Lobby, Parking"
              value={formName}
              onChange={(e) => setFormName(e.target.value)}
              autoFocus
            />
          </div>
          <div className="form-group">
            <label className="form-label">Stream URL *</label>
            <input
              type="text"
              className="form-input"
              placeholder="RTSP URL or device index (e.g. 0 for webcam)"
              value={formUrl}
              onChange={(e) => setFormUrl(e.target.value)}
            />
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
              Examples: <code style={{ color: 'var(--accent-cyan)' }}>0</code> (webcam),
              <code style={{ color: 'var(--accent-cyan)' }}> rtsp://admin:pass@192.168.1.100:554/stream</code>
            </div>
          </div>
          <div className="form-group">
            <label className="form-label">Location</label>
            <input
              type="text"
              className="form-input"
              placeholder="Physical location description"
              value={formLocation}
              onChange={(e) => setFormLocation(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Processing FPS</label>
            <input
              type="number"
              className="form-input"
              min={1}
              max={30}
              value={formFps}
              onChange={(e) => setFormFps(parseInt(e.target.value) || 2)}
            />
            <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
              Higher FPS = more CPU usage. Recommended: 1-5 for face detection.
            </div>
          </div>
        </Modal>
      )}

      {/* ─── Surveillance Station Import Modal ─────────────────────────── */}
      {showSSModal && (
        <Modal
          title={ssStep === 'connect' ? '🔗 Connect to Surveillance Station' : '📹 Select Cameras to Import'}
          onClose={() => setShowSSModal(false)}
          footer={
            ssStep === 'connect' ? (
              <>
                <button className="btn btn-ghost" onClick={() => setShowSSModal(false)}>Cancel</button>
                <button className="btn btn-primary" onClick={handleSSConnect} disabled={ssConnecting}>
                  {ssConnecting ? (
                    <><span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }} /> Connecting...</>
                  ) : '🔗 Connect'}
                </button>
              </>
            ) : (
              <>
                <button className="btn btn-ghost" onClick={() => setSSStep('connect')}>← Back</button>
                <button
                  className="btn btn-success"
                  onClick={handleSSImport}
                  disabled={ssImporting || ssSelected.size === 0}
                >
                  {ssImporting ? 'Importing...' : `📥 Import ${ssSelected.size} Camera${ssSelected.size !== 1 ? 's' : ''}`}
                </button>
              </>
            )
          }
        >
          {ssError && (
            <div style={{
              padding: '10px 14px',
              background: 'rgba(239,68,68,0.1)',
              border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--accent-red)',
              fontSize: '13px',
              marginBottom: '16px',
            }}>
              {ssError}
            </div>
          )}

          {ssStep === 'connect' ? (
            <>
              <div style={{
                padding: '12px 16px',
                background: 'rgba(59,130,246,0.08)',
                border: '1px solid rgba(59,130,246,0.2)',
                borderRadius: 'var(--radius-sm)',
                fontSize: '13px',
                color: 'var(--text-secondary)',
                marginBottom: '20px',
                lineHeight: 1.6,
              }}>
                💡 Connect to your Synology NAS to automatically import all cameras from Surveillance Station with their RTSP stream URLs.
              </div>

              <div className="form-group">
                <label className="form-label">NAS URL *</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="http://192.168.1.100:5000"
                  value={ssUrl}
                  onChange={(e) => setSSUrl(e.target.value)}
                  autoFocus
                />
                <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  Your Synology DSM URL (include port, usually 5000 for HTTP or 5001 for HTTPS)
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Username *</label>
                <input
                  type="text"
                  className="form-input"
                  placeholder="admin"
                  value={ssUser}
                  onChange={(e) => setSSUser(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Password *</label>
                <input
                  type="password"
                  className="form-input"
                  placeholder="Password"
                  value={ssPass}
                  onChange={(e) => setSSPass(e.target.value)}
                />
              </div>

              <div className="form-group" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input
                  type="checkbox"
                  id="ss-verify-ssl"
                  checked={ssVerifySSL}
                  onChange={(e) => setSSVerifySSL(e.target.checked)}
                  style={{ width: 16, height: 16 }}
                />
                <label htmlFor="ss-verify-ssl" style={{ fontSize: '13px', color: 'var(--text-secondary)', cursor: 'pointer' }}>
                  Verify SSL certificate (disable for self-signed certs)
                </label>
              </div>
            </>
          ) : (
            <>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: '16px',
                gap: '12px'
              }}>
                <input
                  type="text"
                  className="form-input"
                  style={{ flex: 1, padding: '8px 12px', fontSize: '13px' }}
                  placeholder="Search cameras by name, IP, or model..."
                  value={ssSearchTerm}
                  onChange={(e) => setSSSearchTerm(e.target.value)}
                  autoFocus
                />
                <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                  <span style={{ fontSize: '13px', color: 'var(--text-secondary)', marginRight: '8px' }}>
                    <strong style={{ color: 'var(--accent-blue)' }}>{ssSelected.size}</strong> selected
                  </span>
                  <button className="btn btn-ghost btn-sm" onClick={selectAllSS}>Select All</button>
                  <button className="btn btn-ghost btn-sm" onClick={deselectAllSS}>Deselect All</button>
                </div>
              </div>

              <div style={{ maxHeight: '400px', overflow: 'auto', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {ssCameras.filter(cam => 
                  cam.name.toLowerCase().includes(ssSearchTerm.toLowerCase()) ||
                  cam.host.toLowerCase().includes(ssSearchTerm.toLowerCase()) ||
                  cam.vendor.toLowerCase().includes(ssSearchTerm.toLowerCase()) ||
                  cam.model.toLowerCase().includes(ssSearchTerm.toLowerCase())
                ).map((cam) => (
                  <div
                    key={cam.ss_id}
                    onClick={() => !cam.already_imported && toggleSSCamera(cam.ss_id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '12px',
                      padding: '12px 16px',
                      background: cam.already_imported
                        ? 'rgba(100,100,100,0.1)'
                        : ssSelected.has(cam.ss_id)
                        ? 'rgba(59,130,246,0.1)'
                        : 'var(--bg-card)',
                      border: `1px solid ${
                        ssSelected.has(cam.ss_id) ? 'var(--border-accent)' : 'var(--border-subtle)'
                      }`,
                      borderRadius: 'var(--radius-md)',
                      cursor: cam.already_imported ? 'default' : 'pointer',
                      opacity: cam.already_imported ? 0.5 : 1,
                      transition: 'all 150ms ease',
                    }}
                  >
                    <input
                      type="checkbox"
                      checked={ssSelected.has(cam.ss_id)}
                      disabled={cam.already_imported}
                      onChange={() => toggleSSCamera(cam.ss_id)}
                      onClick={(e) => e.stopPropagation()}
                      style={{ width: 18, height: 18, flexShrink: 0 }}
                    />

                    <div style={{
                      width: 36,
                      height: 36,
                      borderRadius: 'var(--radius-sm)',
                      background: cam.enabled
                        ? 'rgba(16,185,129,0.15)'
                        : 'rgba(239,68,68,0.15)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: '16px',
                      flexShrink: 0,
                    }}>
                      {cam.enabled ? '🟢' : '🔴'}
                    </div>

                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-primary)' }}>
                        {cam.name}
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-muted)', display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                        <span>{cam.vendor} {cam.model}</span>
                        <span>•</span>
                        <span>{cam.host}</span>
                        <span>•</span>
                        <span>{cam.resolution}</span>
                      </div>
                    </div>

                    {cam.already_imported && (
                      <span style={{
                        padding: '3px 10px',
                        borderRadius: 'var(--radius-full)',
                        background: 'rgba(245,158,11,0.15)',
                        color: 'var(--accent-amber)',
                        fontSize: '11px',
                        fontWeight: 600,
                        whiteSpace: 'nowrap',
                      }}>
                        Already imported
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </Modal>
      )}
    </div>
  );
}
