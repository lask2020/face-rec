import { useEffect, useState, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { Person, Detection } from '../api/client';
import LoadingSpinner from '../components/LoadingSpinner';
import Modal from '../components/Modal';

export default function PersonDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [person, setPerson] = useState<Person | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [editDept, setEditDept] = useState('');
  const [editNotes, setEditNotes] = useState('');
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const personId = parseInt(id || '0');

  useEffect(() => {
    if (personId) loadData();
  }, [personId]);

  async function loadData() {
    setLoading(true);
    try {
      const [personData, detectionsData] = await Promise.all([
        api.getPerson(personId),
        api.listDetections({ person_id: personId, limit: 10 }),
      ]);
      setPerson(personData);
      setDetections(detectionsData.items);
      setEditName(personData.name);
      setEditDept(personData.department);
      setEditNotes(personData.notes);
    } catch (err) {
      console.error('Failed to load person:', err);
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    setUploading(true);
    try {
      await api.uploadFaces(personId, files);
      await loadData();
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function handleDeleteFace(faceId: number) {
    if (!confirm('Remove this face photo?')) return;
    try {
      await api.deleteFace(personId, faceId);
      await loadData();
    } catch (err) {
      alert('Failed to delete face');
    }
  }

  async function handleSaveEdit() {
    try {
      await api.updatePerson(personId, {
        name: editName,
        department: editDept,
        notes: editNotes,
      });
      setEditing(false);
      await loadData();
    } catch (err) {
      alert('Failed to update');
    }
  }

  async function handleDelete() {
    try {
      await api.deletePerson(personId);
      navigate('/persons');
    } catch (err) {
      alert('Failed to delete person');
    }
  }

  if (loading) return <LoadingSpinner />;
  if (!person) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">❌</div>
        <div className="empty-state-title">Person not found</div>
        <button className="btn btn-ghost" onClick={() => navigate('/persons')}>
          ← Back to Persons
        </button>
      </div>
    );
  }

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/persons')}>
            ← Back
          </button>
          <div>
            <h1 className="page-title">{person.name}</h1>
            <p className="page-subtitle">
              {person.department || 'No department'} · Registered {new Date(person.created_at).toLocaleDateString()}
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button className="btn btn-ghost btn-sm" onClick={() => setEditing(true)}>
            ✏️ Edit
          </button>
          <button className="btn btn-danger btn-sm" onClick={() => setShowDeleteConfirm(true)}>
            🗑️ Delete
          </button>
        </div>
      </div>

      <div className="two-col">
        {/* Face Photos */}
        <div>
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">🖼️ Face Photos ({person.faces.length})</h3>
              <button
                className="btn btn-primary btn-sm"
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
              >
                {uploading ? 'Uploading...' : '📸 Add Photo'}
              </button>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={handleUpload}
            />

            {person.faces.length === 0 ? (
              <div className="empty-state" style={{ padding: '40px 20px' }}>
                <div className="empty-state-icon">📸</div>
                <div className="empty-state-title">No face photos</div>
                <div className="empty-state-text">
                  Upload face photos so the system can recognize this person.
                </div>
              </div>
            ) : (
              <div className="face-gallery">
                {person.faces.map((face) => (
                  <div key={face.id} className="face-gallery-item">
                    <img src={face.image_url} alt="Face" />
                    <button
                      className="remove-btn"
                      onClick={() => handleDeleteFace(face.id)}
                      title="Remove this face"
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Person Info */}
          {person.notes && (
            <div className="card" style={{ marginTop: '20px' }}>
              <h3 className="card-title" style={{ marginBottom: '8px' }}>📝 Notes</h3>
              <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: 1.6 }}>
                {person.notes}
              </p>
            </div>
          )}
        </div>

        {/* Recent Detections */}
        <div>
          <div className="card">
            <h3 className="card-title" style={{ marginBottom: '16px' }}>🔍 Recent Detections</h3>
            {detections.length === 0 ? (
              <div className="empty-state" style={{ padding: '40px 20px' }}>
                <div className="empty-state-icon">📋</div>
                <div className="empty-state-title">No detections</div>
                <div className="empty-state-text">
                  This person hasn't been detected by any camera yet.
                </div>
              </div>
            ) : (
              <div className="detection-list">
                {detections.map((d) => (
                  <div key={d.id} className="detection-item">
                    <div className="detection-avatar known">
                      {person.name.charAt(0)}
                    </div>
                    <div className="detection-info">
                      <div className="detection-name">📹 {d.camera_name}</div>
                      <div className="detection-meta">
                        <span>{new Date(d.detected_at).toLocaleString()}</span>
                      </div>
                    </div>
                    <span className={`detection-confidence ${d.confidence >= 0.7 ? 'high' : d.confidence >= 0.4 ? 'medium' : 'low'}`}>
                      {(d.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Edit Modal */}
      {editing && (
        <Modal
          title="Edit Person"
          onClose={() => setEditing(false)}
          footer={
            <>
              <button className="btn btn-ghost" onClick={() => setEditing(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={handleSaveEdit}>✓ Save</button>
            </>
          }
        >
          <div className="form-group">
            <label className="form-label">Name</label>
            <input
              type="text"
              className="form-input"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Department</label>
            <input
              type="text"
              className="form-input"
              value={editDept}
              onChange={(e) => setEditDept(e.target.value)}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Notes</label>
            <textarea
              className="form-textarea"
              value={editNotes}
              onChange={(e) => setEditNotes(e.target.value)}
            />
          </div>
        </Modal>
      )}

      {/* Delete Confirm */}
      {showDeleteConfirm && (
        <Modal
          title="Delete Person"
          onClose={() => setShowDeleteConfirm(false)}
          footer={
            <>
              <button className="btn btn-ghost" onClick={() => setShowDeleteConfirm(false)}>Cancel</button>
              <button className="btn btn-danger" onClick={handleDelete}>🗑️ Delete</button>
            </>
          }
        >
          <p style={{ color: 'var(--text-secondary)', lineHeight: 1.6 }}>
            Are you sure you want to delete <strong>{person.name}</strong>?<br />
            This will remove all face data and detection history. This action cannot be undone.
          </p>
        </Modal>
      )}
    </div>
  );
}
