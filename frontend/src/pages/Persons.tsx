import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api/client';
import type { Person } from '../api/client';
import PersonCard from '../components/PersonCard';
import Modal from '../components/Modal';
import LoadingSpinner from '../components/LoadingSpinner';

export default function Persons() {
  const navigate = useNavigate();
  const [persons, setPersons] = useState<Person[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);

  // Form state
  const [formName, setFormName] = useState('');
  const [formDept, setFormDept] = useState('');
  const [formNotes, setFormNotes] = useState('');
  const [formFiles, setFormFiles] = useState<File[]>([]);
  const [formPreviews, setFormPreviews] = useState<string[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  const limit = 20;

  useEffect(() => {
    loadPersons();
  }, [page, search]);

  async function loadPersons() {
    setLoading(true);
    try {
      const data = await api.listPersons({ search, page, limit });
      setPersons(data.items);
      setTotal(data.total);
    } catch (err) {
      console.error('Failed to load persons:', err);
    } finally {
      setLoading(false);
    }
  }

  function openModal() {
    setFormName('');
    setFormDept('');
    setFormNotes('');
    setFormFiles([]);
    setFormPreviews([]);
    setFormError('');
    setShowModal(true);
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files || []);
    setFormFiles((prev) => [...prev, ...files]);

    // Create previews
    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = (ev) => {
        setFormPreviews((prev) => [...prev, ev.target?.result as string]);
      };
      reader.readAsDataURL(file);
    });
  }

  function removeFile(index: number) {
    setFormFiles((prev) => prev.filter((_, i) => i !== index));
    setFormPreviews((prev) => prev.filter((_, i) => i !== index));
  }

  async function handleSubmit() {
    if (!formName.trim()) {
      setFormError('Name is required');
      return;
    }

    setSubmitting(true);
    setFormError('');

    try {
      // Create person
      const person = await api.createPerson({
        name: formName.trim(),
        department: formDept.trim(),
        notes: formNotes.trim(),
      });

      // Upload faces if any
      if (formFiles.length > 0) {
        try {
          await api.uploadFaces(person.id, formFiles);
        } catch (err) {
          console.error('Face upload warning:', err);
          // Person was created but faces failed — still continue
        }
      }

      setShowModal(false);
      loadPersons();
    } catch (err) {
      setFormError(err instanceof Error ? err.message : 'Failed to create person');
    } finally {
      setSubmitting(false);
    }
  }

  const totalPages = Math.ceil(total / limit);

  return (
    <div>
      <div className="page-header">
        <div>
          <h1 className="page-title">Face Management</h1>
          <p className="page-subtitle">Manage registered persons and their face data</p>
        </div>
        <button className="btn btn-primary" onClick={openModal}>
          ➕ Add Person
        </button>
      </div>

      {/* Search */}
      <div className="search-bar">
        <span className="search-bar-icon">🔍</span>
        <input
          type="text"
          className="form-input"
          placeholder="Search persons by name..."
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(1);
          }}
        />
      </div>

      {loading ? (
        <LoadingSpinner />
      ) : persons.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">👤</div>
            <div className="empty-state-title">
              {search ? 'No persons found' : 'No persons registered'}
            </div>
            <div className="empty-state-text">
              {search
                ? 'Try a different search query.'
                : 'Click "Add Person" to register someone for face recognition.'}
            </div>
          </div>
        </div>
      ) : (
        <>
          <div className="person-grid">
            {persons.map((person) => (
              <PersonCard
                key={person.id}
                person={person}
                onClick={() => navigate(`/persons/${person.id}`)}
              />
            ))}
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
                const pageNum = i + 1;
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
              {totalPages > 7 && <span style={{ color: 'var(--text-muted)' }}>...</span>}
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

      {/* Add Person Modal */}
      {showModal && (
        <Modal
          title="Add New Person"
          onClose={() => setShowModal(false)}
          footer={
            <>
              <button className="btn btn-ghost" onClick={() => setShowModal(false)}>
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={handleSubmit}
                disabled={submitting}
              >
                {submitting ? 'Saving...' : '✓ Save Person'}
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
            <label className="form-label">Name *</label>
            <input
              type="text"
              className="form-input"
              placeholder="Enter person's name"
              value={formName}
              onChange={(e) => setFormName(e.target.value)}
              autoFocus
            />
          </div>

          <div className="form-group">
            <label className="form-label">Department</label>
            <input
              type="text"
              className="form-input"
              placeholder="e.g. IT, HR, Security"
              value={formDept}
              onChange={(e) => setFormDept(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Notes</label>
            <textarea
              className="form-textarea"
              placeholder="Additional notes..."
              value={formNotes}
              onChange={(e) => setFormNotes(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Face Photos</label>
            <div
              className="file-upload"
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="file-upload-icon">📸</div>
              <div className="file-upload-text">
                Click to upload face photos
              </div>
              <div className="file-upload-hint">
                JPG, PNG — Photos with clear, front-facing view work best
              </div>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              hidden
              onChange={handleFileChange}
            />

            {formPreviews.length > 0 && (
              <div className="file-preview-grid">
                {formPreviews.map((src, i) => (
                  <div key={i} className="file-preview-item">
                    <img src={src} alt={`Preview ${i + 1}`} />
                    <button
                      className="file-preview-remove"
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile(i);
                      }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </Modal>
      )}
    </div>
  );
}
