import { useEffect, useState, useCallback } from 'react';
import { api } from '../api/client';
import type { PlateDetection, PlateDetectionEvent, Camera } from '../api/client';
import LoadingSpinner from '../components/LoadingSpinner';
import Modal from '../components/Modal';

interface LicensePlatesProps {
  events?: PlateDetectionEvent[];
}

export default function LicensePlates({ events = [] }: LicensePlatesProps) {
  const [plates, setPlates] = useState<PlateDetection[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [cameraFilter, setCameraFilter] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [lastEventTime, setLastEventTime] = useState('');
  const [selected, setSelected] = useState<PlateDetection | null>(null);
  const [clearing, setClearing] = useState(false);

  const LIMIT = 24;

  const loadPlates = useCallback(async () => {
    try {
      const res = await api.listPlateDetections({
        camera_id: cameraFilter ? Number(cameraFilter) : undefined,
        date_from: dateFrom || undefined,
        date_to: dateTo || undefined,
        page,
        limit: LIMIT,
      });
      setPlates(res.items);
      setTotal(res.total);
    } catch (err) {
      console.error('Failed to load plate detections', err);
    } finally {
      setLoading(false);
    }
  }, [cameraFilter, dateFrom, dateTo, page]);

  useEffect(() => {
    api.listCameras().then((r) => setCameras(r.items));
  }, []);

  useEffect(() => {
    setLoading(true);
    loadPlates();
  }, [loadPlates]);

  // Live WebSocket — prepend new plate events
  useEffect(() => {
    if (!events || events.length === 0) return;
    const latest = events[0];
    if (latest.timestamp === lastEventTime) return;
    setLastEventTime(latest.timestamp);
    if (page !== 1) return;

    const newEntry: PlateDetection = {
      id: Date.now(),
      camera_id: latest.camera_id,
      camera_name: latest.camera_name,
      plate_number: latest.plate_number,
      raw_text: latest.raw_text,
      confidence: latest.confidence,
      plate_type: latest.plate_type,
      province: latest.province,
      snapshot_url: latest.snapshot_url,
      detected_at: latest.timestamp,
    };
    setPlates((prev) => [newEntry, ...prev.slice(0, LIMIT - 1)]);
    setTotal((t) => t + 1);
  }, [events, lastEventTime, page]);

  function clearFilters() {
    setCameraFilter('');
    setDateFrom('');
    setDateTo('');
    setPage(1);
  }

  async function handleClearAll() {
    if (!window.confirm(`ลบรายการทั้งหมด ${total.toLocaleString()} รายการ และรูปภาพทั้งหมด?\n\nการดำเนินการนี้ไม่สามารถย้อนกลับได้`)) return;
    setClearing(true);
    try {
      await api.clearPlateDetections();
      setPlates([]);
      setTotal(0);
      setPage(1);
    } catch (err) {
      console.error('Failed to clear plate detections', err);
      alert('เกิดข้อผิดพลาด ไม่สามารถลบข้อมูลได้');
    } finally {
      setClearing(false);
    }
  }

  const totalPages = Math.ceil(total / LIMIT);
  const hasFilters = cameraFilter || dateFrom || dateTo;

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1 className="page-title">ทะเบียนรถ</h1>
          <p className="page-subtitle">{total.toLocaleString()} รายการทั้งหมด</p>
        </div>
        {total > 0 && (
          <button
            className="btn btn-danger"
            onClick={handleClearAll}
            disabled={clearing}
            style={{ marginLeft: 'auto' }}
          >
            {clearing ? 'กำลังลบ...' : 'ล้างทั้งหมด'}
          </button>
        )}
      </div>

      {/* Filters */}
      <div className="filters-bar">
        <select
          className="filter-select"
          value={cameraFilter}
          onChange={(e) => { setCameraFilter(e.target.value); setPage(1); }}
        >
          <option value="">กล้องทั้งหมด</option>
          {cameras.map((cam) => (
            <option key={cam.id} value={cam.id}>{cam.name}</option>
          ))}
        </select>
        <input
          type="date"
          className="filter-input"
          value={dateFrom}
          onChange={(e) => { setDateFrom(e.target.value); setPage(1); }}
        />
        <input
          type="date"
          className="filter-input"
          value={dateTo}
          onChange={(e) => { setDateTo(e.target.value); setPage(1); }}
        />
        {hasFilters && (
          <button className="btn btn-ghost" onClick={clearFilters}>ล้างตัวกรอง</button>
        )}
      </div>

      {/* Content */}
      {loading ? (
        <LoadingSpinner />
      ) : plates.length === 0 ? (
        <div style={{
          textAlign: 'center', padding: '4rem 2rem',
          color: 'var(--text-muted)', fontSize: '1rem',
        }}>
          <div style={{ fontSize: '3rem', marginBottom: '1rem' }}>🚗</div>
          ไม่พบข้อมูลทะเบียนรถ
        </div>
      ) : (
        <>
          {/* Card Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))',
            gap: '1rem',
            marginTop: '1rem',
          }}>
            {plates.map((plate) => (
              <PlateCard key={plate.id} plate={plate} onClick={() => setSelected(plate)} />
            ))}
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="pagination">
              <button className="btn btn-ghost" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                ← ก่อนหน้า
              </button>
              <span className="page-info">หน้า {page} / {totalPages}</span>
              <button className="btn btn-ghost" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
                ถัดไป →
              </button>
            </div>
          )}
        </>
      )}

      {/* Detail Modal */}
      {selected && (
        <PlateModal plate={selected} onClose={() => setSelected(null)} />
      )}
    </div>
  );
}

// ── Card component ────────────────────────────────────────────────────────────

function PlateCard({ plate, onClick }: { plate: PlateDetection; onClick: () => void }) {
  const [imgError, setImgError] = useState(false);
  const confidence = Math.round(plate.confidence * 100);
  const detectedAt = new Date(plate.detected_at);

  const confidenceColor =
    confidence >= 80 ? 'var(--accent-emerald)' :
    confidence >= 50 ? '#f59e0b' :
    'var(--accent-red)';

  return (
    <div
      onClick={onClick}
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border-color)',
        borderRadius: 'var(--radius-lg)',
        overflow: 'hidden',
        cursor: 'pointer',
        transition: 'transform 0.15s, box-shadow 0.15s',
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLDivElement).style.transform = 'translateY(-2px)';
        (e.currentTarget as HTMLDivElement).style.boxShadow = '0 8px 24px rgba(0,0,0,0.2)';
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLDivElement).style.transform = '';
        (e.currentTarget as HTMLDivElement).style.boxShadow = '';
      }}
    >
      {/* Snapshot */}
      <div style={{ position: 'relative', width: '100%', aspectRatio: '16/9', background: 'var(--bg-tertiary)' }}>
        {plate.snapshot_url && !imgError ? (
          <img
            src={plate.snapshot_url}
            alt="vehicle snapshot"
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
            onError={() => setImgError(true)}
          />
        ) : (
          <div style={{
            width: '100%', height: '100%',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: '3rem', color: 'var(--text-muted)',
          }}>
            🚗
          </div>
        )}

        {/* Plate number badge overlay */}
        <div style={{
          position: 'absolute', bottom: 8, left: 8, right: 8,
          background: 'rgba(0,0,0,0.75)',
          backdropFilter: 'blur(6px)',
          borderRadius: 6,
          padding: '4px 10px',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>
          <span style={{
            fontFamily: 'monospace',
            fontWeight: 700,
            fontSize: '1rem',
            letterSpacing: 2,
            color: '#fff',
          }}>
            {plate.plate_number || plate.raw_text || '—'}
          </span>
          <span style={{ color: confidenceColor, fontWeight: 600, fontSize: '0.8rem' }}>
            {confidence}%
          </span>
        </div>
      </div>

      {/* Meta */}
      <div style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 4 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            {plate.province || 'ไม่ทราบจังหวัด'}
          </span>
          <span style={{
            fontSize: '0.72rem', padding: '2px 7px', borderRadius: 999,
            background: plate.plate_type === 'commercial' ? 'rgba(245,158,11,0.15)' : 'rgba(99,102,241,0.15)',
            color: plate.plate_type === 'commercial' ? '#f59e0b' : '#818cf8',
          }}>
            {plate.plate_type === 'commercial' ? 'พาณิชย์' : plate.plate_type === 'standard' ? 'มาตรฐาน' : plate.plate_type || 'ไม่ทราบ'}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            📹 {plate.camera_name}
          </span>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            {detectedAt.toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Detail Modal ──────────────────────────────────────────────────────────────

function PlateModal({ plate, onClose }: { plate: PlateDetection; onClose: () => void }) {
  const [imgError, setImgError] = useState(false);
  const confidence = Math.round(plate.confidence * 100);
  const detectedAt = new Date(plate.detected_at);

  const confidenceColor =
    confidence >= 80 ? 'var(--accent-emerald)' :
    confidence >= 50 ? '#f59e0b' :
    'var(--accent-red)';

  return (
    <Modal title="รายละเอียดทะเบียนรถ" onClose={onClose} size="lg">
      <div className="detection-view-layout">
        {/* Left — snapshot */}
        <div className="snapshot-side">
          <span className="person-side-label">ภาพจากกล้อง</span>
          <div className="snapshot-modal" style={{ marginTop: 8 }}>
            {plate.snapshot_url && !imgError ? (
              <img
                src={plate.snapshot_url}
                alt="vehicle snapshot"
                style={{ cursor: 'zoom-in' }}
                onClick={() => window.open(plate.snapshot_url!, '_blank')}
                onError={() => setImgError(true)}
              />
            ) : (
              <div style={{
                width: '100%', aspectRatio: '16/9',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)',
                fontSize: '4rem', color: 'var(--text-muted)',
              }}>
                🚗
              </div>
            )}
          </div>

          {/* Plate number display (physical plate style) */}
          <div style={{
            marginTop: 16,
            background: '#fff',
            border: '3px solid #1a237e',
            borderRadius: 8,
            padding: '8px 16px',
            textAlign: 'center',
            boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
          }}>
            <div style={{
              fontSize: '0.6rem',
              color: '#1a237e',
              fontWeight: 700,
              letterSpacing: 3,
              textTransform: 'uppercase',
              marginBottom: 2,
            }}>
              {plate.province || 'ประเทศไทย'}
            </div>
            <div style={{
              fontFamily: 'monospace',
              fontSize: '1.8rem',
              fontWeight: 900,
              color: '#111',
              letterSpacing: 4,
              lineHeight: 1,
            }}>
              {plate.plate_number || plate.raw_text || '— — —'}
            </div>
          </div>
        </div>

        {/* Right — details */}
        <div className="person-side">
          <div className="person-side-title">🚗 ข้อมูลทะเบียน</div>

          <div className="person-side-info" style={{ marginTop: 12 }}>
            <div className="person-side-field">
              <span className="person-side-label">ทะเบียน</span>
              <span className="person-side-value" style={{ fontFamily: 'monospace', fontSize: '1.1rem', fontWeight: 700, letterSpacing: 2 }}>
                {plate.plate_number || <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>ไม่ผ่านการ validate</span>}
              </span>
            </div>

            {plate.raw_text && plate.raw_text !== plate.plate_number && (
              <div className="person-side-field">
                <span className="person-side-label">ข้อความ OCR ดิบ</span>
                <span className="person-side-value" style={{ fontFamily: 'monospace', color: 'var(--text-muted)' }}>
                  {plate.raw_text}
                </span>
              </div>
            )}

            <div className="person-side-field">
              <span className="person-side-label">จังหวัด</span>
              <span className="person-side-value">{plate.province || '—'}</span>
            </div>

            <div className="person-side-field">
              <span className="person-side-label">ประเภทป้าย</span>
              <span className="person-side-value">
                <span style={{
                  padding: '3px 10px', borderRadius: 999, fontSize: '0.85rem',
                  background: plate.plate_type === 'commercial' ? 'rgba(245,158,11,0.15)' : 'rgba(99,102,241,0.15)',
                  color: plate.plate_type === 'commercial' ? '#f59e0b' : '#818cf8',
                }}>
                  {plate.plate_type === 'commercial' ? 'พาณิชย์' : plate.plate_type === 'standard' ? 'มาตรฐาน' : plate.plate_type || 'ไม่ทราบ'}
                </span>
              </span>
            </div>

            <div className="person-side-field">
              <span className="person-side-label">ความมั่นใจ</span>
              <span className="person-side-value" style={{ color: confidenceColor, fontWeight: 700, fontSize: '1.1rem' }}>
                {confidence}%
              </span>
            </div>

            <div className="person-side-field">
              <span className="person-side-label">กล้อง</span>
              <span className="person-side-value">📹 {plate.camera_name}</span>
            </div>

            <div className="person-side-field">
              <span className="person-side-label">วันที่ตรวจพบ</span>
              <span className="person-side-value">
                {detectedAt.toLocaleDateString('th-TH', { year: 'numeric', month: 'long', day: 'numeric' })}
              </span>
            </div>

            <div className="person-side-field">
              <span className="person-side-label">เวลา</span>
              <span className="person-side-value">
                {detectedAt.toLocaleTimeString('th-TH')}
              </span>
            </div>
          </div>

          {plate.snapshot_url && (
            <div style={{ marginTop: 'auto', paddingTop: 16 }}>
              <button
                className="btn btn-ghost"
                style={{ width: '100%' }}
                onClick={() => window.open(plate.snapshot_url!, '_blank')}
              >
                เปิดรูปเต็ม ↗
              </button>
            </div>
          )}
        </div>
      </div>
    </Modal>
  );
}
