import { useEffect, useState, useCallback, useRef } from 'react';
import { trainingApi } from '../api/client';
import type { TrainingSample, TrainingStats, CharLabel } from '../api/client';

const LIMIT = 20;

function parsedLabels(raw: string): CharLabel[] {
  try { return JSON.parse(raw) as CharLabel[]; } catch { return []; }
}

// ── Canvas overlay component ──────────────────────────────────────────────────

interface PlateCanvasProps {
  imageUrl: string;
  labels: CharLabel[];
  onCharClick?: (idx: number) => void;
}

function PlateCanvas({ imageUrl, labels, onCharClick }: PlateCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [displaySize, setDisplaySize] = useState<[number, number]>([240, 80]);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const maxW = 300;
      const scale = maxW / img.naturalWidth;
      setDisplaySize([maxW, Math.round(img.naturalHeight * scale)]);
    };
    img.src = imageUrl;
  }, [imageUrl]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = displaySize[0];
      canvas.height = displaySize[1];
      ctx.drawImage(img, 0, 0, displaySize[0], displaySize[1]);
      labels.forEach((lbl) => {
        const x = (lbl.cx - lbl.bw / 2) * displaySize[0];
        const y = (lbl.cy - lbl.bh / 2) * displaySize[1];
        const w = lbl.bw * displaySize[0];
        const h = lbl.bh * displaySize[1];
        ctx.strokeStyle = lbl.confidence > 0.7 ? '#22c55e' : lbl.confidence > 0.4 ? '#f59e0b' : '#ef4444';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = '9px sans-serif';
        ctx.fillText(lbl.class_name, x + 1, y > 10 ? y - 2 : y + h + 9);
      });
    };
    img.src = imageUrl;
  }, [imageUrl, labels, displaySize]);

  return (
    <canvas
      ref={canvasRef}
      style={{ display: 'block', cursor: onCharClick ? 'pointer' : 'default' }}
      onClick={(e) => {
        if (!onCharClick || labels.length === 0) return;
        const rect = (e.target as HTMLCanvasElement).getBoundingClientRect();
        const mx = (e.clientX - rect.left) / displaySize[0];
        const my = (e.clientY - rect.top) / displaySize[1];
        let closest = 0;
        let minDist = Infinity;
        labels.forEach((lbl, i) => {
          const d = Math.hypot(mx - lbl.cx, my - lbl.cy);
          if (d < minDist) { minDist = d; closest = i; }
        });
        onCharClick(closest);
      }}
    />
  );
}

// ── Stat bar chart ────────────────────────────────────────────────────────────

function ClassDistChart({ data }: { data: { class_name: string; count: number }[] }) {
  if (!data.length) return <p style={{ color: 'var(--text-secondary)', fontSize: 12 }}>No approved samples yet.</p>;
  const sorted = [...data].sort((a, b) => b.count - a.count).slice(0, 20);
  const max = sorted[0]?.count || 1;
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px 8px', alignItems: 'flex-end' }}>
      {sorted.map((item) => (
        <div key={item.class_name} style={{ textAlign: 'center', width: 28 }}>
          <div
            style={{
              background: 'var(--accent)',
              width: '100%',
              height: Math.max(4, (item.count / max) * 60),
              borderRadius: 2,
            }}
            title={`${item.class_name}: ${item.count}`}
          />
          <span style={{ fontSize: 9, color: 'var(--text-secondary)' }}>{item.class_name}</span>
        </div>
      ))}
    </div>
  );
}

// ── Sample card ───────────────────────────────────────────────────────────────

interface SampleCardProps {
  sample: TrainingSample;
  selected: boolean;
  onToggleSelect: () => void;
  onApprove: () => void;
  onReject: () => void;
  onCorrect: (correctedText: string) => void;
}

function SampleCard({ sample, selected, onToggleSelect, onApprove, onReject, onCorrect }: SampleCardProps) {
  const labels = parsedLabels(sample.char_labels);
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState(sample.corrected_text || sample.raw_text || '');

  const confColor = sample.confidence > 0.7 ? '#22c55e' : sample.confidence > 0.4 ? '#f59e0b' : '#ef4444';
  const statusBadge: Record<string, string> = {
    pending: '#64748b',
    approved: '#22c55e',
    rejected: '#ef4444',
  };

  return (
    <div
      style={{
        border: `2px solid ${selected ? 'var(--accent)' : 'var(--border)'}`,
        borderRadius: 8,
        overflow: 'hidden',
        background: 'var(--bg-card)',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* header */}
      <div style={{ display: 'flex', alignItems: 'center', padding: '6px 8px', gap: 6, background: 'var(--bg-secondary)' }}>
        <input type="checkbox" checked={selected} onChange={onToggleSelect} />
        <span style={{ fontSize: 11, fontWeight: 600 }}>{sample.raw_text || '—'}</span>
        <span style={{ marginLeft: 'auto', fontSize: 10, color: confColor }}>
          {(sample.confidence * 100).toFixed(0)}%
        </span>
        <span
          style={{
            fontSize: 10,
            padding: '1px 5px',
            borderRadius: 99,
            background: statusBadge[sample.status] ?? '#64748b',
            color: '#fff',
          }}
        >
          {sample.status}
        </span>
      </div>

      {/* image canvas */}
      <div style={{ padding: 6, background: '#000' }}>
        {sample.image_url ? (
          <PlateCanvas imageUrl={sample.image_url} labels={labels} />
        ) : (
          <div style={{ width: 240, height: 80, background: '#111', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666', fontSize: 11 }}>
            No image
          </div>
        )}
      </div>

      {/* char labels list */}
      {labels.length > 0 && (
        <div style={{ padding: '4px 6px', display: 'flex', flexWrap: 'wrap', gap: 3 }}>
          {labels.map((lbl, i) => (
            <span
              key={i}
              style={{
                fontSize: 11,
                padding: '1px 4px',
                borderRadius: 3,
                border: '1px solid var(--border)',
                color: lbl.confidence > 0.7 ? 'var(--text)' : '#f59e0b',
              }}
              title={`conf: ${(lbl.confidence * 100).toFixed(0)}%`}
            >
              {lbl.class_name}
            </span>
          ))}
        </div>
      )}

      {/* correction */}
      <div style={{ padding: '4px 6px', display: 'flex', gap: 4, alignItems: 'center' }}>
        {editing ? (
          <>
            <input
              value={editText}
              onChange={(e) => setEditText(e.target.value)}
              style={{ flex: 1, fontSize: 12, padding: '2px 4px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--bg-input)', color: 'var(--text)' }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') { onCorrect(editText); setEditing(false); }
                if (e.key === 'Escape') setEditing(false);
              }}
              autoFocus
            />
            <button onClick={() => { onCorrect(editText); setEditing(false); }} style={{ fontSize: 11, padding: '2px 6px' }}>✓</button>
            <button onClick={() => setEditing(false)} style={{ fontSize: 11, padding: '2px 6px' }}>✕</button>
          </>
        ) : (
          <button
            onClick={() => setEditing(true)}
            style={{ fontSize: 11, padding: '2px 6px', flex: 1, textAlign: 'left', background: 'transparent', border: '1px dashed var(--border)', borderRadius: 4, color: 'var(--text-secondary)', cursor: 'pointer' }}
          >
            {sample.corrected_text || '✎ correct text'}
          </button>
        )}
      </div>

      {/* action buttons */}
      <div style={{ display: 'flex', gap: 4, padding: '4px 6px' }}>
        <button
          onClick={onApprove}
          style={{ flex: 1, fontSize: 11, padding: '3px 0', background: '#16a34a', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer' }}
        >
          ✓ Approve
        </button>
        <button
          onClick={onReject}
          style={{ flex: 1, fontSize: 11, padding: '3px 0', background: '#dc2626', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer' }}
        >
          ✕ Reject
        </button>
      </div>

      {/* meta */}
      <div style={{ fontSize: 10, color: 'var(--text-secondary)', padding: '2px 6px 4px' }}>
        {sample.camera_name} · {new Date(sample.detected_at).toLocaleString()}
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function TrainingReview() {
  const [samples, setSamples] = useState<TrainingSample[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [stats, setStats] = useState<TrainingStats | null>(null);
  const [statusFilter, setStatusFilter] = useState('');
  const [confMax, setConfMax] = useState<string>('');
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [exportCount, setExportCount] = useState<number | null>(null);
  const [showStats, setShowStats] = useState(false);

  const totalPages = Math.max(1, Math.ceil(total / LIMIT));

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await trainingApi.list({
        page,
        limit: LIMIT,
        status: statusFilter || undefined,
        conf_max: confMax ? Number(confMax) : undefined,
      });
      setSamples(res.items);
      setTotal(res.total);
    } finally {
      setLoading(false);
    }
  }, [page, statusFilter, confMax]);

  const loadStats = useCallback(async () => {
    const s = await trainingApi.stats();
    setStats(s);
  }, []);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { loadStats(); }, [loadStats]);

  // Refresh export preview count whenever filters change
  useEffect(() => {
    trainingApi.exportPreview('approved', confMax ? Number(confMax) : undefined)
      .then((r) => setExportCount(r.total))
      .catch(() => setExportCount(null));
  }, [confMax]);

  const handleApprove = async (id: number) => {
    await trainingApi.update(id, { status: 'approved' });
    load(); loadStats();
  };
  const handleReject = async (id: number) => {
    await trainingApi.update(id, { status: 'rejected' });
    load(); loadStats();
  };
  const handleCorrect = async (id: number, correctedText: string) => {
    await trainingApi.update(id, { corrected_text: correctedText, status: 'approved' });
    load(); loadStats();
  };

  const handleBulkApprove = async () => {
    if (!selectedIds.size) return;
    await trainingApi.bulkUpdate([...selectedIds], 'approved');
    setSelectedIds(new Set());
    load(); loadStats();
  };
  const handleBulkReject = async () => {
    if (!selectedIds.size) return;
    await trainingApi.bulkUpdate([...selectedIds], 'rejected');
    setSelectedIds(new Set());
    load(); loadStats();
  };

  const toggleSelect = (id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const selectAll = () => {
    if (selectedIds.size === samples.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(samples.map((s) => s.id)));
    }
  };

  const pendingCount = stats?.by_status.find((s) => s.status === 'pending')?.count ?? 0;
  const approvedCount = stats?.by_status.find((s) => s.status === 'approved')?.count ?? 0;
  const rejectedCount = stats?.by_status.find((s) => s.status === 'rejected')?.count ?? 0;

  return (
    <div style={{ padding: '16px 20px' }}>
      {/* Page title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>Training Review</h1>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          Low-confidence plate crops for OCR retraining
        </span>
      </div>

      {/* Stats bar */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <div style={{ display: 'flex', gap: 8 }}>
          {[
            { label: 'Pending', count: pendingCount, color: '#64748b' },
            { label: 'Approved', count: approvedCount, color: '#22c55e' },
            { label: 'Rejected', count: rejectedCount, color: '#ef4444' },
          ].map(({ label, count, color }) => (
            <div
              key={label}
              style={{
                background: 'var(--bg-card)',
                border: '1px solid var(--border)',
                borderRadius: 8,
                padding: '6px 14px',
                textAlign: 'center',
                cursor: 'pointer',
              }}
              onClick={() => setStatusFilter(statusFilter === label.toLowerCase() ? '' : label.toLowerCase())}
            >
              <div style={{ fontSize: 18, fontWeight: 700, color }}>{count}</div>
              <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{label}</div>
            </div>
          ))}
        </div>

        <button
          onClick={() => setShowStats((s) => !s)}
          style={{ padding: '6px 12px', fontSize: 12, borderRadius: 6, border: '1px solid var(--border)', cursor: 'pointer', background: 'var(--bg-card)', color: 'var(--text)' }}
        >
          {showStats ? 'Hide' : 'Show'} Class Distribution
        </button>

        {/* Export button */}
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6, alignItems: 'center' }}>
          {exportCount !== null && (
            <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              {exportCount} approved sample{exportCount !== 1 ? 's' : ''} ready
            </span>
          )}
          <a
            href={trainingApi.exportUrl('approved', confMax ? Number(confMax) : undefined)}
            download
            style={{
              padding: '7px 14px',
              background: 'var(--accent)',
              color: '#fff',
              borderRadius: 6,
              fontSize: 12,
              fontWeight: 600,
              textDecoration: 'none',
            }}
          >
            Export ZIP
          </a>
        </div>
      </div>

      {/* Class distribution chart */}
      {showStats && stats && (
        <div
          style={{
            background: 'var(--bg-card)',
            border: '1px solid var(--border)',
            borderRadius: 8,
            padding: '12px 16px',
            marginBottom: 12,
          }}
        >
          <p style={{ margin: '0 0 8px', fontSize: 12, fontWeight: 600 }}>
            Class distribution (approved samples)
          </p>
          <ClassDistChart data={stats.by_class} />
        </div>
      )}

      {/* Filters + bulk actions */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <select
          value={statusFilter}
          onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
          style={{ padding: '6px 10px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg-input)', color: 'var(--text)', fontSize: 13 }}
        >
          <option value="">All statuses</option>
          <option value="pending">Pending</option>
          <option value="approved">Approved</option>
          <option value="rejected">Rejected</option>
        </select>

        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <label style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Conf ≤</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={confMax}
            onChange={(e) => { setConfMax(e.target.value); setPage(1); }}
            placeholder="max conf"
            style={{ width: 80, padding: '5px 8px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg-input)', color: 'var(--text)', fontSize: 13 }}
          />
        </div>

        {selectedIds.size > 0 && (
          <>
            <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{selectedIds.size} selected</span>
            <button
              onClick={handleBulkApprove}
              style={{ padding: '5px 10px', fontSize: 12, background: '#16a34a', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' }}
            >
              Approve all
            </button>
            <button
              onClick={handleBulkReject}
              style={{ padding: '5px 10px', fontSize: 12, background: '#dc2626', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' }}
            >
              Reject all
            </button>
          </>
        )}

        <button
          onClick={selectAll}
          style={{ marginLeft: 'auto', padding: '5px 10px', fontSize: 12, background: 'transparent', border: '1px solid var(--border)', borderRadius: 6, cursor: 'pointer', color: 'var(--text)' }}
        >
          {selectedIds.size === samples.length && samples.length > 0 ? 'Deselect all' : 'Select all'}
        </button>
      </div>

      {/* Grid */}
      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
          <div style={{ width: 32, height: 32, border: '3px solid var(--border)', borderTop: '3px solid var(--accent)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
        </div>
      ) : samples.length === 0 ? (
        <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-secondary)', fontSize: 14 }}>
          No training samples found.
        </div>
      ) : (
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: 12,
          }}
        >
          {samples.map((s) => (
            <SampleCard
              key={s.id}
              sample={s}
              selected={selectedIds.has(s.id)}
              onToggleSelect={() => toggleSelect(s.id)}
              onApprove={() => handleApprove(s.id)}
              onReject={() => handleReject(s.id)}
              onCorrect={(text) => handleCorrect(s.id, text)}
            />
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', gap: 6, marginTop: 16, justifyContent: 'center', alignItems: 'center' }}>
          <button
            disabled={page <= 1}
            onClick={() => setPage((p) => p - 1)}
            style={{ padding: '5px 12px', borderRadius: 6, border: '1px solid var(--border)', cursor: page <= 1 ? 'not-allowed' : 'pointer', background: 'var(--bg-card)', color: 'var(--text)' }}
          >
            ‹
          </button>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            {page} / {totalPages}
          </span>
          <button
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
            style={{ padding: '5px 12px', borderRadius: 6, border: '1px solid var(--border)', cursor: page >= totalPages ? 'not-allowed' : 'pointer', background: 'var(--bg-card)', color: 'var(--text)' }}
          >
            ›
          </button>
        </div>
      )}
    </div>
  );
}
