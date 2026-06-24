import { useEffect, useState, useCallback, useRef } from 'react';
import { trainingApi, modelApi, settingsApi, workersApi } from '../api/client';
import type { TrainingSample, TrainingStats, CharLabel, FinetuneStatus, ModelVersion, Worker } from '../api/client';

const LIMIT = 20;

// ── Class options for char label editor ──────────────────────────────────────

const DIGIT_OPTIONS = ['0','1','2','3','4','5','6','7','8','9'];

const THAI_CHAR_OPTIONS = [
  'ก','ข','ค','ฆ','ง','จ','ฉ','ช','ซ','ฌ',
  'ญ','ฎ','ฏ','ฐ','ฑ','ฒ','ณ','ด','ต','ถ',
  'ท','ธ','น','บ','ป','ผ','ฝ','พ','ฟ','ภ',
  'ม','ย','ร','ล','ว','ศ','ษ','ส','ห','ฬ',
  'อ','ฮ',
];

const PROVINCE_OPTIONS = [
  'ACR','ATG','AYA','BKK','BKN','BRM','CBI','CCO','CMI','CNT',
  'CPM','CPN','CRI','CTI','KBI','KKN','KPT','KRI','KSN','LEI',
  'LPG','LPN','LRI','MDH','MKM','NAN','NBI','NBP','NKI','NMA',
  'NPM','NPT','NRT','NSN','NST','NWT','NYK','PBI','PCT','PKN',
  'PKT','PLG','PLK','PNA','PNB','PRE','PRI','PTE','PTN','PYO',
  'RBR','RET','RNG','RYG','SBR','SKA','SKM','SKN','SKW','SNI',
  'SNK','SPB','SPK','SRI','SRN','SSK','STI','STN','TAK','TRG',
  'TRT','UBN','UDN','UTI','UTT','YLA','YST',
];

// ── Helpers ───────────────────────────────────────────────────────────────────

function parsedLabels(raw: string): CharLabel[] {
  try { return JSON.parse(raw) as CharLabel[]; } catch { return []; }
}

// ── Canvas bbox overlay ───────────────────────────────────────────────────────

interface PlateCanvasProps {
  imageUrl: string;
  labels: CharLabel[];
  selectedIdx: number | null;
  onCharClick: (idx: number) => void;
}

function PlateCanvas({ imageUrl, labels, selectedIdx, onCharClick }: PlateCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [displaySize, setDisplaySize] = useState<[number, number]>([280, 80]);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      const maxW = 280;
      setDisplaySize([maxW, Math.round(img.naturalHeight * (maxW / img.naturalWidth))]);
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
      const [w, h] = displaySize;
      canvas.width = w;
      canvas.height = h;
      ctx.drawImage(img, 0, 0, w, h);
      labels.forEach((lbl, i) => {
        const x = (lbl.cx - lbl.bw / 2) * w;
        const y = (lbl.cy - lbl.bh / 2) * h;
        const bw = lbl.bw * w;
        const bh = lbl.bh * h;
        const isSelected = i === selectedIdx;
        ctx.strokeStyle = isSelected
          ? '#facc15'
          : lbl.confidence > 0.7 ? '#22c55e' : lbl.confidence > 0.4 ? '#f59e0b' : '#ef4444';
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.strokeRect(x, y, bw, bh);
        if (isSelected) {
          ctx.fillStyle = 'rgba(250,204,21,0.15)';
          ctx.fillRect(x, y, bw, bh);
        }
        ctx.fillStyle = ctx.strokeStyle;
        ctx.font = 'bold 9px sans-serif';
        ctx.fillText(lbl.class_name, x + 1, y > 10 ? y - 2 : y + bh + 9);
      });
    };
    img.src = imageUrl;
  }, [imageUrl, labels, displaySize, selectedIdx]);

  return (
    <canvas
      ref={canvasRef}
      style={{ display: 'block', cursor: 'pointer' }}
      onClick={(e) => {
        if (!labels.length) return;
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

// ── Per-char label editor ─────────────────────────────────────────────────────

interface CharLabelEditorProps {
  labels: CharLabel[];
  selectedIdx: number | null;
  onSelect: (idx: number) => void;
  onChange: (idx: number, newClass: string) => void;
}

function CharLabelEditor({ labels, selectedIdx, onSelect, onChange }: CharLabelEditorProps) {
  if (!labels.length) return null;
  const selected = selectedIdx !== null ? labels[selectedIdx] : null;

  return (
    <div style={{ padding: '6px 6px 2px' }}>
      {/* Char badges row */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 6 }}>
        {labels.map((lbl, i) => {
          const isSelected = i === selectedIdx;
          const confColor = lbl.confidence > 0.7 ? '#22c55e' : lbl.confidence > 0.4 ? '#f59e0b' : '#ef4444';
          return (
            <button
              key={i}
              onClick={() => onSelect(i === selectedIdx ? -1 : i)}
              title={`conf: ${(lbl.confidence * 100).toFixed(0)}%`}
              style={{
                fontSize: 14,
                padding: '2px 7px',
                borderRadius: 4,
                border: `2px solid ${isSelected ? '#facc15' : confColor}`,
                background: isSelected ? 'rgba(250,204,21,0.15)' : 'transparent',
                color: 'var(--text)',
                cursor: 'pointer',
                fontWeight: isSelected ? 700 : 400,
              }}
            >
              {lbl.class_name}
            </button>
          );
        })}
      </div>

      {/* Class selector — shown when a char is selected */}
      {selected !== null && selectedIdx !== null && (
        <div
          style={{
            background: 'var(--bg-secondary)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            padding: '6px 8px',
          }}
        >
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginBottom: 4 }}>
            แก้ตัวที่ {selectedIdx + 1}: <strong style={{ color: 'var(--text)' }}>{selected.class_name}</strong>
            &nbsp;(conf {(selected.confidence * 100).toFixed(0)}%)
          </div>
          {/* Digit row */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 4 }}>
            {DIGIT_OPTIONS.map((d) => (
              <button key={d} onClick={() => onChange(selectedIdx, d)}
                style={classBtn(d === selected.class_name)}>{d}</button>
            ))}
          </div>
          {/* Thai char grid */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginBottom: 4 }}>
            {THAI_CHAR_OPTIONS.map((ch) => (
              <button key={ch} onClick={() => onChange(selectedIdx, ch)}
                style={classBtn(ch === selected.class_name)}>{ch}</button>
            ))}
          </div>
          {/* Province codes */}
          <details style={{ marginTop: 2 }}>
            <summary style={{ fontSize: 11, color: 'var(--text-secondary)', cursor: 'pointer', userSelect: 'none' }}>
              Province codes
            </summary>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginTop: 4 }}>
              {PROVINCE_OPTIONS.map((p) => (
                <button key={p} onClick={() => onChange(selectedIdx, p)}
                  style={classBtn(p === selected.class_name, true)}>{p}</button>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  );
}

function classBtn(active: boolean, small = false): React.CSSProperties {
  return {
    fontSize: small ? 10 : 12,
    padding: small ? '1px 4px' : '2px 6px',
    borderRadius: 3,
    border: `1px solid ${active ? '#facc15' : 'var(--border)'}`,
    background: active ? 'rgba(250,204,21,0.2)' : 'transparent',
    color: 'var(--text)',
    cursor: 'pointer',
    fontWeight: active ? 700 : 400,
  };
}

// ── Stats bar chart ───────────────────────────────────────────────────────────

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
  onApproveTrack: () => void;
  onReject: () => void;
  onSaveLabels: (charLabels: CharLabel[]) => void;
}

function SampleCard({
  sample, selected, onToggleSelect, onApprove, onApproveTrack, onReject, onSaveLabels,
}: SampleCardProps) {
  const [labels, setLabels] = useState<CharLabel[]>(() => parsedLabels(sample.char_labels));
  const [selectedCharIdx, setSelectedCharIdx] = useState<number | null>(null);
  const [labelsDirty, setLabelsDirty] = useState(false);

  // Sync if parent updates sample
  useEffect(() => {
    setLabels(parsedLabels(sample.char_labels));
    setLabelsDirty(false);
    setSelectedCharIdx(null);
  }, [sample.char_labels]);

  const handleCharChange = (idx: number, newClass: string) => {
    setLabels((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], class_name: newClass };
      return next;
    });
    setLabelsDirty(true);
    setSelectedCharIdx(null);
  };

  const handleSelectChar = (idx: number) => {
    setSelectedCharIdx(idx < 0 ? null : idx);
  };

  const confColor = sample.confidence > 0.7 ? '#22c55e' : sample.confidence > 0.4 ? '#f59e0b' : '#ef4444';
  const statusBadge: Record<string, string> = {
    pending: '#64748b', approved: '#22c55e', rejected: '#ef4444',
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
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', padding: '5px 8px', gap: 6, background: 'var(--bg-secondary)' }}>
        <input type="checkbox" checked={selected} onChange={onToggleSelect} />
        <span style={{ fontSize: 11, fontWeight: 600 }}>{sample.raw_text || '—'}</span>
        <span style={{ marginLeft: 'auto', fontSize: 10, color: confColor }}>
          {(sample.confidence * 100).toFixed(0)}%
        </span>
        <span style={{
          fontSize: 10, padding: '1px 5px', borderRadius: 99,
          background: statusBadge[sample.status] ?? '#64748b', color: '#fff',
        }}>
          {sample.status}
        </span>
      </div>

      {/* Canvas */}
      <div style={{ padding: 4, background: '#000' }}>
        {sample.image_url ? (
          <PlateCanvas
            imageUrl={sample.image_url}
            labels={labels}
            selectedIdx={selectedCharIdx}
            onCharClick={handleSelectChar}
          />
        ) : (
          <div style={{ width: 280, height: 80, background: '#111', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#666', fontSize: 11 }}>
            No image
          </div>
        )}
      </div>

      {/* Per-char editor */}
      <CharLabelEditor
        labels={labels}
        selectedIdx={selectedCharIdx}
        onSelect={handleSelectChar}
        onChange={handleCharChange}
      />

      {/* Save labels button — only when dirty */}
      {labelsDirty && (
        <div style={{ padding: '0 6px 4px' }}>
          <button
            onClick={() => { onSaveLabels(labels); setLabelsDirty(false); }}
            style={{
              width: '100%', padding: '4px', fontSize: 12,
              background: '#d97706', color: '#fff', border: 'none',
              borderRadius: 4, cursor: 'pointer', fontWeight: 600,
            }}
          >
            💾 Save labels
          </button>
        </div>
      )}

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: 3, padding: '4px 6px' }}>
        <button onClick={onApprove}
          style={{ flex: 1, fontSize: 11, padding: '4px 0', background: '#16a34a', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer' }}>
          ✓ Approve
        </button>
        <button onClick={onApproveTrack}
          title="Approve all frames from the same track"
          style={{ flex: 1, fontSize: 11, padding: '4px 0', background: '#0891b2', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer' }}>
          ✓✓ Track
        </button>
        <button onClick={onReject}
          style={{ flex: 1, fontSize: 11, padding: '4px 0', background: '#dc2626', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer' }}>
          ✕ Reject
        </button>
      </div>

      {/* Meta */}
      <div style={{ fontSize: 10, color: 'var(--text-secondary)', padding: '2px 6px 4px' }}>
        {sample.camera_name} · {new Date(sample.detected_at).toLocaleString()}
        {sample.track_id && (
          <span style={{ marginLeft: 6, opacity: 0.5 }}>#{sample.track_id.slice(0, 8)}</span>
        )}
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
  const [clearModal, setClearModal] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [finetune, setFinetune] = useState<FinetuneStatus | null>(null);
  const [finetuning, setFinetuning] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [epochs, setEpochs] = useState(30);
  const [roboflowKey, setRoboflowKey] = useState('');
  const [roboflowKeySaved, setRoboflowKeySaved] = useState(false);
  const [savingKey, setSavingKey] = useState(false);
  const [workers, setWorkers] = useState<Worker[]>([]);
  const [selectedWorker, setSelectedWorker] = useState<string>('');
  const [renamingWorker, setRenamingWorker] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const logBoxRef = useRef<HTMLDivElement>(null);
  const [modelVersions, setModelVersions] = useState<ModelVersion[]>([]);
  const [deployingVersion, setDeployingVersion] = useState<string | null>(null);
  const [showVersions, setShowVersions] = useState(false);

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

  const loadVersions = useCallback(async () => {
    try {
      const r = await modelApi.listVersions();
      setModelVersions(r.versions);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => { loadVersions(); }, [loadVersions]);

  useEffect(() => {
    settingsApi.get('roboflow_api_key').then((s) => {
      if (s.value) setRoboflowKey(s.value);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    const load = () => workersApi.list().then((r) => setWorkers(r.workers)).catch(() => {});
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  // Poll finetune status on load, and every 2s while running
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await trainingApi.finetuneStatus();
        if (!cancelled) setFinetune(s);
      } catch { /* ignore */ }
    };
    poll();
    const id = setInterval(async () => {
      const s = await trainingApi.finetuneStatus().catch(() => null);
      if (cancelled || !s) return;
      setFinetune(s);
      if (s.status !== 'running') {
        clearInterval(id);
        if (s.status === 'done') loadVersions(); // refresh versions list after training
      }
    }, 2000);
    return () => { cancelled = true; clearInterval(id); };
  }, [finetuning, loadVersions]);

  // Auto-scroll log box to bottom
  useEffect(() => {
    if (logBoxRef.current) logBoxRef.current.scrollTop = logBoxRef.current.scrollHeight;
  }, [finetune?.log?.length]);

  useEffect(() => {
    trainingApi.exportPreview('approved', confMax ? Number(confMax) : undefined)
      .then((r) => setExportCount(r.total))
      .catch(() => setExportCount(null));
  }, [confMax]);

  const handleApprove = async (id: number) => {
    await trainingApi.update(id, { status: 'approved' });
    load(); loadStats();
  };

  const handleApproveTrack = async (trackId: string) => {
    if (!trackId) return;
    await trainingApi.approveTrack(trackId, 'approved');
    load(); loadStats();
  };

  const handleReject = async (id: number) => {
    await trainingApi.update(id, { status: 'rejected' });
    load(); loadStats();
  };

  const handleClear = async (status?: string) => {
    setClearing(true);
    try {
      await trainingApi.clear(status);
      setClearModal(false);
      load();
      loadStats();
    } finally {
      setClearing(false);
    }
  };

  const handleSaveLabels = async (id: number, charLabels: CharLabel[]) => {
    await trainingApi.update(id, { char_labels: JSON.stringify(charLabels) });
    // no full reload needed — just update local state
    setSamples((prev) =>
      prev.map((s) => s.id === id ? { ...s, char_labels: JSON.stringify(charLabels) } : s)
    );
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

  const handleDeploy = async (version: string) => {
    if (deployingVersion) return;
    setDeployingVersion(version);
    try {
      await modelApi.deploy(version);
      await loadVersions();
    } catch (e: unknown) {
      alert('Deploy failed: ' + (e instanceof Error ? e.message : String(e)));
    } finally {
      setDeployingVersion(null);
    }
  };

  const handleSaveRoboflowKey = async () => {
    setSavingKey(true);
    try {
      await settingsApi.set('roboflow_api_key', roboflowKey);
      setRoboflowKeySaved(true);
      setTimeout(() => setRoboflowKeySaved(false), 2000);
    } catch { /* ignore */ } finally {
      setSavingKey(false);
    }
  };

  const handleStartFinetune = async () => {
    if (finetuning || finetune?.status === 'running') return;
    setFinetuning(true);
    try {
      await trainingApi.startFinetune(epochs, selectedWorker || undefined);
      setFinetuning(false);
    } catch (e: unknown) {
      alert('Failed to start training: ' + (e instanceof Error ? e.message : String(e)));
      setFinetuning(false);
    }
  };

  const handleStopFinetune = async () => {
    if (stopping) return;
    setStopping(true);
    try {
      await trainingApi.stopFinetune();
    } catch (e: unknown) {
      alert('Failed to stop training: ' + (e instanceof Error ? e.message : String(e)));
    } finally {
      setStopping(false);
    }
  };

  const pendingCount = stats?.by_status.find((s) => s.status === 'pending')?.count ?? 0;
  const approvedCount = stats?.by_status.find((s) => s.status === 'approved')?.count ?? 0;
  const rejectedCount = stats?.by_status.find((s) => s.status === 'rejected')?.count ?? 0;

  return (
    <div style={{ padding: '16px 20px' }}>
      {/* Clear confirmation modal */}
      {clearModal && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000,
        }}>
          <div style={{
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            borderRadius: 10, padding: 24, width: 340, boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
          }}>
            <h3 style={{ margin: '0 0 8px', fontSize: 16 }}>ล้างข้อมูล Training</h3>
            <p style={{ margin: '0 0 20px', fontSize: 13, color: 'var(--text-secondary)' }}>
              เลือกข้อมูลที่ต้องการลบ รวมถึงไฟล์รูปใน S3 ด้วย
            </p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <button
                disabled={clearing}
                onClick={() => handleClear('rejected')}
                style={{ padding: '9px 0', borderRadius: 6, border: 'none', cursor: 'pointer', background: '#7c3aed', color: '#fff', fontSize: 13, fontWeight: 600 }}
              >
                ลบเฉพาะ Rejected ({rejectedCount} รายการ)
              </button>
              <button
                disabled={clearing}
                onClick={() => handleClear('pending')}
                style={{ padding: '9px 0', borderRadius: 6, border: 'none', cursor: 'pointer', background: '#b45309', color: '#fff', fontSize: 13, fontWeight: 600 }}
              >
                ลบเฉพาะ Pending ({pendingCount} รายการ)
              </button>
              <button
                disabled={clearing}
                onClick={() => handleClear()}
                style={{ padding: '9px 0', borderRadius: 6, border: 'none', cursor: 'pointer', background: '#dc2626', color: '#fff', fontSize: 13, fontWeight: 600 }}
              >
                ลบทั้งหมด ({(pendingCount + approvedCount + rejectedCount)} รายการ)
              </button>
              <button
                disabled={clearing}
                onClick={() => setClearModal(false)}
                style={{ padding: '9px 0', borderRadius: 6, border: '1px solid var(--border)', cursor: 'pointer', background: 'transparent', color: 'var(--text)', fontSize: 13 }}
              >
                ยกเลิก
              </button>
            </div>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>Training Review</h1>
        <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          High-confidence auto-labeled plate crops for OCR retraining
        </span>
        <button
          onClick={() => setClearModal(true)}
          style={{
            marginLeft: 'auto', padding: '6px 14px', fontSize: 12, borderRadius: 6,
            border: '1px solid #dc2626', color: '#dc2626', background: 'transparent',
            cursor: 'pointer', fontWeight: 600,
          }}
        >
          ล้างข้อมูล
        </button>
      </div>

      {/* Stats summary */}
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
                background: 'var(--bg-card)', border: '1px solid var(--border)',
                borderRadius: 8, padding: '6px 14px', textAlign: 'center', cursor: 'pointer',
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
              padding: '7px 14px', background: 'var(--accent)', color: '#fff',
              borderRadius: 6, fontSize: 12, fontWeight: 600, textDecoration: 'none',
            }}
          >
            Export ZIP
          </a>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, minWidth: 200 }}>
            {/* Worker selector rows */}
            {[{ id: '', name: '', uptime: '', avg_process_ms: 0, is_paused: false, cameras: [], connected_at: '' } as Worker, ...workers].map((w, idx) => {
              const isAll = w.id === '';
              const label = isAll
                ? `All workers (${workers.length})`
                : (w.name || w.id.slice(0, 8));
              const isSelected = selectedWorker === w.id;
              const isRenaming = renamingWorker === w.id && !isAll;
              return (
                <div key={w.id || '__all'} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                  <button
                    onClick={() => !isAll || workers.length !== 1 ? setSelectedWorker(w.id) : null}
                    disabled={finetune?.status === 'running'}
                    style={{
                      flex: 1, textAlign: 'left', padding: '4px 8px', borderRadius: 6, fontSize: 12,
                      border: isSelected ? '1px solid var(--accent)' : '1px solid var(--border)',
                      background: isSelected ? 'color-mix(in srgb, var(--accent) 15%, var(--bg))' : 'var(--bg-input)',
                      color: 'var(--text)', cursor: 'pointer',
                    }}
                  >
                    {isRenaming ? (
                      <input
                        autoFocus
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onKeyDown={async (e) => {
                          if (e.key === 'Enter') {
                            await workersApi.rename(w.id, renameValue);
                            setWorkers(ws => ws.map(x => x.id === w.id ? { ...x, name: renameValue } : x));
                            setRenamingWorker(null);
                          } else if (e.key === 'Escape') {
                            setRenamingWorker(null);
                          }
                        }}
                        onClick={(e) => e.stopPropagation()}
                        style={{ background: 'transparent', border: 'none', outline: 'none', color: 'var(--text)', width: '100%', fontSize: 12 }}
                      />
                    ) : (
                      <span>
                        {label}
                        {!isAll && <span style={{ color: 'var(--text-muted)', marginLeft: 4, fontSize: 11 }}>
                          · {w.avg_process_ms.toFixed(0)}ms{w.is_paused ? ' ⏸' : ''}
                        </span>}
                      </span>
                    )}
                  </button>
                  {!isAll && !isRenaming && (
                    <button
                      title="Rename"
                      onClick={() => { setRenamingWorker(w.id); setRenameValue(w.name || ''); }}
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 12, padding: '2px 4px' }}
                    >✏️</button>
                  )}
                </div>
              );
            })}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <input
              type="password"
              value={roboflowKey}
              onChange={(e) => setRoboflowKey(e.target.value)}
              placeholder="Roboflow API key"
              style={{
                width: 160, padding: '5px 8px', borderRadius: 6,
                border: '1px solid var(--border)', background: 'var(--bg-input)',
                color: 'var(--text)', fontSize: 12,
              }}
            />
            <button
              onClick={handleSaveRoboflowKey}
              disabled={savingKey}
              style={{
                padding: '5px 10px', fontSize: 12, borderRadius: 6,
                border: '1px solid var(--border)',
                cursor: savingKey ? 'not-allowed' : 'pointer',
                background: roboflowKeySaved ? '#16a34a' : 'var(--bg-card)',
                color: roboflowKeySaved ? '#fff' : 'var(--text)',
              }}
            >
              {roboflowKeySaved ? 'Saved' : 'Save'}
            </button>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <label style={{ fontSize: 12, color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>Epochs</label>
            <input
              type="number" min={1} max={300} step={1}
              value={epochs}
              onChange={(e) => setEpochs(Math.max(1, Math.min(300, parseInt(e.target.value) || 30)))}
              disabled={finetune?.status === 'running'}
              style={{
                width: 56, padding: '5px 8px', borderRadius: 6,
                border: '1px solid var(--border)', background: 'var(--bg-input)',
                color: 'var(--text)', fontSize: 13, textAlign: 'center',
              }}
            />
          </div>
          {finetune?.status === 'running' ? (
            <button
              onClick={handleStopFinetune}
              disabled={stopping}
              style={{
                padding: '7px 14px', fontSize: 12, fontWeight: 600, borderRadius: 6, border: 'none',
                cursor: stopping ? 'not-allowed' : 'pointer',
                background: '#dc2626', color: '#fff', opacity: stopping ? 0.7 : 1,
              }}
            >
              {stopping ? 'Stopping...' : 'Stop Training'}
            </button>
          ) : (
            <button
              onClick={handleStartFinetune}
              disabled={finetuning}
              style={{
                padding: '7px 14px', fontSize: 12, fontWeight: 600, borderRadius: 6, border: 'none',
                cursor: finetuning ? 'not-allowed' : 'pointer',
                background: '#7c3aed', color: '#fff', opacity: finetuning ? 0.7 : 1,
              }}
            >
              {finetuning ? 'Starting...' : 'Train Model'}
            </button>
          )}
        </div>
      </div>

      {/* Class distribution */}
      {showStats && stats && (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, padding: '12px 16px', marginBottom: 12 }}>
          <p style={{ margin: '0 0 8px', fontSize: 12, fontWeight: 600 }}>Class distribution (pending + approved)</p>
          <ClassDistChart data={stats.by_class} />
        </div>
      )}

      {/* Finetune status panel */}
      {finetune && finetune.status !== 'idle' && (
        <div style={{
          background: 'var(--bg-card)', border: `1px solid ${
            finetune.status === 'error' ? '#dc2626' :
            finetune.status === 'done' ? '#16a34a' : '#7c3aed'
          }`,
          borderRadius: 8, padding: '12px 16px', marginBottom: 12,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
            <span style={{
              fontSize: 11, fontWeight: 700, padding: '2px 8px', borderRadius: 4,
              background: finetune.status === 'error' ? '#dc2626' : finetune.status === 'done' ? '#16a34a' : '#7c3aed',
              color: '#fff', textTransform: 'uppercase',
            }}>
              {finetune.status}
            </span>
            {finetune.status === 'running' && finetune.epochs > 0 && (
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                Epoch {finetune.epoch} / {finetune.epochs}
              </span>
            )}
            {finetune.status === 'running' && finetune.epochs > 0 && (
              <div style={{ flex: 1, height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{
                  width: `${Math.round((finetune.epoch / finetune.epochs) * 100)}%`,
                  height: '100%', background: '#7c3aed', transition: 'width 0.4s',
                }} />
              </div>
            )}
            {finetune.started_at && (
              <span style={{ fontSize: 11, color: 'var(--text-secondary)', marginLeft: 'auto' }}>
                Started {new Date(finetune.started_at).toLocaleTimeString()}
              </span>
            )}
          </div>
          {finetune.status === 'error' && finetune.error && (
            <div style={{ fontSize: 12, color: '#ef4444', marginBottom: 6 }}>{finetune.error}</div>
          )}
          {finetune.status === 'done' && (
            <div style={{ fontSize: 12, color: '#22c55e', marginBottom: 6 }}>
              Model saved (.pt + .onnx). Restart the AI worker to apply the new model.
            </div>
          )}
          {finetune.log && finetune.log.length > 0 && (
            <div
              ref={logBoxRef}
              style={{
                background: '#0a0a0a', borderRadius: 6, padding: '8px 10px',
                maxHeight: 180, overflowY: 'auto', fontFamily: 'monospace', fontSize: 11,
                color: '#a3a3a3', lineHeight: 1.6,
              }}
            >
              {finetune.log.map((line, i) => (
                <div key={i}>{line}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Model versions */}
      {modelVersions.length > 0 && (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, marginBottom: 12 }}>
          <button
            onClick={() => setShowVersions(v => !v)}
            style={{
              width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '10px 16px', background: 'transparent', border: 'none', cursor: 'pointer',
              color: 'var(--text)', fontSize: 13, fontWeight: 600,
            }}
          >
            <span>Model Versions ({modelVersions.length})</span>
            <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>{showVersions ? '▲' : '▼'}</span>
          </button>

          {showVersions && (
            <div style={{ borderTop: '1px solid var(--border)', padding: '8px 12px', display: 'flex', flexDirection: 'column', gap: 6 }}>
              {modelVersions.map((v) => (
                <div key={v.version} style={{
                  display: 'flex', alignItems: 'center', gap: 10, padding: '8px 10px',
                  borderRadius: 6, background: v.active ? 'rgba(124,58,237,0.1)' : 'transparent',
                  border: v.active ? '1px solid #7c3aed' : '1px solid transparent',
                }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{ fontSize: 12, fontWeight: 600, fontFamily: 'monospace' }}>
                        {v.version.replace(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/, '$1-$2-$3 $4:$5:$6')}
                      </span>
                      {v.active && (
                        <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 3, background: '#7c3aed', color: '#fff', fontWeight: 700 }}>
                          ACTIVE
                        </span>
                      )}
                      {!v.has_onnx && (
                        <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 3, background: '#78350f', color: '#fbbf24' }}>
                          .pt only
                        </span>
                      )}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>
                      {v.samples > 0 ? `${v.samples} samples · ` : ''}{v.epochs} epochs · {v.base_model}
                    </div>
                  </div>
                  {!v.active && (
                    <button
                      onClick={() => handleDeploy(v.version)}
                      disabled={!!deployingVersion}
                      style={{
                        padding: '5px 12px', fontSize: 12, fontWeight: 600, borderRadius: 6,
                        border: 'none', cursor: deployingVersion ? 'not-allowed' : 'pointer',
                        background: deployingVersion === v.version ? '#78350f' : '#7c3aed',
                        color: '#fff', opacity: deployingVersion && deployingVersion !== v.version ? 0.5 : 1,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {deployingVersion === v.version ? 'Deploying...' : 'Deploy'}
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Filters + bulk */}
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
            type="number" min={0} max={1} step={0.05}
            value={confMax}
            onChange={(e) => { setConfMax(e.target.value); setPage(1); }}
            placeholder="max conf"
            style={{ width: 80, padding: '5px 8px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--bg-input)', color: 'var(--text)', fontSize: 13 }}
          />
        </div>

        {selectedIds.size > 0 && (
          <>
            <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{selectedIds.size} selected</span>
            <button onClick={handleBulkApprove}
              style={{ padding: '5px 10px', fontSize: 12, background: '#16a34a', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' }}>
              Approve all
            </button>
            <button onClick={handleBulkReject}
              style={{ padding: '5px 10px', fontSize: 12, background: '#dc2626', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' }}>
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

      {/* Legend */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 10, fontSize: 11, color: 'var(--text-secondary)' }}>
        <span>คลิกตัวอักษรบน canvas หรือ badge เพื่อแก้ label &nbsp;·&nbsp;</span>
        <span style={{ color: '#0891b2' }}>✓✓ Track = approve ทุก frame ในกลุ่มเดียวกัน</span>
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
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 12 }}>
          {samples.map((s) => (
            <SampleCard
              key={s.id}
              sample={s}
              selected={selectedIds.has(s.id)}
              onToggleSelect={() => toggleSelect(s.id)}
              onApprove={() => handleApprove(s.id)}
              onApproveTrack={() => handleApproveTrack(s.track_id)}
              onReject={() => handleReject(s.id)}
              onSaveLabels={(labels) => handleSaveLabels(s.id, labels)}
            />
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div style={{ display: 'flex', gap: 6, marginTop: 16, justifyContent: 'center', alignItems: 'center' }}>
          <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}
            style={{ padding: '5px 12px', borderRadius: 6, border: '1px solid var(--border)', cursor: page <= 1 ? 'not-allowed' : 'pointer', background: 'var(--bg-card)', color: 'var(--text)' }}>
            ‹
          </button>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{page} / {totalPages}</span>
          <button disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}
            style={{ padding: '5px 12px', borderRadius: 6, border: '1px solid var(--border)', cursor: page >= totalPages ? 'not-allowed' : 'pointer', background: 'var(--bg-card)', color: 'var(--text)' }}>
            ›
          </button>
        </div>
      )}
    </div>
  );
}
