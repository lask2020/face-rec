import { useState, useEffect, useRef, useCallback } from 'react';

interface CameraFeedProps {
  cameraId: number;
  isActive: boolean;
}

/**
 * Live camera feed via MSE (Media Source Extensions) through go2rtc.
 *
 * Uses go2rtc's WebSocket endpoint to receive fMP4 segments,
 * which are fed into a MediaSource for real-time playback.
 *
 * Why MSE instead of WebRTC:
 * - No STUN/ICE/NAT issues — works through any HTTP proxy
 * - Just a WebSocket connection — same as any other web technology
 * - Low latency (similar to WebRTC for most use cases)
 * - Falls back to snapshot polling if MSE is not supported
 */
export default function CameraFeed({ cameraId, isActive }: CameraFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const msRef = useRef<MediaSource | null>(null);
  const bufferRef = useRef<SourceBuffer | null>(null);
  const queueRef = useRef<ArrayBuffer[]>([]);
  const [connected, setConnected] = useState(false);
  const [mode, setMode] = useState<'mse' | 'snapshot'>('mse');
  const [snapshotSrc, setSnapshotSrc] = useState<string | null>(null);
  const [snapshotError, setSnapshotError] = useState(false);
  const mountedRef = useRef(true);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const objectUrlRef = useRef<string | null>(null);

  const streamName = `cam_${cameraId}`;

  // ─── MSE via go2rtc WebSocket ───────────────────────────────────────
  const connectMSE = useCallback(() => {
    if (!isActive || !mountedRef.current) return;

    cleanup();

    // Check if MSE is supported
    if (!('MediaSource' in window)) {
      console.warn(`[Camera ${cameraId}] MSE not supported, using snapshot`);
      setMode('snapshot');
      return;
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/go2rtc/api/ws?src=${streamName}`;

    const ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    const ms = new MediaSource();
    msRef.current = ms;

    const url = URL.createObjectURL(ms);
    objectUrlRef.current = url;

    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = url;
    }

    const sendMseRequest = () => {
      if (ws.readyState === WebSocket.OPEN && ms.readyState === 'open') {
        ws.send(JSON.stringify({ type: 'mse' }));
      }
    };

    ms.addEventListener('sourceopen', () => {
      sendMseRequest();
    });

    ws.onopen = () => {
      retryCountRef.current = 0;
      sendMseRequest();
    };

    ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        // JSON message — contains codec info for MSE
        const msg = JSON.parse(event.data);
        if (msg.type === 'mse') {
          // msg.value contains the codec MIME type
          const mimeCodec = msg.value;

          if (MediaSource.isTypeSupported(mimeCodec)) {
            if (ms.sourceBuffers.length > 0 || bufferRef.current) {
              console.log(`[Camera ${cameraId}] SourceBuffer already exists, skipping creation`);
              return;
            }
            try {
              const sb = ms.addSourceBuffer(mimeCodec);
              bufferRef.current = sb;

              sb.mode = 'segments';
              sb.addEventListener('updateend', () => {
                // Process queued buffers
                if (queueRef.current.length > 0 && !sb.updating) {
                  const next = queueRef.current.shift()!;
                  try {
                    sb.appendBuffer(next);
                  } catch (e) {
                    // Buffer full, skip
                  }
                  return;
                }

                // Clean up old buffered data to prevent QuotaExceededError (memory/buffer full freeze)
                if (videoRef.current && sb.buffered.length > 0 && !sb.updating) {
                  const start = sb.buffered.start(0);
                  const current = videoRef.current.currentTime;
                  if (current - start > 15) {
                    try {
                      sb.remove(start, current - 5);
                      return;
                    } catch (e) {
                      // Ignore removal error
                    }
                  }
                }

                // Keep latency low: seek to live edge if behind
                if (videoRef.current && sb.buffered.length > 0) {
                  const end = sb.buffered.end(sb.buffered.length - 1);
                  const current = videoRef.current.currentTime;
                  if (end - current > 2) {
                    videoRef.current.currentTime = end - 0.5;
                  }
                }
              });

              setConnected(true);
            } catch (e) {
              console.error(`[Camera ${cameraId}] Failed to create SourceBuffer:`, e);
              fallbackToSnapshot();
            }
          } else {
            console.warn(`[Camera ${cameraId}] Unsupported codec: ${mimeCodec}`);
            fallbackToSnapshot();
          }
        }
      } else {
        // Binary data — fMP4 segment
        const sb = bufferRef.current;
        if (sb && !sb.updating) {
          try {
            sb.appendBuffer(event.data);
          } catch (e) {
            // Buffer full or error — skip frame
          }
        } else if (sb) {
          // Queue the buffer if SourceBuffer is busy
          queueRef.current.push(event.data);
          // Keep queue small to avoid memory leak
          if (queueRef.current.length > 10) {
            queueRef.current.shift();
          }
        }
      }
    };

    ws.onerror = () => {
      console.warn(`[Camera ${cameraId}] WebSocket error`);
    };

    ws.onclose = () => {
      if (!mountedRef.current || !isActive) return;

      setConnected(false);

      // Auto-retry with exponential backoff
      const delay = Math.min(5000, 1000 * Math.pow(2, retryCountRef.current));
      retryCountRef.current++;

      if (retryCountRef.current <= 5) {
        retryTimerRef.current = setTimeout(() => {
          if (mountedRef.current && isActive && mode === 'mse') {
            connectMSE();
          }
        }, delay);
      } else {
        fallbackToSnapshot();
      }
    };
  }, [isActive, cameraId, streamName, mode]);

  function fallbackToSnapshot() {
    cleanup();
    if (mountedRef.current) {
      setMode('snapshot');
      setConnected(false);
    }
  }

  function cleanup() {
    if (retryTimerRef.current) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = undefined;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (msRef.current && msRef.current.readyState === 'open') {
      try {
        msRef.current.endOfStream();
      } catch (e) {
        // ignore
      }
    }
    msRef.current = null;
    bufferRef.current = null;
    queueRef.current = [];
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = '';
    }
    if (objectUrlRef.current) {
      try {
        URL.revokeObjectURL(objectUrlRef.current);
      } catch (e) {
        // ignore
      }
      objectUrlRef.current = null;
    }
    setConnected(false);
  }

  // ─── Snapshot Fallback ──────────────────────────────────────────────
  useEffect(() => {
    if (!isActive || mode !== 'snapshot') return;

    const refresh = () => {
      setSnapshotSrc(`/go2rtc/api/frame.jpeg?src=${streamName}&t=${Date.now()}`);
    };

    refresh();
    const interval = setInterval(refresh, 2000);
    return () => clearInterval(interval);
  }, [isActive, mode, streamName]);

  // ─── Lifecycle ──────────────────────────────────────────────────────
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      cleanup();
    };
  }, []);

  useEffect(() => {
    if (isActive && mode === 'mse') {
      connectMSE();
    }
    if (!isActive) {
      cleanup();
      setMode('mse');
      setSnapshotSrc(null);
      setSnapshotError(false);
      retryCountRef.current = 0;
    }
  }, [isActive, mode, connectMSE]);

  // ─── Render ─────────────────────────────────────────────────────────
  if (!isActive) {
    return (
      <div className="camera-feed">
        <div className="camera-feed-placeholder">
          <span className="icon">📷</span>
          <span>Camera Offline</span>
        </div>
      </div>
    );
  }

  return (
    <div className="camera-feed" style={{ position: 'relative' }}>
      {/* MSE Video */}
      {mode === 'mse' && (
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            display: connected ? 'block' : 'none',
          }}
        />
      )}

      {/* Snapshot Fallback */}
      {mode === 'snapshot' && snapshotSrc && (
        <img
          src={snapshotSrc}
          alt="Camera feed"
          onError={() => setSnapshotError(true)}
          onLoad={() => setSnapshotError(false)}
          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
        />
      )}

      {/* Loading state */}
      {!connected && mode === 'mse' && (
        <div className="camera-feed-placeholder">
          <span className="icon">📡</span>
          <span>Connecting...</span>
        </div>
      )}

      {/* Snapshot error */}
      {mode === 'snapshot' && snapshotError && (
        <div className="camera-feed-placeholder">
          <span className="icon">⚠️</span>
          <span>Stream unavailable</span>
        </div>
      )}

      {/* Status overlay */}
      <div className="camera-feed-overlay">
        <span className="live-indicator">
          {mode === 'mse' && connected
            ? '● LIVE'
            : mode === 'snapshot'
            ? '◉ SNAPSHOT'
            : '⏳ CONNECTING'}
        </span>
        {mode === 'snapshot' && (
          <button
            className="btn btn-ghost btn-sm"
            style={{
              position: 'absolute',
              bottom: '8px',
              right: '8px',
              fontSize: '11px',
              padding: '2px 8px',
              background: 'rgba(0,0,0,0.6)',
              color: '#fff',
              borderRadius: '4px',
            }}
            onClick={(e) => {
              e.stopPropagation();
              retryCountRef.current = 0;
              setMode('mse');
            }}
          >
            ↻ Try Live
          </button>
        )}
      </div>
    </div>
  );
}
