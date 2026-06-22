import React, { useEffect, useState, useRef } from 'react';
import type { DetectionEvent } from '../api/client';
import './Signage.css';

interface SignageProps {
  events: DetectionEvent[];
}

interface SignageCard extends DetectionEvent {
  uniqueId: string;
  accentIndex: number;
  initX: number;
  initY: number;
  dx1: number; dy1: number;
  dx2: number; dy2: number;
  dx3: number; dy3: number;
}

const CARD_LIFETIME_MS  = 30000;
const CARD_LIFETIME_CSS = `${CARD_LIFETIME_MS / 1000}s`;
const MAX_PROCESSED_SIGS = 200;

const ACCENTS = [
  { border: '#818cf8', glow: 'rgba(129,140,248,0.35)', bg: 'rgba(129,140,248,0.07)' },
  { border: '#c084fc', glow: 'rgba(192,132,252,0.35)', bg: 'rgba(192,132,252,0.07)' },
  { border: '#22d3ee', glow: 'rgba(34,211,238,0.35)',  bg: 'rgba(34,211,238,0.07)'  },
  { border: '#f472b6', glow: 'rgba(244,114,182,0.35)', bg: 'rgba(244,114,182,0.07)' },
  { border: '#34d399', glow: 'rgba(52,211,153,0.35)',  bg: 'rgba(52,211,153,0.07)'  },
  { border: '#fb923c', glow: 'rgba(251,146,60,0.35)',  bg: 'rgba(251,146,60,0.07)'  },
];

function rand(min: number, max: number) {
  return min + Math.random() * (max - min);
}

function drift() {
  return (Math.random() - 0.5) * 70;
}

export default function Signage({ events }: SignageProps) {
  const [cards, setCards] = useState<SignageCard[]>([]);
  const [currentTime, setCurrentTime] = useState(new Date());
  const processedSet = useRef<Set<string>>(new Set());
  const accentCounter = useRef(0);

  useEffect(() => {
    const interval = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const newCards: SignageCard[] = [];
    events.forEach(event => {
      const sig = `${event.timestamp}-${event.person_id ?? 'unknown'}-${event.camera_id}`;
      if (!processedSet.current.has(sig) && event.person_name && event.person_name !== 'Unknown') {
        processedSet.current.add(sig);
        if (Date.now() - new Date(event.timestamp).getTime() < CARD_LIFETIME_MS) {
          newCards.push({
            ...event,
            uniqueId: sig,
            accentIndex: accentCounter.current++ % ACCENTS.length,
            // Spawn position: spread across cards area, keeping card fully visible
            initX: rand(2, 62),
            initY: rand(2, 72),
            dx1: drift(), dy1: drift(),
            dx2: drift(), dy2: drift(),
            dx3: drift(), dy3: drift(),
          });
        }
      }
    });
    if (newCards.length > 0) {
      setCards(prev => [...prev, ...newCards.reverse()].slice(-12));
    }
  }, [events]);

  useEffect(() => {
    const interval = setInterval(() => {
      setCards(prev => {
        const now = Date.now();
        return prev.filter(card => now - new Date(card.timestamp).getTime() < CARD_LIFETIME_MS);
      });
      const seen = processedSet.current;
      if (seen.size > MAX_PROCESSED_SIGS) {
        processedSet.current = new Set([...seen].slice(seen.size - MAX_PROCESSED_SIGS));
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="signage-container">
      <div className="signage-aurora">
        <div className="aurora-orb orb-1" />
        <div className="aurora-orb orb-2" />
        <div className="aurora-orb orb-3" />
      </div>

      <header className="signage-header">
        <div className="signage-header-left">
          <div className="signage-logo">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="url(#logo-grad)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <defs>
                <linearGradient id="logo-grad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#818cf8" />
                  <stop offset="100%" stopColor="#c084fc" />
                </linearGradient>
              </defs>
              <circle cx="12" cy="8" r="4" />
              <path d="M6 20v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2" />
            </svg>
          </div>
          <div>
            <h1 className="signage-title">Class Attendance</h1>
            <p className="signage-subtitle">Real-time Face Recognition</p>
          </div>
        </div>

        <div className="signage-header-right">
          <div className="live-badge">
            <span className="live-dot" />
            LIVE
          </div>
          <div className="signage-clock">
            {currentTime.toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </div>
          <div className="signage-date">
            {currentTime.toLocaleDateString('th-TH', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
          </div>
        </div>
      </header>

      <main className="signage-cards-area">
        {cards.length === 0 && (
          <div className="signage-empty">
            <div className="empty-scanner">
              <svg width="72" height="72" viewBox="0 0 24 24" fill="none" stroke="rgba(148,163,184,0.25)" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="8" r="4" />
                <path d="M6 20v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2" />
              </svg>
              <div className="scanner-line" />
            </div>
            <p>กำลังรอข้อมูลการตรวจจับ</p>
            <span>จะแสดงเมื่อมีการตรวจพบบุคคลที่ระบบรู้จัก</span>
          </div>
        )}

        {cards.map((card) => {
          const accent = ACCENTS[card.accentIndex];
          return (
            <div
              key={card.uniqueId}
              className="signage-card"
              style={{
                '--accent-border': accent.border,
                '--accent-glow':   accent.glow,
                '--accent-bg':     accent.bg,
                '--dx1': `${card.dx1}px`, '--dy1': `${card.dy1}px`,
                '--dx2': `${card.dx2}px`, '--dy2': `${card.dy2}px`,
                '--dx3': `${card.dx3}px`, '--dy3': `${card.dy3}px`,
                '--lifetime': CARD_LIFETIME_CSS,
                left: `${card.initX}%`,
                top:  `${card.initY}%`,
              } as React.CSSProperties}
            >
              <div className="card-face-wrapper">
                <img
                  src={card.restored_face_url || card.face_crop_url || card.snapshot_url || ''}
                  alt={card.person_name}
                  className="card-face"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src =
                      'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="%2394a3b8" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/><path d="M6 20v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2"/></svg>';
                  }}
                />
              </div>

              <div className="card-body">
                <p className="card-name">{card.person_name}</p>
                {card.camera_name && (
                  <p className="card-camera">{card.camera_name}</p>
                )}
                <p className="card-time">
                  {new Date(card.timestamp).toLocaleTimeString('th-TH', {
                    hour: '2-digit', minute: '2-digit', second: '2-digit',
                  })}
                </p>
                <div className="card-progress">
                  <div className="card-progress-bar" />
                </div>
              </div>
            </div>
          );
        })}
      </main>
    </div>
  );
}
