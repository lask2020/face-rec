import React, { useEffect, useState, useRef } from 'react';
import type { DetectionEvent } from '../api/client';
import './Signage.css';

interface SignageProps {
  events: DetectionEvent[];
}

interface SignageCard extends DetectionEvent {
  uniqueId: string;
}

const CARD_LIFETIME_MS = 15000; // 15 seconds
// Cap the dedup set so a long-running wall display doesn't leak memory.
// Safe because useWebSocket keeps at most 100 events, so anything still
// reachable is well within the most recent entries.
const MAX_PROCESSED_SIGS = 200;

export default function Signage({ events }: SignageProps) {
  const [cards, setCards] = useState<SignageCard[]>([]);
  const [currentTime, setCurrentTime] = useState(new Date());
  const processedSet = useRef<Set<string>>(new Set());

  // Clock tick
  useEffect(() => {
    const interval = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(interval);
  }, []);

  // Process incoming events
  useEffect(() => {
    const newCards: SignageCard[] = [];
    
    events.forEach(event => {
      // Create a unique signature for this detection
      const sig = `${event.timestamp}-${event.person_id || 'unknown'}-${event.camera_id}`;
      
      // Only process known persons
      if (!processedSet.current.has(sig) && event.person_name && event.person_name !== 'Unknown') {
        processedSet.current.add(sig);
        
        // Check if event is too old to be shown on mount
        const eventTime = new Date(event.timestamp).getTime();
        const now = new Date().getTime();
        if (now - eventTime < CARD_LIFETIME_MS) {
          newCards.push({ ...event, uniqueId: sig });
        }
      }
    });

    if (newCards.length > 0) {
      // Reverse because events are newest first, and we want to append oldest first
      setCards(prev => [...prev, ...newCards.reverse()].slice(-12)); // Max 12 cards on screen
    }
  }, [events]);

  // Clean up old cards periodically
  useEffect(() => {
    const interval = setInterval(() => {
      setCards(prev => {
        const now = new Date().getTime();
        return prev.filter(card => {
          const cardTime = new Date(card.timestamp).getTime();
          return now - cardTime < CARD_LIFETIME_MS;
        });
      });

      // Prune the dedup set so it never grows unbounded. Set preserves
      // insertion order, so keep only the most recent signatures.
      const seen = processedSet.current;
      if (seen.size > MAX_PROCESSED_SIGS) {
        processedSet.current = new Set(
          [...seen].slice(seen.size - MAX_PROCESSED_SIGS)
        );
      }
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="signage-container">
      <div className="signage-background"></div>
      
      <div className="signage-header">
        <div className="signage-header-title">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="url(#gradient)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <defs>
              <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#818cf8" />
                <stop offset="100%" stopColor="#c084fc" />
              </linearGradient>
            </defs>
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"></path>
            <path d="M12 8v4l3 3"></path>
          </svg>
          <h1>Class Attendance</h1>
        </div>
        <p className="time">
          {currentTime.toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
        </p>
      </div>

      <div className="signage-cards-area">
        {cards.length === 0 && (
          <div className="signage-empty">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="rgba(148,163,184,0.4)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="8" r="4"/>
              <path d="M6 20v-2a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v2"/>
            </svg>
            <p>กำลังรอข้อมูลการตรวจจับ...</p>
            <span>จะแสดงเมื่อมีการตรวจพบบุคคลที่ระบบรู้จัก</span>
          </div>
        )}
        {cards.map((card) => (
          <div key={card.uniqueId} className="signage-card">
            <div className="card-image-wrapper">
              <img 
                src={card.restored_face_url || card.face_crop_url || card.snapshot_url || ''} 
                alt={card.person_name} 
                className="card-image"
                onError={(e) => {
                  // Fallback if image fails to load
                  (e.target as HTMLImageElement).src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="160" height="160" viewBox="0 0 24 24" fill="none" stroke="%2394a3b8" stroke-width="1" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>';
                }}
              />
            </div>
            <div className="card-info">
              <h2 className="person-name">{card.person_name}</h2>
              <p className="detection-time">
                {new Date(card.timestamp).toLocaleTimeString('th-TH', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
