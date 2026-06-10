import type { DetectionEvent } from '../api/client';

interface DetectionCardProps {
  event: DetectionEvent;
  onClick?: () => void;
}

function formatTime(timestamp: string): string {
  try {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', { hour12: false });
  } catch {
    return timestamp;
  }
}

function getConfidenceClass(confidence: number): string {
  if (confidence >= 0.7) return 'high';
  if (confidence >= 0.4) return 'medium';
  return 'low';
}

export default function DetectionCard({ event, onClick }: DetectionCardProps) {
  const isKnown = event.person_id !== null;
  const initial = event.person_name.charAt(0).toUpperCase();

  return (
    <div className="detection-item animate-in" onClick={onClick}>
      <div className={`detection-avatar ${isKnown ? 'known' : 'unknown'}`}>
        {initial}
      </div>
      <div className="detection-info">
        <div className="detection-name">
          {event.person_name}
        </div>
        <div className="detection-meta">
          <span>📹 {event.camera_name}</span>
        </div>
      </div>
      <span className={`detection-confidence ${getConfidenceClass(event.confidence)}`}>
        {(event.confidence * 100).toFixed(0)}%
      </span>
      <span className="detection-time">
        {formatTime(event.timestamp)}
      </span>
    </div>
  );
}
