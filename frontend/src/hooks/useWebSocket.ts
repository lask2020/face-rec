import { useEffect, useRef, useState, useCallback } from 'react';
import type { DetectionEvent, PlateDetectionEvent } from '../api/client';

export function useWebSocket() {
  const [events, setEvents] = useState<DetectionEvent[]>([]);
  const [plateEvents, setPlateEvents] = useState<PlateDetectionEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<number>();

  const connect = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/events`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        console.log('WebSocket connected');
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'plate_detection') {
            setPlateEvents((prev) => [data as PlateDetectionEvent, ...prev].slice(0, 100));
          } else {
            setEvents((prev) => [data as DetectionEvent, ...prev].slice(0, 100));
          }
        } catch (e) {
          console.error('Failed to parse WS event:', e);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimer.current = window.setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch (e) {
      console.error('WebSocket connection error:', e);
      reconnectTimer.current = window.setTimeout(connect, 3000);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [connect]);

  const clearEvents = useCallback(() => setEvents([]), []);

  return { events, plateEvents, connected, clearEvents };
}
