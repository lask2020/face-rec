import React, { useEffect } from 'react';
// @ts-ignore
import { VideoRTC } from './video-rtc.js';

if (!customElements.get('video-rtc')) {
    customElements.define('video-rtc', VideoRTC);
}

declare global {
  namespace JSX {
    interface IntrinsicElements {
      'video-rtc': React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & { 
        src?: string; 
        autoplay?: boolean; 
        muted?: boolean; 
        playsinline?: boolean; 
        mode?: string; 
      };
    }
  }
}

interface CameraFeedProps {
  cameraId: number;
  isActive: boolean;
}

export default function CameraFeed({ cameraId, isActive }: CameraFeedProps) {
  const rtcRef = React.useRef<any>(null);

  const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${wsProtocol}//${window.location.host}/go2rtc/api/ws?src=cam_${cameraId}`;

  React.useEffect(() => {
    if (rtcRef.current && isActive) {
      // Must set as properties, not attributes, because video-rtc.js uses property setters
      rtcRef.current.mode = 'webrtc';
      rtcRef.current.media = 'video';
      rtcRef.current.src = wsUrl;
    }
  }, [wsUrl, isActive]);

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
      <video-rtc
        ref={rtcRef}
        mode="webrtc"
        autoplay={true}
        muted={true}
        playsinline={true}
        style={{
          width: '100%',
          height: '100%',
          display: 'block',
        }}
      ></video-rtc>
      <div className="camera-feed-overlay">
        <span className="live-indicator">● LIVE</span>
      </div>
    </div>
  );
}
