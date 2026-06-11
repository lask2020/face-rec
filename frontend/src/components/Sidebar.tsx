import { NavLink, useLocation } from 'react-router-dom';

const navItems = [
  { path: '/', icon: '📊', label: 'Dashboard' },
  { path: '/persons', icon: '👤', label: 'Face Management' },
  { path: '/cameras', icon: '📹', label: 'Cameras' },
  { path: '/detections', icon: '🔍', label: 'Detection Logs' },
  { path: '/workers', icon: '🤖', label: 'AI Workers' },
];

interface SidebarProps {
  wsConnected: boolean;
}

export default function Sidebar({ wsConnected }: SidebarProps) {
  const location = useLocation();

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">🎯</div>
        <div>
          <div className="sidebar-logo-text">FaceRec</div>
          <div className="sidebar-logo-sub">CCTV System</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={`sidebar-nav-link ${
              location.pathname === item.path ||
              (item.path !== '/' && location.pathname.startsWith(item.path))
                ? 'active'
                : ''
            }`}
          >
            <span className="nav-icon">{item.icon}</span>
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-status">
          <span
            className="status-dot"
            style={{
              background: wsConnected ? 'var(--accent-emerald)' : 'var(--accent-red)',
              boxShadow: wsConnected
                ? '0 0 8px rgba(16,185,129,0.5)'
                : '0 0 8px rgba(239,68,68,0.5)',
            }}
          />
          <span>{wsConnected ? 'System Online' : 'Connecting...'}</span>
        </div>
      </div>
    </aside>
  );
}
