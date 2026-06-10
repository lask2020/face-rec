import Sidebar from './Sidebar';

interface LayoutProps {
  children: React.ReactNode;
  wsConnected: boolean;
}

export default function Layout({ children, wsConnected }: LayoutProps) {
  return (
    <div className="app-layout">
      <Sidebar wsConnected={wsConnected} />
      <main className="main-content">
        {children}
      </main>
    </div>
  );
}
