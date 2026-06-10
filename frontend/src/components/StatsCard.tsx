interface StatsCardProps {
  icon: string;
  label: string;
  value: number | string;
  color: 'blue' | 'green' | 'amber' | 'purple';
}

export default function StatsCard({ icon, label, value, color }: StatsCardProps) {
  return (
    <div className={`stat-card ${color} animate-in`}>
      <div className={`stat-icon ${color}`}>{icon}</div>
      <div className="stat-info">
        <div className="stat-value">{value}</div>
        <div className="stat-label">{label}</div>
      </div>
    </div>
  );
}
