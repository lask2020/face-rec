import type { Person } from '../api/client';

interface PersonCardProps {
  person: Person;
  onClick: () => void;
}

export default function PersonCard({ person, onClick }: PersonCardProps) {
  const initial = person.name.charAt(0).toUpperCase();
  const hasAvatar = person.faces.length > 0;

  return (
    <div className="person-card animate-in" onClick={onClick}>
      <div className="person-avatar">
        {hasAvatar ? (
          <img src={person.faces[0].image_url} alt={person.name} />
        ) : (
          initial
        )}
      </div>
      <div className="person-name">{person.name}</div>
      <div className="person-dept">{person.department || 'No department'}</div>
      <div className="person-faces-count">
        🖼️ {person.face_count} face{person.face_count !== 1 ? 's' : ''}
      </div>
    </div>
  );
}
