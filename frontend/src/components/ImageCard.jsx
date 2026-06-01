import { useState } from 'react'

const PHASE_COLORS = {
  IDLE: '#6b7280',
  TURNOVER: '#f59e0b',
  PATIENT_IN_ROOM: '#3b82f6',
  SURGERY_ACTIVE: '#ef4444',
  UNKNOWN: '#9ca3af',
}

export default function ImageCard({ item, onClick }) {
  const [loaded, setLoaded] = useState(false)
  const color = PHASE_COLORS[item.phase] || '#9ca3af'

  return (
    <div className="image-card" onClick={() => onClick(item)}>
      <div className={`image-placeholder ${loaded ? 'loaded' : ''}`}>
        {!loaded && <div className="skeleton" />}
      </div>
      <img
        src={item.image_url}
        alt={item.path}
        loading="lazy"
        decoding="async"
        className={loaded ? 'loaded' : ''}
        onLoad={() => setLoaded(true)}
      />
      <div className="image-card-overlay">
        <span className="phase-badge" style={{ backgroundColor: color }}>
          {item.phase}
        </span>
        <div className="image-card-meta">
          <span>{item.camera}</span>
          <span>{(item.confidence * 100).toFixed(0)}%</span>
        </div>
        <div className="image-card-detail">
          <span>P{item.procedure_id} / T{item.take_id}</span>
          <span>Frame {item.frame_id}</span>
        </div>
      </div>
    </div>
  )
}
