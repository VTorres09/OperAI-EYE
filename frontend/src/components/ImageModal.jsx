import { useEffect } from 'react'

export default function ImageModal({ item, onClose }) {
  useEffect(() => {
    const handleEsc = (e) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleEsc)
    return () => document.removeEventListener('keydown', handleEsc)
  }, [onClose])

  if (!item) return null

  const PHASE_COLORS = {
    IDLE: '#6b7280',
    TURNOVER: '#f59e0b',
    PATIENT_IN_ROOM: '#3b82f6',
    SURGERY_ACTIVE: '#ef4444',
    UNKNOWN: '#9ca3af',
  }
  const color = PHASE_COLORS[item.phase] || '#9ca3af'

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>&times;</button>
        <img src={item.image_url} alt={item.path} className="modal-image" />
        <div className="modal-info">
          <span className="phase-badge" style={{ backgroundColor: color }}>
            {item.phase}
          </span>
          <div className="modal-details">
            <div><strong>Camera:</strong> {item.camera}</div>
            <div><strong>Confidence:</strong> {(item.confidence * 100).toFixed(0)}%</div>
            <div><strong>Procedure:</strong> {item.procedure_id}</div>
            <div><strong>Take:</strong> {item.take_id}</div>
            <div><strong>Frame:</strong> {item.frame_id}</div>
          </div>
          {item.key_visual_cues && (
            <div className="modal-cues">
              <strong>Visual Cues:</strong>
              <ul>
                {item.key_visual_cues.split('|').map((cue, i) => (
                  <li key={i}>{cue}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
