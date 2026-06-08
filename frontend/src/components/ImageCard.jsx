import { useState } from 'react'

const PHASE_COLORS = {
  IDLE: '#6b7280',
  PATIENT_IN_ROOM: '#3b82f6',
  SURGERY_ACTIVE: '#ef4444',
  UNKNOWN: '#9ca3af',
}

export default function ImageCard({ item, onClick }) {
  const [loaded, setLoaded] = useState(false)
  const color = PHASE_COLORS[item.phase] || '#9ca3af'
  const hasModelPred = item.model_predicted != null
  const modelColor = PHASE_COLORS[item.model_predicted] || '#9ca3af'
  const isCorrect = hasModelPred && item.model_predicted === item.phase

  const borderStyle = hasModelPred
    ? isCorrect
      ? '2px solid #059669'
      : '2px solid #ef4444'
    : 'none'

  return (
    <div
      className="image-card"
      onClick={() => onClick(item)}
      style={{ border: borderStyle }}
    >
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
        <div className="image-card-badges">
          <span className="phase-badge" style={{ backgroundColor: color }}>
            {item.phase}
          </span>
          {hasModelPred && (
            <span
              className="phase-badge model-badge"
              style={{ backgroundColor: modelColor }}
              title={`Model prediction: ${item.model_predicted} (${isCorrect ? 'correct' : 'incorrect'})`}
            >
              {item.model_predicted}
            </span>
          )}
        </div>
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
