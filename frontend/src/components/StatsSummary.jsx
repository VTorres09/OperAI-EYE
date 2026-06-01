const PHASE_COLORS = {
  IDLE: '#6b7280',
  TURNOVER: '#f59e0b',
  PATIENT_IN_ROOM: '#3b82f6',
  SURGERY_ACTIVE: '#ef4444',
  UNKNOWN: '#9ca3af',
  ERROR: '#dc2626',
}

function Bar({ label, count, pct, color }) {
  return (
    <div className="stat-bar-row">
      <div className="stat-bar-header">
        <span className="stat-bar-label">{label}</span>
        <span className="stat-bar-count">{count} <span className="stat-bar-pct">({pct}%)</span></span>
      </div>
      <div className="stat-bar-track">
        <div
          className="stat-bar-fill"
          style={{ width: `${pct}%`, backgroundColor: color || 'var(--accent)' }}
        />
      </div>
    </div>
  )
}

function Section({ title, items, colorMap }) {
  if (!items || items.length === 0) return null
  return (
    <div className="stat-section">
      <h3>{title}</h3>
      {items.map((item) => (
        <Bar
          key={item.label}
          label={item.label}
          count={item.count}
          pct={item.pct}
          color={colorMap?.[item.label]}
        />
      ))}
    </div>
  )
}

export default function StatsSummary({ stats }) {
  if (!stats) return null

  return (
    <div className="stats-summary">
      <h2>Distribution</h2>
      {stats.avg_confidence != null && (
        <div className="stat-avg">
          Avg confidence: <strong>{(stats.avg_confidence * 100).toFixed(1)}%</strong>
        </div>
      )}
      <Section title="Phase" items={stats.by_phase} colorMap={PHASE_COLORS} />
      <Section title="Camera" items={stats.by_camera} />
      {stats.by_surgery_type?.length > 1 && (
        <Section title="Surgery Type" items={stats.by_surgery_type} />
      )}
    </div>
  )
}
