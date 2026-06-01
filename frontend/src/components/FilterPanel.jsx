import StatsSummary from './StatsSummary'

export default function FilterPanel({ filterOptions, filters, stats, onChange }) {
  const update = (key, value) => {
    onChange({ ...filters, [key]: value })
  }

  const reset = () => {
    onChange({})
  }

  if (!filterOptions) return null

  const confMin = filterOptions.confidence_range?.[0] ?? 0
  const confMax = filterOptions.confidence_range?.[1] ?? 1

  return (
    <aside className="filter-panel">
      <h2>Filters</h2>

      <label>
        Phase
        <select value={filters.phase || ''} onChange={(e) => update('phase', e.target.value)}>
          <option value="">All</option>
          {filterOptions.phases.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </label>

      <label>
        Surgery Type
        <select value={filters.surgery_type || ''} onChange={(e) => update('surgery_type', e.target.value)}>
          <option value="">All</option>
          {filterOptions.surgery_types.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </label>

      <label>
        Camera
        <select value={filters.camera || ''} onChange={(e) => update('camera', e.target.value)}>
          <option value="">All</option>
          {filterOptions.cameras.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </label>

      <label>
        Procedure ID
        <select value={filters.procedure_id || ''} onChange={(e) => update('procedure_id', e.target.value)}>
          <option value="">All</option>
          {filterOptions.procedure_ids.map((p) => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </label>

      <label>
        Take ID
        <select value={filters.take_id || ''} onChange={(e) => update('take_id', e.target.value)}>
          <option value="">All</option>
          {filterOptions.take_ids.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </label>

      <label className="range-label">
        Confidence: {filters.confidence_min ?? confMin} - {filters.confidence_max ?? confMax}
        <div className="range-inputs">
          <input
            type="range"
            min={confMin}
            max={confMax}
            step="0.01"
            value={filters.confidence_min ?? confMin}
            onChange={(e) => update('confidence_min', parseFloat(e.target.value))}
          />
          <input
            type="range"
            min={confMin}
            max={confMax}
            step="0.01"
            value={filters.confidence_max ?? confMax}
            onChange={(e) => update('confidence_max', parseFloat(e.target.value))}
          />
        </div>
      </label>

      <button className="reset-btn" onClick={reset}>Reset Filters</button>

      <StatsSummary stats={stats} />
    </aside>
  )
}
