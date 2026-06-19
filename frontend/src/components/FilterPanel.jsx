import { useState, useEffect } from 'react'
import StatsSummary from './StatsSummary'

export default function FilterPanel({ filterOptions, filters, stats, onChange }) {
  const [models, setModels] = useState([])

  useEffect(() => {
    fetch('/api/eval/models')
      .then((r) => r.json())
      .then((data) => setModels(data))
      .catch((err) => console.error('Failed to fetch models:', err))
  }, [])

  const update = (key, value) => {
    onChange({ ...filters, [key]: value })
  }

  const reset = () => {
    onChange({})
  }

  if (!filterOptions) return null

  return (
    <aside className="filter-panel">
      <h2>Filters</h2>

      <label>
        Model
        <select
          value={filters.model_id || ''}
          onChange={(e) => {
            const modelId = e.target.value
            onChange({
              ...filters,
              model_id: modelId,
              prediction: modelId ? filters.prediction : '',
            })
          }}
        >
          <option value="">None</option>
          {models.map((m) => (
            <option key={m.model_id} value={m.model_id}>
              {m.model_name}
            </option>
          ))}
        </select>
      </label>

      {filters.model_id && (
        <label>
          Prediction
          <select
            value={filters.prediction || ''}
            onChange={(e) => update('prediction', e.target.value)}
          >
            <option value="">All</option>
            <option value="incorrect">Incorrect</option>
            <option value="correct">Correct</option>
            <option value="missing">Missing</option>
          </select>
        </label>
      )}

      <label>
        Split
        <select value={filters.split || ''} onChange={(e) => update('split', e.target.value)}>
          <option value="">All</option>
          {filterOptions.splits.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </label>

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

      <button className="reset-btn" onClick={reset}>Reset Filters</button>

      <StatsSummary stats={stats} />
    </aside>
  )
}
