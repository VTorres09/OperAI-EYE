import React, { useState, useEffect, useCallback } from 'react'
import ModelSelector from './ModelSelector'
import MetricsDisplay from './MetricsDisplay'
import ConfusionMatrix from './ConfusionMatrix'
import './EvalDashboard.css'

function EvalDashboard() {
  const [models, setModels] = useState([])
  const [selectedModels, setSelectedModels] = useState([])
  const [results, setResults] = useState({})
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading] = useState(false)
  const [filterOptions, setFilterOptions] = useState(null)
  const [filters, setFilters] = useState({})
  const [mergePatientAndSurgery, setMergePatientAndSurgery] = useState(false)

  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch('/api/eval/models')
      const data = await res.json()
      setModels(data)
    } catch (err) {
      console.error('Failed to fetch models:', err)
    }
  }, [])

  const fetchFilterOptions = useCallback(async () => {
    try {
      const res = await fetch('/api/eval/filters')
      const data = await res.json()
      setFilterOptions(data)
    } catch (err) {
      console.error('Failed to fetch filter options:', err)
    }
  }, [])

  const buildFilterParams = useCallback(() => {
    const params = new URLSearchParams()
    Object.entries(filters).forEach(([key, val]) => {
      if (val !== undefined && val !== null && val !== '') params.set(key, val)
    })
    if (mergePatientAndSurgery) {
      params.set('merge_patient_and_surgery', 'true')
    }
    return params
  }, [filters, mergePatientAndSurgery])

  const fetchResults = useCallback(async () => {
    setLoading(true)
    try {
      const newResults = {}
      const filterParams = buildFilterParams()
      for (const modelId of selectedModels) {
        const res = await fetch(`/api/eval/results/${modelId}?${filterParams}`)
        if (res.ok) {
          newResults[modelId] = await res.json()
        }
      }
      setResults(newResults)

      if (selectedModels.length > 1) {
        const res = await fetch(`/api/eval/compare?model_ids=${selectedModels.join(',')}&${filterParams}`)
        if (res.ok) {
          setComparison(await res.json())
        }
      } else {
        setComparison(null)
      }
    } catch (err) {
      console.error('Failed to fetch results:', err)
    } finally {
      setLoading(false)
    }
  }, [buildFilterParams, selectedModels])

  useEffect(() => {
    const id = window.setTimeout(() => {
      fetchModels()
      fetchFilterOptions()
    }, 0)
    return () => window.clearTimeout(id)
  }, [fetchModels, fetchFilterOptions])

  useEffect(() => {
    if (selectedModels.length === 0) return undefined
    const id = window.setTimeout(() => {
      fetchResults()
    }, 0)
    return () => window.clearTimeout(id)
  }, [selectedModels, filters, mergePatientAndSurgery, fetchResults])

  const handleModelToggle = (modelId) => {
    setSelectedModels(prev =>
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    )
  }

  const updateFilter = (key, value) => {
    setFilters(prev => ({ ...prev, [key]: value }))
  }

  const resetFilters = () => {
    setFilters({})
    setMergePatientAndSurgery(false)
  }

  return (
    <div className="eval-dashboard">
      <div className="eval-header">
        <h2>Model Evaluation</h2>
        <p className="eval-description">
          Compare evaluation results across different model versions
        </p>
      </div>

      {filterOptions && (
        <div className="eval-filters">
          <label>
            Phase
            <select value={filters.phase || ''} onChange={(e) => updateFilter('phase', e.target.value)}>
              <option value="">All</option>
              {filterOptions.phases.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </label>
          <label>
            Surgery Type
            <select value={filters.surgery_type || ''} onChange={(e) => updateFilter('surgery_type', e.target.value)}>
              <option value="">All</option>
              {filterOptions.surgery_types.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </label>
          <label>
            Camera
            <select value={filters.camera || ''} onChange={(e) => updateFilter('camera', e.target.value)}>
              <option value="">All</option>
              {filterOptions.cameras.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </label>
          <label>
            Procedure ID
            <select value={filters.procedure_id || ''} onChange={(e) => updateFilter('procedure_id', e.target.value)}>
              <option value="">All</option>
              {filterOptions.procedure_ids.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </label>
          <label>
            Take ID
            <select value={filters.take_id || ''} onChange={(e) => updateFilter('take_id', e.target.value)}>
              <option value="">All</option>
              {filterOptions.take_ids.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </label>
          <label className="eval-checkbox-label">
            <input
              type="checkbox"
              checked={mergePatientAndSurgery}
              onChange={(e) => setMergePatientAndSurgery(e.target.checked)}
            />
            Merge patient/surgery
          </label>
          <button className="reset-btn" onClick={resetFilters} style={{ alignSelf: 'flex-end' }}>
            Reset
          </button>
        </div>
      )}

      <ModelSelector
        models={models}
        selectedModels={selectedModels}
        onToggle={handleModelToggle}
      />

      {loading && <div className="loading">Loading results...</div>}

      {!loading && selectedModels.length === 0 && (
        <div className="empty-state">
          Select one or more models to view evaluation results
        </div>
      )}

      {!loading && selectedModels.length === 1 && results[selectedModels[0]] && (
        <div className="single-model-view">
          <MetricsDisplay result={results[selectedModels[0]]} />
          <ConfusionMatrix
            matrix={results[selectedModels[0]].confusion_matrix}
            classes={Object.keys(results[selectedModels[0]].confusion_matrix)}
          />
        </div>
      )}

      {!loading && selectedModels.length > 1 && comparison && (
        <div className="comparison-view">
          <h3>Model Comparison</h3>
          <div className="comparison-table">
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Accuracy</th>
                  <th>Total</th>
                  {comparison.classes.map(cls => (
                    <th key={cls} colSpan="3">{cls}</th>
                  ))}
                </tr>
                <tr>
                  <th></th>
                  <th></th>
                  <th></th>
                  {comparison.classes.map(cls => (
                    <React.Fragment key={cls}>
                      <th>P</th>
                      <th>R</th>
                      <th>F1</th>
                    </React.Fragment>
                  ))}
                </tr>
              </thead>
              <tbody>
                {comparison.models.map(model => (
                  <tr key={model.model_id}>
                    <td>{model.model_name}</td>
                    <td className="accuracy">{(model.accuracy * 100).toFixed(1)}%</td>
                    <td>{model.total}</td>
                    {comparison.classes.map(cls => {
                      const metrics = model.per_class[cls] || { precision: 0, recall: 0, f1: 0 }
                      return (
                        <React.Fragment key={`${model.model_id}-${cls}`}>
                          <td>{(metrics.precision * 100).toFixed(1)}%</td>
                          <td>{(metrics.recall * 100).toFixed(1)}%</td>
                          <td>{(metrics.f1 * 100).toFixed(1)}%</td>
                        </React.Fragment>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

export default EvalDashboard
