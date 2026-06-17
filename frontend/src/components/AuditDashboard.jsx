import { useCallback, useEffect, useState } from 'react'
import './AuditDashboard.css'

const PHASES = ['IDLE', 'PATIENT_IN_ROOM', 'SURGERY_ACTIVE', 'UNKNOWN']
const PAGE_SIZE = 18

const REASON_LABELS = {
  isolated_temporal_island: 'temporal island',
  local_majority_mismatch: 'local majority',
  low_confidence: 'low confidence',
  visual_near_duplicate_disagreement: 'visual neighbor',
}

function reasonLabel(reason) {
  const key = reason.split(':')[0]
  return REASON_LABELS[key] || key.replaceAll('_', ' ')
}

function shortRevision(value = '') {
  return value ? value.slice(0, 10) : ''
}

function stateMessage(status) {
  if (!status || status.state === 'idle') return ''
  if (status.state === 'error') return status.error || status.message || 'Action failed'
  return status.message || status.state
}

function auditStatusMessage(status) {
  if (!status) return ''
  if (status.ready) return ''
  if (status.cache_missing) {
    return `Published revision ${shortRevision(status.revision)} is not cached locally yet. Click Prepare Data to download it.`
  }
  return status.error || 'Prepare the SFT dataset to review train and validation labels.'
}

function CandidateCard({ item, onSave, onPreview }) {
  const correction = item.correction
  const initialPhase = correction?.corrected_phase || item.suggested_phase || item.phase
  const [phase, setPhase] = useState(initialPhase)
  const [note, setNote] = useState(correction?.note || '')
  const displayPhase = correction?.corrected_phase || item.phase

  useEffect(() => {
    setPhase(correction?.corrected_phase || item.suggested_phase || item.phase)
    setNote(correction?.note || '')
  }, [item])

  const save = (correctedPhase, reviewed = true) => {
    onSave({
      path: item.path,
      split: item.split,
      original_phase: item.phase,
      corrected_phase: correctedPhase,
      note,
      reviewed,
    })
  }

  return (
    <article className={`audit-card audit-${item.priority}`}>
      <button
        className="audit-image-button"
        type="button"
        aria-label={`Preview ${item.path}`}
        onClick={() => onPreview(item)}
      >
        <img src={item.image_url} alt={item.path} loading="lazy" />
        <span className="audit-zoom-label">Preview</span>
      </button>
      <div className="audit-card-body">
        <div className="audit-card-topline">
          <span className={`audit-priority ${item.priority}`}>{item.priority}</span>
          <span>{item.split}</span>
          <span>score {item.score}</span>
          {item.reviewed && <span className="audit-reviewed">reviewed</span>}
        </div>

        <div className="audit-phase-line">
          <span>{displayPhase}</span>
          {item.suggested_phase && (
            <>
              <span className="audit-arrow">to</span>
              <strong>{item.suggested_phase}</strong>
            </>
          )}
        </div>

        <div className="audit-path" title={item.path}>{item.path}</div>

        <div className="audit-reasons">
          {item.reasons.split('|').map((reason) => (
            <span key={reason}>{reasonLabel(reason)}</span>
          ))}
        </div>

        <div className="audit-neighbors">
          <span>prev: {item.prev_phase || '-'} ({item.prev_gap || '-'})</span>
          <span>next: {item.next_phase || '-'} ({item.next_gap || '-'})</span>
        </div>

        <label className="audit-select-label">
          Correction
          <select value={phase} onChange={(event) => setPhase(event.target.value)}>
            {PHASES.map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>

        <textarea
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="Reviewer note"
          rows="2"
        />

        <div className="audit-actions">
          <button onClick={() => save(item.suggested_phase || phase)} disabled={!item.suggested_phase && !phase}>
            Use Suggested
          </button>
          <button onClick={() => save(phase)}>Save</button>
          <button onClick={() => save('', true)}>No Change</button>
        </div>
      </div>
    </article>
  )
}

function AuditImagePreview({ item, onClose }) {
  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  if (!item) return null

  return (
    <div className="audit-lightbox-backdrop" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="audit-lightbox" onClick={(event) => event.stopPropagation()}>
        <div className="audit-lightbox-image">
          <img src={item.image_url} alt={item.path} />
        </div>
        <aside className="audit-lightbox-panel">
          <div className="audit-lightbox-top">
            <div>
              <span className={`audit-priority ${item.priority}`}>{item.priority}</span>
              <h3>{item.phase}{item.suggested_phase ? ` to ${item.suggested_phase}` : ''}</h3>
            </div>
            <button type="button" onClick={onClose}>Close</button>
          </div>
          <div className="audit-path" title={item.path}>{item.path}</div>
          <div className="audit-reasons">
            {item.reasons.split('|').map((reason) => (
              <span key={reason}>{reasonLabel(reason)}</span>
            ))}
          </div>
          <div className="audit-lightbox-meta">
            <span>split: {item.split}</span>
            <span>score: {item.score}</span>
            <span>frame: {item.frame_id}</span>
            <span>prev: {item.prev_phase || '-'} ({item.prev_gap || '-'})</span>
            <span>next: {item.next_phase || '-'} ({item.next_gap || '-'})</span>
          </div>
        </aside>
      </div>
    </div>
  )
}

export default function AuditDashboard() {
  const [status, setStatus] = useState(null)
  const [filters, setFilters] = useState({ reviewed: 'pending' })
  const [data, setData] = useState(null)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')
  const [previewItem, setPreviewItem] = useState(null)

  const refreshStatus = useCallback(async () => {
    const res = await fetch('/api/audit/status')
    const next = await res.json()
    setStatus(next)
    return next
  }, [])

  const fetchCandidates = useCallback(async () => {
    setLoading(true)
    const params = new URLSearchParams()
    params.set('page', page)
    params.set('page_size', PAGE_SIZE)
    Object.entries(filters).forEach(([key, value]) => {
      if (!value || value === 'all') return
      if (key === 'reviewed') {
        params.set('reviewed', value === 'reviewed' ? 'true' : 'false')
      } else {
        params.set(key, value)
      }
    })
    try {
      const res = await fetch(`/api/audit/candidates?${params}`)
      if (!res.ok) throw new Error((await res.json()).detail || 'Failed to load audit candidates')
      setData(await res.json())
    } catch (error) {
      setMessage(error.message)
    } finally {
      setLoading(false)
    }
  }, [filters, page])

  useEffect(() => {
    refreshStatus().catch((error) => setMessage(error.message))
  }, [refreshStatus])

  useEffect(() => {
    if (!status?.ready) return
    fetchCandidates()
  }, [status, fetchCandidates])

  useEffect(() => {
    if (status?.ready || status?.prepare?.state !== 'running') return
    const id = window.setInterval(() => {
      refreshStatus().catch((error) => setMessage(error.message))
    }, 2500)
    return () => window.clearInterval(id)
  }, [status, refreshStatus])

  useEffect(() => {
    if (status?.publish?.state !== 'running') return
    const id = window.setInterval(() => {
      refreshStatus().catch((error) => setMessage(error.message))
    }, 2500)
    return () => window.clearInterval(id)
  }, [status, refreshStatus])

  const prepare = async () => {
    setMessage('')
    const res = await fetch('/api/audit/prepare', { method: 'POST' })
    setStatus(await res.json())
  }

  const saveCorrection = async (payload) => {
    setMessage('')
    const res = await fetch('/api/audit/corrections', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!res.ok) {
      setMessage((await res.json()).detail || 'Correction failed')
      return
    }
    await fetchCandidates()
  }

  const stage = async () => {
    setMessage('Rebuilding corrected archives...')
    const res = await fetch('/api/audit/stage', { method: 'POST' })
    const result = await res.json()
    setMessage(res.ok ? `Stage ready: ${result.stage_dir}` : result.detail)
  }

  const publish = async () => {
    setMessage('Publishing corrected dataset to Hugging Face...')
    const res = await fetch('/api/audit/publish', { method: 'POST' })
    const result = await res.json()
    if (!res.ok) {
      setMessage(result.detail || 'Publish failed')
      return
    }
    setStatus((prev) => ({ ...prev, publish: result }))
    setMessage(stateMessage(result))
    refreshStatus().catch(console.error)
  }

  const updateFilter = (key, value) => {
    setFilters((prev) => ({ ...prev, [key]: value }))
    setPage(1)
  }

  const ready = status?.ready
  const prepareRunning = status?.prepare?.state === 'running'
  const publishState = status?.publish
  const publishRunning = publishState?.state === 'running'
  const publishText = stateMessage(publishState)
  const revision = ready ? shortRevision(status.revision) : ''

  return (
    <div className="audit-dashboard">
      <div className="audit-header">
        <div className="audit-title-block">
          <h2>Label Audit</h2>
          {ready ? (
            <div className="audit-status-row">
              <span>{status.repo_id}</span>
              <span>rev {revision}</span>
              {data && <span>{data.dataset.image_count} images</span>}
            </div>
          ) : (
            <p>{auditStatusMessage(status)}</p>
          )}
        </div>
        <div className="audit-header-actions">
          <button
            className="audit-toolbar-button"
            type="button"
            title="Download and extract the Hugging Face SFT dataset for local review"
            onClick={prepare}
            disabled={prepareRunning}
          >
            {prepareRunning ? 'Preparing...' : 'Prepare Data'}
          </button>
          <button
            className="audit-toolbar-button"
            type="button"
            title="Apply saved corrections and rebuild local train/validation archives"
            onClick={stage}
            disabled={!ready}
          >
            Rebuild Stage
          </button>
          <button
            className="audit-toolbar-button audit-publish-button"
            type="button"
            title="Upload the rebuilt stage back to the Hugging Face dataset"
            onClick={publish}
            disabled={!ready || publishRunning}
          >
            {publishRunning ? 'Publishing...' : 'Publish'}
          </button>
        </div>
      </div>

      {message && <div className="audit-message">{message}</div>}
      {publishText && publishText !== message && (
        <div className={`audit-message audit-message-${publishState.state}`}>
          {publishText}
        </div>
      )}

      {ready && (
        <>
          <div className="audit-filters">
            <label>
              Priority
              <select value={filters.priority || 'all'} onChange={(e) => updateFilter('priority', e.target.value)}>
                <option value="all">All</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
              </select>
            </label>
            <label>
              Split
              <select value={filters.split || 'all'} onChange={(e) => updateFilter('split', e.target.value)}>
                <option value="all">All</option>
                <option value="train">Train</option>
                <option value="validation">Validation</option>
              </select>
            </label>
            <label>
              Reason
              <select value={filters.reason || 'all'} onChange={(e) => updateFilter('reason', e.target.value)}>
                <option value="all">All</option>
                <option value="isolated_temporal_island">Temporal Island</option>
                <option value="local_majority_mismatch">Local Majority</option>
                <option value="low_confidence">Low Confidence</option>
                <option value="visual_near_duplicate_disagreement">Visual Neighbor</option>
              </select>
            </label>
            <label>
              Review
              <select value={filters.reviewed || 'pending'} onChange={(e) => updateFilter('reviewed', e.target.value)}>
                <option value="pending">Pending</option>
                <option value="reviewed">Reviewed</option>
                <option value="all">All</option>
              </select>
            </label>
          </div>

          {data && (
            <div className="audit-summary">
              <span>{data.filtered_summary.total} candidates</span>
              <span>{data.summary.by_priority.high || 0} high</span>
              <span>{data.total_pages} pages</span>
            </div>
          )}

          {loading && <div className="loading">Loading audit candidates...</div>}

          {!loading && data?.items?.length === 0 && (
            <div className="empty-state">No candidates match the current filters</div>
          )}

          <div className="audit-grid">
            {data?.items?.map((item) => (
              <CandidateCard
                key={item.path}
                item={item}
                onSave={saveCorrection}
                onPreview={setPreviewItem}
              />
            ))}
          </div>

          {data && data.total_pages > 1 && (
            <div className="audit-pagination">
              <button onClick={() => setPage((value) => Math.max(1, value - 1))} disabled={page <= 1}>
                Previous
              </button>
              <span>Page {page} of {data.total_pages}</span>
              <button onClick={() => setPage((value) => Math.min(data.total_pages, value + 1))} disabled={page >= data.total_pages}>
                Next
              </button>
            </div>
          )}
        </>
      )}
      <AuditImagePreview item={previewItem} onClose={() => setPreviewItem(null)} />
    </div>
  )
}
