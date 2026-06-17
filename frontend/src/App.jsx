import { useCallback, useEffect, useState } from 'react'
import FilterPanel from './components/FilterPanel'
import ImageCarousel from './components/ImageCarousel'
import ImageModal from './components/ImageModal'
import EvalDashboard from './components/EvalDashboard'
import AuditDashboard from './components/AuditDashboard'
import { useImages } from './hooks/useImages'
import './App.css'

function App() {
  const [activeView, setActiveView] = useState('explorer')
  const { items, filters, filterOptions, stats, loading, hasMore, total, loadMore, updateFilters, refresh } = useImages()
  const [selectedImage, setSelectedImage] = useState(null)
  const [datasetStatus, setDatasetStatus] = useState(null)

  const refreshDatasetStatus = useCallback(async () => {
    const res = await fetch('/api/dataset/status')
    const data = await res.json()
    setDatasetStatus(data)
    if (data.ready) refresh()
    return data
  }, [refresh])

  async function downloadDataset() {
    const res = await fetch('/api/dataset/download', { method: 'POST' })
    const data = await res.json()
    setDatasetStatus(data)
  }

  useEffect(() => {
    const id = window.setTimeout(() => {
      refreshDatasetStatus().catch(console.error)
    }, 0)
    return () => window.clearTimeout(id)
  }, [refreshDatasetStatus])

  useEffect(() => {
    const downloadState = datasetStatus?.download?.state
    if (datasetStatus?.ready || (downloadState !== 'running' && downloadState !== 'complete')) return
    const id = window.setInterval(() => {
      refreshDatasetStatus().catch(console.error)
    }, 2500)
    return () => window.clearInterval(id)
  }, [datasetStatus, refreshDatasetStatus])

  const datasetReady = datasetStatus?.ready !== false || Boolean(filterOptions?.splits?.length)
  const downloadState = datasetStatus?.download?.state
  const downloadRunning = downloadState === 'running'

  return (
    <div className="app">
      <header className="app-header">
        <h1>OperAI-EYE</h1>
        <nav className="app-nav">
          <button
            className={`nav-button ${activeView === 'explorer' ? 'active' : ''}`}
            onClick={() => setActiveView('explorer')}
          >
            Explorer
          </button>
          <button
            className={`nav-button ${activeView === 'evaluation' ? 'active' : ''}`}
            onClick={() => setActiveView('evaluation')}
          >
            Evaluation
          </button>
          <button
            className={`nav-button ${activeView === 'audit' ? 'active' : ''}`}
            onClick={() => setActiveView('audit')}
          >
            Audit
          </button>
        </nav>
        {activeView === 'explorer' && (
          <span className="total-count">{total} images</span>
        )}
      </header>

      {activeView === 'explorer' ? (
        <div className="app-body">
          <FilterPanel filterOptions={filterOptions} filters={filters} stats={stats} onChange={updateFilters} />
          <main className="app-main">
            {datasetReady ? (
              <ImageCarousel
                items={items}
                hasMore={hasMore}
                loading={loading}
                loadMore={loadMore}
                onImageClick={setSelectedImage}
              />
            ) : (
              <section className="dataset-panel">
                <div>
                  <h2>Dataset Not Local</h2>
                  <p>{datasetStatus?.download?.error || datasetStatus?.download?.message || 'Download the private OperAI-EYE test set to browse images.'}</p>
                </div>
                <button className="dataset-download-button" onClick={downloadDataset} disabled={downloadRunning}>
                  {downloadRunning ? 'Downloading...' : 'Download from Hugging Face'}
                </button>
              </section>
            )}
          </main>
        </div>
      ) : activeView === 'audit' ? (
        <AuditDashboard />
      ) : (
        <EvalDashboard />
      )}

      {selectedImage && (
        <ImageModal item={selectedImage} onClose={() => setSelectedImage(null)} />
      )}
    </div>
  )
}

export default App
