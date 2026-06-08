import { useState } from 'react'
import FilterPanel from './components/FilterPanel'
import ImageCarousel from './components/ImageCarousel'
import ImageModal from './components/ImageModal'
import EvalDashboard from './components/EvalDashboard'
import { useImages } from './hooks/useImages'
import './App.css'

function App() {
  const [activeView, setActiveView] = useState('explorer')
  const { items, filters, filterOptions, stats, loading, hasMore, total, loadMore, updateFilters } = useImages()
  const [selectedImage, setSelectedImage] = useState(null)

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
        </nav>
        {activeView === 'explorer' && (
          <span className="total-count">{total} images</span>
        )}
      </header>

      {activeView === 'explorer' ? (
        <div className="app-body">
          <FilterPanel filterOptions={filterOptions} filters={filters} stats={stats} onChange={updateFilters} />
          <main className="app-main">
            <ImageCarousel
              items={items}
              hasMore={hasMore}
              loading={loading}
              loadMore={loadMore}
              onImageClick={setSelectedImage}
            />
          </main>
        </div>
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
