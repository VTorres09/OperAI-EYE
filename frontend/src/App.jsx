import { useState } from 'react'
import FilterPanel from './components/FilterPanel'
import ImageCarousel from './components/ImageCarousel'
import ImageModal from './components/ImageModal'
import { useImages } from './hooks/useImages'
import './App.css'

function App() {
  const { items, filters, filterOptions, stats, loading, hasMore, total, loadMore, updateFilters } = useImages()
  const [selectedImage, setSelectedImage] = useState(null)

  return (
    <div className="app">
      <header className="app-header">
        <h1>OperAI-EYE Explorer</h1>
        <span className="total-count">{total} images</span>
      </header>
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
      {selectedImage && (
        <ImageModal item={selectedImage} onClose={() => setSelectedImage(null)} />
      )}
    </div>
  )
}

export default App
