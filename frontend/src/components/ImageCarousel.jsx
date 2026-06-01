import { useRef, useCallback } from 'react'
import ImageCard from './ImageCard'

export default function ImageCarousel({ items, hasMore, loading, loadMore, onImageClick }) {
  const observerRef = useRef(null)

  const sentinelRef = useCallback(
    (node) => {
      if (observerRef.current) observerRef.current.disconnect()
      if (!node || !hasMore || loading) return
      observerRef.current = new IntersectionObserver(
        (entries) => {
          if (entries[0].isIntersecting) loadMore()
        },
        { rootMargin: '400px' }
      )
      observerRef.current.observe(node)
    },
    [hasMore, loading, loadMore]
  )

  return (
    <div className="image-grid">
      {items.map((item, i) => (
        <ImageCard key={`${item.path}-${i}`} item={item} onClick={onImageClick} />
      ))}
      {hasMore && <div ref={sentinelRef} className="sentinel" />}
      {loading && (
        <div className="loading-indicator">Loading...</div>
      )}
    </div>
  )
}
