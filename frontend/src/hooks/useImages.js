import { useSyncExternalStore } from 'react'

const PAGE_SIZE = 24

function createImageStore() {
  let state = {
    items: [],
    filters: {},
    filterOptions: null,
    stats: null,
    page: 1,
    total: 0,
    totalPages: 0,
    loading: false,
    hasMore: true,
  }
  let listeners = new Set()
  let abortController = null
  let statsAbortController = null

  function setState(partial) {
    state = { ...state, ...partial }
    listeners.forEach((l) => l())
  }

  function buildParams() {
    const params = new URLSearchParams()
    Object.entries(state.filters).forEach(([key, val]) => {
      if (val !== undefined && val !== null && val !== '') params.set(key, val)
    })
    return params
  }

  async function fetchStats() {
    if (statsAbortController) statsAbortController.abort()
    statsAbortController = new AbortController()

    const params = buildParams()
    try {
      const res = await fetch(`/api/stats?${params}`, { signal: statsAbortController.signal })
      const data = await res.json()
      setState({ stats: data })
    } catch (e) {
      if (e.name !== 'AbortError') console.error(e)
    }
  }

  async function fetchPage(pageNum, append = false) {
    if (abortController) abortController.abort()
    abortController = new AbortController()

    setState({ loading: true })
    const params = buildParams()
    params.set('page', pageNum)
    params.set('page_size', PAGE_SIZE)

    try {
      const res = await fetch(`/api/images?${params}`, { signal: abortController.signal })
      const data = await res.json()
      setState({
        items: append ? [...state.items, ...data.items] : data.items,
        total: data.total,
        totalPages: data.total_pages,
        page: pageNum,
        hasMore: pageNum < data.total_pages,
        loading: false,
      })
    } catch (e) {
      if (e.name !== 'AbortError') {
        console.error(e)
        setState({ loading: false })
      }
    }
  }

  async function loadFilterOptions() {
    const res = await fetch('/api/filters')
    const data = await res.json()
    setState({ filterOptions: data })
  }

  function updateFilters(newFilters) {
    setState({ filters: newFilters })
    fetchPage(1)
    fetchStats()
  }

  function loadMore() {
    if (!state.loading && state.hasMore) {
      fetchPage(state.page + 1, true)
    }
  }

  function getSnapshot() {
    return state
  }

  function subscribe(listener) {
    listeners.add(listener)
    return () => listeners.delete(listener)
  }

  loadFilterOptions()
  fetchPage(1)
  fetchStats()

  return { getSnapshot, subscribe, updateFilters, loadMore }
}

const store = createImageStore()

export function useImages() {
  const state = useSyncExternalStore(store.subscribe, store.getSnapshot)
  return {
    ...state,
    updateFilters: store.updateFilters,
    loadMore: store.loadMore,
  }
}
