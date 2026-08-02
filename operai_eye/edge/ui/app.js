const state = {
  config: {
    burst_size: 5,
    capture_spacing_seconds: 1,
    interval_seconds: 60,
    camera_width: 1920,
    camera_height: 1080,
  },
  stream: null,
  running: false,
  busy: false,
  nextBurstAt: null,
  cycle: 0,
  timer: null,
}

const $ = (selector) => document.querySelector(selector)
const video = $('#cameraVideo')
const canvas = $('#captureCanvas')
const cameraButton = $('#cameraButton')
const cameraCard = $('#cameraCard')
const toast = $('#toast')

const phaseInfo = {
  IDLE: { className: 'idle', icon: '○', label: 'IDLE' },
  PATIENT_IN_ROOM: { className: 'patient', icon: '●', label: 'PATIENT IN ROOM' },
  SURGERY_ACTIVE: { className: 'active', icon: '◆', label: 'SURGERY ACTIVE' },
  UNKNOWN: { className: 'unknown', icon: '?', label: 'UNKNOWN' },
}

function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

function percentage(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—'
}

function showToast(message) {
  toast.textContent = message
  toast.classList.add('show')
  window.clearTimeout(showToast.timeout)
  showToast.timeout = window.setTimeout(() => toast.classList.remove('show'), 6500)
}

function renderEmptyFrames() {
  $('#frames').innerHTML = Array.from({ length: state.config.burst_size }, (_, index) => `
    <div class="frame" data-index="${index}">
      <span class="frame-placeholder">+</span>
      <span class="frame-number">${index + 1}</span>
    </div>
  `).join('')
  $('#frameCount').textContent = `0 / ${state.config.burst_size}`
}

function setConnection(online) {
  const connection = $('#connectionStatus')
  connection.classList.toggle('online', online)
  connection.lastElementChild.textContent = online ? 'Camera online' : 'Camera offline'
}

function updateCountdown() {
  if (!state.running || !state.nextBurstAt) {
    $('#countdown').textContent = '—'
    return
  }
  const remaining = Math.max(0, state.nextBurstAt - Date.now())
  const seconds = Math.ceil(remaining / 1000)
  $('#countdown').textContent = state.busy ? 'Observing' : `00:${String(seconds).padStart(2, '0')}`
}

function captureFrame() {
  const sourceWidth = video.videoWidth
  const sourceHeight = video.videoHeight
  if (!sourceWidth || !sourceHeight) throw new Error('The camera has not produced a frame yet.')
  const scale = Math.min(1, 1280 / sourceWidth, 720 / sourceHeight)
  canvas.width = Math.round(sourceWidth * scale)
  canvas.height = Math.round(sourceHeight * scale)
  const context = canvas.getContext('2d', { alpha: false })
  context.drawImage(video, 0, 0, canvas.width, canvas.height)
  return canvas.toDataURL('image/jpeg', 0.86)
}

function showCapturedFrame(index, dataUrl) {
  const frame = document.querySelector(`.frame[data-index="${index}"]`)
  frame.classList.add('captured', 'current')
  frame.innerHTML = `<img src="${dataUrl}" alt="Captured frame ${index + 1}"><span class="frame-number">${index + 1}</span>`
  document.querySelectorAll('.frame').forEach((item) => {
    if (item !== frame) item.classList.remove('current')
  })
  $('#frameCount').textContent = `${index + 1} / ${state.config.burst_size}`
}

function renderPredictions(images, predictions) {
  predictions.forEach((prediction, index) => {
    const frame = document.querySelector(`.frame[data-index="${index}"]`)
    const label = phaseInfo[prediction.phase]?.label || prediction.phase
    frame.classList.remove('current')
    frame.innerHTML = `
      <img src="${images[index]}" alt="Classified frame ${index + 1}">
      <span class="frame-number">${index + 1}</span>
      <span class="frame-prediction">${label} · ${percentage(prediction.confidence)}</span>
    `
  })
}

function renderVote(result) {
  const vote = result.vote
  const info = phaseInfo[vote.phase] || phaseInfo.UNKNOWN
  const resultHero = $('#resultHero')
  resultHero.className = `result-hero ${info.className}`
  $('#resultIcon').textContent = info.icon
  $('#resultStatus').textContent = vote.uncertain_reason ? 'Uncertain observation' : 'Five-frame majority result'
  $('#resultPhase').textContent = info.label
  $('#winnerConfidence').textContent = percentage(vote.confidence)
  $('#voteFraction').textContent = percentage(vote.vote_fraction)
  $('#overlayLabel').textContent = 'Latest majority result'
  $('#overlayPhase').textContent = info.label
  $('#overlayPhase').style.color = `var(--${info.className === 'patient' ? 'lime' : info.className === 'active' ? 'orange' : info.className === 'idle' ? 'blue' : 'purple'})`
  $('#overlayConfidence').textContent = percentage(vote.confidence)
  $('#inferenceTime').textContent = `${result.inference_ms.toFixed(0)} ms`
  $('#lastUpdated').textContent = new Date(result.completed_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  $('#cycleNumber').textContent = `#${String(state.cycle).padStart(3, '0')}`

  document.querySelectorAll('.vote-row').forEach((row) => {
    const phase = row.dataset.phase
    const count = vote.votes[phase] || 0
    row.querySelector('.vote-label strong').textContent = count
    row.querySelector('.vote-track i').style.width = `${(count / state.config.burst_size) * 100}%`
  })
}

async function runBurst() {
  if (!state.running || state.busy) return
  state.busy = true
  state.cycle += 1
  const cycleStarted = Date.now()
  state.nextBurstAt = cycleStarted + state.config.interval_seconds * 1000
  renderEmptyFrames()
  cameraCard.classList.add('capturing')
  $('#overlayLabel').textContent = 'Capturing observation'
  $('#overlayPhase').textContent = 'FRAME 1 OF 5'
  $('#overlayPhase').style.color = 'var(--text)'
  $('#overlayConfidence').textContent = '—'
  $('#resultStatus').textContent = 'Collecting five frames'
  $('#resultPhase').textContent = 'OBSERVING'
  $('#resultHero').className = 'result-hero neutral'

  try {
    const images = []
    for (let index = 0; index < state.config.burst_size; index += 1) {
      if (!state.running) return
      $('#overlayPhase').textContent = `FRAME ${index + 1} OF ${state.config.burst_size}`
      const dataUrl = captureFrame()
      images.push(dataUrl)
      showCapturedFrame(index, dataUrl)
      if (index < state.config.burst_size - 1) {
        await sleep(state.config.capture_spacing_seconds * 1000)
      }
    }

    cameraCard.classList.remove('capturing')
    $('#overlayLabel').textContent = 'Running DINOv3 batch'
    $('#overlayPhase').textContent = 'ANALYZING'
    const response = await fetch('/api/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ images }),
    })
    const result = await response.json()
    if (!response.ok) throw new Error(result.detail || `Inference request failed (${response.status})`)
    renderPredictions(images, result.predictions)
    renderVote(result)
  } catch (error) {
    showToast(error.message || String(error))
    $('#overlayLabel').textContent = 'Observation failed'
    $('#overlayPhase').textContent = 'CHECK LOGS'
    $('#resultStatus').textContent = 'Inference error'
    $('#resultPhase').textContent = 'NO RESULT'
  } finally {
    cameraCard.classList.remove('capturing')
    state.busy = false
    if (state.running) {
      const delay = Math.max(0, state.nextBurstAt - Date.now())
      state.timer = window.setTimeout(runBurst, delay)
    }
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error('Camera access requires localhost or a secure HTTPS connection.')
  }
  state.stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: state.config.camera_width },
      height: { ideal: state.config.camera_height },
      facingMode: 'environment',
    },
    audio: false,
  })
  video.srcObject = state.stream
  await video.play()
  state.running = true
  cameraCard.classList.add('active')
  cameraButton.classList.add('active')
  cameraButton.lastElementChild.textContent = 'Stop camera'
  cameraButton.firstElementChild.textContent = '■'
  setConnection(true)
  $('#resolutionChip').textContent = `${video.videoWidth} × ${video.videoHeight}`
  window.clearInterval(state.countdownTimer)
  state.countdownTimer = window.setInterval(updateCountdown, 250)
  runBurst()
}

function stopCamera() {
  state.running = false
  state.busy = false
  window.clearTimeout(state.timer)
  window.clearInterval(state.countdownTimer)
  state.stream?.getTracks().forEach((track) => track.stop())
  state.stream = null
  video.srcObject = null
  cameraCard.classList.remove('active', 'capturing')
  cameraButton.classList.remove('active')
  cameraButton.lastElementChild.textContent = 'Enable camera'
  cameraButton.firstElementChild.textContent = '●'
  setConnection(false)
  state.nextBurstAt = null
  updateCountdown()
}

cameraButton.addEventListener('click', async () => {
  if (state.running) {
    stopCamera()
    return
  }
  cameraButton.disabled = true
  try {
    await startCamera()
  } catch (error) {
    stopCamera()
    showToast(`${error.message || error} Check the browser's Camera permission and try again.`)
  } finally {
    cameraButton.disabled = false
  }
})

async function initialize() {
  try {
    const response = await fetch('/api/config')
    if (!response.ok) throw new Error(`Configuration request failed (${response.status})`)
    state.config = await response.json()
  } catch (error) {
    showToast(error.message || String(error))
  }
  renderEmptyFrames()
}

initialize()
