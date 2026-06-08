import './ModelSelector.css'

function ModelSelector({ models, selectedModels, onToggle }) {
  return (
    <div className="model-selector">
      <h3>Select Models</h3>
      <div className="model-list">
        {models.map(model => (
          <label key={model.model_id} className="model-option">
            <input
              type="checkbox"
              checked={selectedModels.includes(model.model_id)}
              onChange={() => onToggle(model.model_id)}
            />
            <div className="model-info">
              <div className="model-name">{model.model_name}</div>
              <div className="model-meta">
                {model.accuracy !== null && (
                  <span className="accuracy">{(model.accuracy * 100).toFixed(1)}%</span>
                )}
                <span className="count">{model.total_evaluated} images</span>
                <span className="id">{model.model_id}</span>
              </div>
            </div>
          </label>
        ))}
        {models.length === 0 && (
          <div className="no-models">No evaluation results found</div>
        )}
      </div>
    </div>
  )
}

export default ModelSelector
