import './MetricsDisplay.css'

function MetricsDisplay({ result }) {
  const { model_info, total, correct, accuracy, per_class } = result

  return (
    <div className="metrics-display">
      <div className="metrics-header">
        <h3>{model_info.model_name}</h3>
        <p className="model-description">{model_info.description}</p>
      </div>

      <div className="metrics-summary">
        <div className="metric-card">
          <div className="metric-value">{(accuracy * 100).toFixed(1)}%</div>
          <div className="metric-label">Accuracy</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">{correct}/{total}</div>
          <div className="metric-label">Correct</div>
        </div>
      </div>

      <div className="per-class-metrics">
        <h4>Per-Class Metrics</h4>
        <table>
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(per_class).map(([cls, metrics]) => (
              <tr key={cls}>
                <td>{cls}</td>
                <td>{(metrics.precision * 100).toFixed(1)}%</td>
                <td>{(metrics.recall * 100).toFixed(1)}%</td>
                <td>{(metrics.f1 * 100).toFixed(1)}%</td>
                <td>{metrics.support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default MetricsDisplay
