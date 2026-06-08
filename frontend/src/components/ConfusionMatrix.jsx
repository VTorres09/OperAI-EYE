import './ConfusionMatrix.css'

function ConfusionMatrix({ matrix, classes }) {
  const maxValue = Math.max(...classes.flatMap(gt => 
    classes.map(pred => matrix[gt][pred])
  ))

  const getCellColor = (value) => {
    const intensity = value / maxValue
    return `rgba(59, 130, 246, ${intensity * 0.8})`
  }

  return (
    <div className="confusion-matrix">
      <h4>Confusion Matrix</h4>
      <div className="matrix-container">
        <table>
          <thead>
            <tr>
              <th></th>
              <th colSpan={classes.length}>Predicted</th>
            </tr>
            <tr>
              <th>Actual</th>
              {classes.map(cls => <th key={cls}>{cls}</th>)}
            </tr>
          </thead>
          <tbody>
            {classes.map(gt => (
              <tr key={gt}>
                <th>{gt}</th>
                {classes.map(pred => {
                  const value = matrix[gt][pred]
                  const isCorrect = gt === pred
                  return (
                    <td
                      key={pred}
                      className={`matrix-cell ${isCorrect ? 'correct' : 'incorrect'}`}
                      style={{ backgroundColor: getCellColor(value) }}
                      title={`${gt} → ${pred}: ${value}`}
                    >
                      {value}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default ConfusionMatrix
