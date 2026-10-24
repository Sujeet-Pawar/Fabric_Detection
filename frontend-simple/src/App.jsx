import { useCallback, useEffect, useRef, useState } from 'react'
import './styles.css'

const API = 'http://127.0.0.1:5001'

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [backendUp, setBackendUp] = useState(false)
  const [modelInfo, setModelInfo] = useState(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    fetch(API + '/api/health')
      .then((r) => r.json())
      .then((d) => {
        setBackendUp(true)
        setModelInfo(d)
      })
    .catch(() => setBackendUp(false))
  }, [])

  const pickFile = useCallback((f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) {
      setError('That file is not an image.')
      return
    }
    setError(null)
    setResult(null)
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }, [])

  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    pickFile(e.dataTransfer.files && e.dataTransfer.files[0])
  }

  const identify = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const body = new FormData()
      body.append('file', file)
      const res = await fetch(API + '/api/predict', { method: 'POST', body })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Prediction failed')
      setResult(data)
    } catch (err) {
      const msg = String((err && err.message) || err)
      setError(
        msg.includes('Failed to fetch')
          ? 'Cannot reach the backend. Is simple-backend/server.py running on port 5001?'
          : msg
      )
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    if (preview) URL.revokeObjectURL(preview)
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  const pct = result ? Math.round(result.confidence * 100) : 0
  // The server flags low confidence; the 50% fallback covers an older backend without the flag.
  const unsure = result && (result.unsure || pct < 50)
  // The model predicts a dedicated "not a fabric" class (or the photo sits far
  // from every trained cluster) -> reject instead of a confident wrong answer.
  const unknown = result && (result.is_unknown || !!result.unknown)
  // When rejected we still show the closest trained fabric, not the raw class name.
  const displayFabric = !result ? '' : (unknown && result.best_guess ? result.best_guess.fabric : result.fabric)

  // Which model won for THIS photo (higher confidence wins). The server always
  // sends better_model; the CNN-fallback covers a backend older than the RF upgrade.
  const winner = result ? result.better_model || 'cnn' : null
  const winnerLabel =
    winner === 'randomforest'
      ? 'Random Forest'
      : result && result.rf ? 'CNN (ResNet18)' : 'CNN (ResNet18)'
  const overallWinner = modelInfo && modelInfo.better_model
    ? (modelInfo.better_model === 'randomforest' ? 'Random Forest' : 'CNN (ResNet18)')
    : null
  // Count only the TRAINED FABRICS (the 'unknown' rejection class is not a fabric).
  const unknownName = modelInfo && modelInfo.unknown_class ? modelInfo.unknown_class : 'unknown'
  const trained = modelInfo && Array.isArray(modelInfo.fabric_classes)
    ? modelInfo.fabric_classes.length
    : (modelInfo && Array.isArray(modelInfo.classes)
      ? modelInfo.classes.filter((c) => c !== unknownName).length
      : 6)
  const quality = result && result.quality ? result.quality : null
  const care = result && result.care && result.care.tips && result.care.tips.length ? result.care : null
  const qualityTone = quality
    ? (quality.score >= 85 ? 'good' : quality.score >= 55 ? 'fair' : 'poor')
    : ''

  return (
    <div className="shell">
      <header>
        <h1>
          FABRIC<span>IQ</span>
        </h1>
        <p>Photograph a fabric. Get the answer, the confidence, and the alternatives.</p>
        <div className={'dot ' + (backendUp ? 'up' : 'down')}>
          {backendUp ? 'backend online' : 'backend offline — start simple-backend/server.py'}
        </div>
      </header>

      <main>
        {!preview ? (
          <div
            className={'dropzone' + (dragging ? ' drag' : '')}
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current && inputRef.current.click()}
          >
            <div className="drop-icon">🧵</div>
            <p className="drop-title">Drop a fabric photo here</p>
            <p className="drop-sub">or click to browse — JPG, PNG, WEBP</p>
            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={(e) => pickFile(e.target.files && e.target.files[0])}
            />
          </div>
        ) : (
          <div className="review">
            <img src={preview} alt="fabric preview" />
            <div className="actions">
              <button className="primary" onClick={identify} disabled={loading}>
                {loading ? 'Identifying…' : 'Identify fabric'}
              </button>
              <button className="ghost" onClick={reset} disabled={loading}>
                Choose another
              </button>
            </div>
          </div>
        )}

        {error && <div className="error">{error}</div>}

        {result && (
          <section className="result">
            {result.rf && (
              <div className="model-verdict">
                🏆 Higher confidence on this photo: <b>{winnerLabel}</b>
                <span className="model-scores">
                  CNN {Math.round((result.cnn ? result.cnn.confidence : result.confidence) * 100)}%
                  {' · '}RF {Math.round(result.rf.confidence * 100)}%
                </span>
              </div>
            )}
            {unknown ? (
              <div className="notsure unknown">
                <div className="notsure-title">Not one of the trained fabrics 🚫</div>
                <p>
                  This photo looks unlike any of the {trained} trained fabric types, so the guess below is
                  unreliable. To teach the app this fabric, add photos of it to the dataset folder
                  and retrain (Run All in the notebook).
                </p>
              </div>
            ) : unsure ? (
              <div className="notsure">
                <div className="notsure-title">Not sure 🤔</div>
                <p>
                  The photo may be unclear, or this fabric is not one of the {trained} trained classes.
                  Best guess below — for a confident answer, photograph the fabric in flat daylight,
                  filling the frame with the weave.
                </p>
              </div>
            ) : result.similar ? (
              <div className="confident">Likely match</div>
            ) : (
              <div className="confident">Confident match</div>
            )}
            {result.similar && result.similar.length === 2 && (
              <div className="similar-note">
                ⚠ These are near-identical smooth weaves — the photo can't reliably separate{' '}
                <b>{result.similar[0]}</b> from <b>{result.similar[1]}</b>.
              </div>
            )}
            <div className="answer">
              <div className="fabric">{displayFabric}</div>
              <div className="conf-label">confidence</div>
              <div className="meter">
                <div className="fill" style={{ width: pct + '%' }} />
              </div>
              <div className="conf-num">{pct}%</div>
              {unknown && result.best_guess && (
                <p className="guess-note">
                  Closest trained fabric: <b>{result.best_guess.fabric}</b>{' '}
                  ({Math.round(result.best_guess.confidence * 100)}%)
                </p>
              )}
            </div>
            {result.alternatives && result.alternatives.length > 0 && (
              <div className="alts">
                <h3>Confused with</h3>
                {result.alternatives.map((a) => (
                  <div className="alt" key={a.fabric}>
                    <span>{a.fabric}</span>
                    <span>{Math.round(a.confidence * 100)}%</span>
                  </div>
                ))}
              </div>
            )}

            {quality && (
              <div className="quality">
                <div className="quality-head">
                  <h3>Fabric condition</h3>
                  <span className={'quality-label ' + qualityTone}>{quality.label}</span>
                </div>
                <div className="qmeter">
                  <div className={'qfill ' + qualityTone} style={{ width: quality.score + '%' }} />
                </div>
                <div className="qnum">
                  <span>{quality.score}/100</span>
                  {quality.photo === 'blurry' && (
                    <span className="qnote">⚠ blurry photo — score is approximate</span>
                  )}
                </div>
                {quality.issues && quality.issues.length > 0 ? (
                  <ul className="qissues">
                    {quality.issues.map((issue) => (
                      <li key={issue}>{issue}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="qnoproblem">No surface problems detected — looks well maintained.</p>
                )}
              </div>
            )}

            {care && (
              <div className="care">
                <h3>How to care for {care.fabric}</h3>
                <ul>
                  {care.tips.map((tip) => (
                    <li key={tip}>{tip}</li>
                  ))}
                </ul>
              </div>
            )}
          </section>
        )}

        {overallWinner && (
          <div className="model-verdict overall">
            Overall accuracy winner: <b>{overallWinner}</b>
            {modelInfo.cnn_test_accuracy != null && modelInfo.rf_test_accuracy != null && (
              <span className="model-scores">
                CNN {Math.round(modelInfo.cnn_test_accuracy * 100)}%
                {' · '}RF {Math.round(modelInfo.rf_test_accuracy * 100)}%
              </span>
            )}
          </div>
        )}
      </main>
      <footer>Trained in the notebook · ml/notebooks/fabric_classifier_simple.ipynb</footer>
    </div>
  )
}
