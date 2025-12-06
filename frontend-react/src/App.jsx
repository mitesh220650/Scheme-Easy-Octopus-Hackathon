import { useState, useRef, useCallback } from 'react'

const API_BASE = '/api'

function App() {
  const [isRecording, setIsRecording] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')
  const [showSources, setShowSources] = useState(false)
  const [language, setLanguage] = useState('auto')
  
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        stream.getTracks().forEach(track => track.stop())
        await transcribeAudio(audioBlob)
      }

      mediaRecorder.start()
      setIsRecording(true)
      setError('')
    } catch (err) {
      setError('Could not access microphone. Please allow microphone access.')
      console.error('Recording error:', err)
    }
  }, [])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }, [isRecording])

  const transcribeAudio = async (audioBlob) => {
    setIsLoading(true)
    setError('')
    
    try {
      const formData = new FormData()
      formData.append('audio', audioBlob, 'recording.webm')

      const response = await fetch(`${API_BASE}/transcribe`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Transcription failed')
      }

      const data = await response.json()
      setTranscript(data.transcript)
    } catch (err) {
      setError('Failed to transcribe audio. Please try again or type your question.')
      console.error('Transcription error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const querySchemes = async () => {
    if (!transcript.trim()) {
      setError('Please record or type your situation first.')
      return
    }

    setIsLoading(true)
    setError('')
    setResults(null)

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: transcript,
          top_k: 5,
          language: language
        })
      })

      if (!response.ok) {
        throw new Error('Query failed')
      }

      const data = await response.json()
      setResults(data)
    } catch (err) {
      setError('Failed to find schemes. Please check if the backend is running.')
      console.error('Query error:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const speakText = (text) => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 0.9
      utterance.pitch = 1
      window.speechSynthesis.speak(utterance)
    }
  }

  const reset = () => {
    setTranscript('')
    setResults(null)
    setError('')
    setShowSources(false)
    window.speechSynthesis?.cancel()
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <header className="bg-blue-600 text-white py-6 px-4 shadow-lg">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl md:text-4xl font-bold text-center">Scheme-Easy</h1>
          <p className="text-center text-blue-100 mt-2 text-lg">Find government schemes you qualify for</p>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-4 py-8">
        <div className="mb-6 flex justify-center gap-4 items-center">
          <label className="text-gray-700 font-medium">Language:</label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className="px-4 py-2 rounded-lg border border-gray-300 bg-white text-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="auto">Auto-detect</option>
            <option value="en">English</option>
            <option value="hi">Hindi</option>
            <option value="ta">Tamil</option>
            <option value="te">Telugu</option>
            <option value="bn">Bengali</option>
            <option value="mr">Marathi</option>
            <option value="gu">Gujarati</option>
            <option value="kn">Kannada</option>
          </select>
        </div>

        <div className="bg-white rounded-2xl shadow-xl p-8 mb-8">
          <div className="flex flex-col items-center mb-8">
            <div className="relative">
              {isRecording && (
                <div className="absolute inset-0 bg-red-400 rounded-full recording-pulse"></div>
              )}
              <button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isLoading}
                className={`relative w-32 h-32 rounded-full flex items-center justify-center text-white text-xl font-bold transition-all transform hover:scale-105 ${
                  isRecording 
                    ? 'bg-red-500 hover:bg-red-600' 
                    : 'bg-blue-500 hover:bg-blue-600'
                } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {isRecording ? (
                  <span className="flex flex-col items-center">
                    <svg className="w-10 h-10 mb-1" fill="currentColor" viewBox="0 0 24 24">
                      <rect x="6" y="6" width="12" height="12" rx="2" />
                    </svg>
                    STOP
                  </span>
                ) : (
                  <span className="flex flex-col items-center">
                    <svg className="w-10 h-10 mb-1" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                      <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                    </svg>
                    RECORD
                  </span>
                )}
              </button>
            </div>
            <p className="mt-4 text-gray-600 text-lg">
              {isRecording ? 'Recording... Click to stop' : 'Click to speak your situation'}
            </p>
          </div>

          <div className="mb-6">
            <label className="block text-gray-700 font-semibold mb-2 text-lg">
              Your Situation:
            </label>
            <textarea
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              placeholder="Speak or type your situation here. For example: 'I am a farmer, my crop was destroyed by rain. I need help.'"
              className="w-full h-32 px-4 py-3 text-lg border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none"
            />
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-700 text-lg">
              {error}
            </div>
          )}

          <div className="flex gap-4 justify-center">
            <button
              onClick={querySchemes}
              disabled={isLoading || !transcript.trim()}
              className={`px-8 py-4 bg-green-500 text-white text-xl font-bold rounded-xl transition-all transform hover:scale-105 hover:bg-green-600 ${
                (isLoading || !transcript.trim()) ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {isLoading ? 'Searching...' : 'Check My Schemes'}
            </button>
            <button
              onClick={reset}
              className="px-8 py-4 bg-gray-200 text-gray-700 text-xl font-bold rounded-xl transition-all hover:bg-gray-300"
            >
              Start Over
            </button>
          </div>
        </div>

        {isLoading && (
          <div className="flex justify-center py-8">
            <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent"></div>
          </div>
        )}

        {results && (
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="mb-6 p-6 bg-blue-50 rounded-xl">
              <h2 className="text-2xl font-bold text-blue-800 mb-3">Summary</h2>
              <p className="text-xl text-gray-700">{results.summary}</p>
              <button
                onClick={() => speakText(results.summary)}
                className="mt-3 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 flex items-center gap-2"
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/>
                </svg>
                Listen
              </button>
            </div>

            {results.schemes && results.schemes.length > 0 && (
              <div className="space-y-6">
                <h2 className="text-2xl font-bold text-gray-800">Schemes You May Qualify For</h2>
                
                {results.schemes.map((scheme, index) => (
                  <div key={index} className="border-2 border-gray-100 rounded-xl p-6 hover:shadow-lg transition-shadow">
                    <div className="flex justify-between items-start mb-4">
                      <h3 className="text-xl font-bold text-green-700">{scheme.title}</h3>
                      {scheme.confidence && (
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                          scheme.confidence > 0.7 ? 'bg-green-100 text-green-700' :
                          scheme.confidence > 0.4 ? 'bg-yellow-100 text-yellow-700' :
                          'bg-gray-100 text-gray-700'
                        }`}>
                          {Math.round(scheme.confidence * 100)}% match
                        </span>
                      )}
                    </div>

                    {scheme.why && (
                      <p className="text-gray-600 mb-4 italic">"{scheme.why}"</p>
                    )}

                    {scheme.documents && scheme.documents.length > 0 && (
                      <div className="mb-4">
                        <h4 className="font-semibold text-gray-700 mb-2 text-lg">Documents Needed:</h4>
                        <ul className="list-disc list-inside space-y-1">
                          {scheme.documents.map((doc, i) => (
                            <li key={i} className="text-gray-600 text-lg">{doc}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {scheme.next_steps && scheme.next_steps.length > 0 && (
                      <div className="mb-4">
                        <h4 className="font-semibold text-gray-700 mb-2 text-lg">Next Steps:</h4>
                        <ol className="list-decimal list-inside space-y-1">
                          {scheme.next_steps.map((step, i) => (
                            <li key={i} className="text-gray-600 text-lg">{step}</li>
                          ))}
                        </ol>
                      </div>
                    )}

                    <div className="flex gap-3 mt-4">
                      <button
                        onClick={() => speakText(
                          `${scheme.title}. Documents needed: ${scheme.documents?.join(', ')}. Next steps: ${scheme.next_steps?.join('. ')}`
                        )}
                        className="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 flex items-center gap-2"
                      >
                        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z"/>
                        </svg>
                        Play Instructions
                      </button>
                      {scheme.source_file && (
                        <span className="px-3 py-2 text-sm text-gray-500">
                          Source: {scheme.source_file}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {results.follow_up_questions && results.follow_up_questions.length > 0 && (
              <div className="mt-6 p-6 bg-yellow-50 rounded-xl">
                <h3 className="text-xl font-bold text-yellow-800 mb-3">We may need more information:</h3>
                <ul className="space-y-2">
                  {results.follow_up_questions.map((question, i) => (
                    <li key={i} className="text-lg text-gray-700">• {question}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="mt-6">
              <button
                onClick={() => setShowSources(!showSources)}
                className="text-blue-600 hover:text-blue-800 font-medium"
              >
                {showSources ? 'Hide Sources' : 'Show Sources'}
              </button>
              
              {showSources && results.raw_retrieved && (
                <div className="mt-4 p-4 bg-gray-50 rounded-xl">
                  <h4 className="font-semibold text-gray-700 mb-3">Retrieved Documents:</h4>
                  <div className="space-y-3">
                    {results.raw_retrieved.map((doc, i) => (
                      <div key={i} className="p-3 bg-white rounded-lg border">
                        <div className="flex justify-between text-sm text-gray-500 mb-1">
                          <span>{doc.file_name}</span>
                          <span>Similarity: {(doc.similarity * 100).toFixed(1)}%</span>
                        </div>
                        <p className="text-gray-600 text-sm">{doc.chunk_text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <footer className="bg-gray-100 py-6 mt-12">
        <div className="max-w-4xl mx-auto px-4 text-center text-gray-600">
          <p>Scheme-Easy helps you find government welfare schemes you may qualify for.</p>
          <p className="text-sm mt-2">This is a prototype. Please verify information with official sources.</p>
        </div>
      </footer>
    </div>
  )
}

export default App
