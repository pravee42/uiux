import { useState, useEffect } from 'react';
import './App.css';
import UXAnalysisDashboard from '../ui';
import result from "./data/dd.json"

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [analysisData, setAnalysisData] = useState(result);
  const [error, setError] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loadingProgress, setLoadingProgress] = useState(0);

  // Simulate loading progress
  useEffect(() => {
    if (isLoading) {
      const interval = setInterval(() => {
        setLoadingProgress(prev => {
          const newProgress = prev + Math.random() * 15;
          return newProgress >= 95 ? 95 : newProgress;
        });
      }, 600);
      
      return () => {
        clearInterval(interval);
        setLoadingProgress(0);
      };
    }
  }, [isLoading]);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImagePreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    const formData = new FormData(e.target);
    
    try {
      const response = await fetch('http://localhost:8000/analyze-ui', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      // Set progress to 100% when complete
      setLoadingProgress(100);
      
      // Short delay to show complete animation
      setTimeout(() => {
        setAnalysisData(data);
        setIsLoading(false);
      }, 500);
    } catch (err) {
      console.error('Error during analysis:', err);
      setError('Failed to analyze the UI. Please try again.');
      setIsLoading(false);
    }
  };

  const resetForm = () => {
    setAnalysisData(null);
    setImagePreview(null);
    setError(null);
  };

  return (
    <div>
      {!analysisData &&!isLoading && (
        <div className="upload-container">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="image">Upload UI Screenshot</label>
              <input 
                type="file" 
                id="image" 
                name="image" 
                accept="image/*" 
                onChange={handleImageChange}
                required 
              />
              
              {imagePreview && (
                <div className="image-preview">
                  <img src={imagePreview} alt="Preview" />
                </div>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="purpose">Purpose</label>
              <input 
                type="text" 
                id="purpose" 
                name="purpose" 
                placeholder="e.g., Parking Management Dashboard" 
                required 
              />
            </div>

            <div className="form-group">
              <label htmlFor="screen">Screen Type</label>
              <select id="screen" name="screen" required>
                <option value="Desktop">Desktop</option>
                <option value="Mobile">Mobile</option>
                <option value="Tablet">Tablet</option>
              </select>
            </div>

            <button type="submit" className="submit-btn">Analyze UI</button>
          </form>
        </div>
      )}

      {isLoading && (
        <div className="loading-container">
          <div className="fancy-loader">
            <div className="loader-content">
              <div className="wireframe-grid">
                <div className="wireframe-box"></div>
                <div className="wireframe-box"></div>
                <div className="wireframe-line"></div>
                <div className="wireframe-box"></div>
                <div className="wireframe-circle"></div>
              </div>
              
              <div className="analysis-beam"></div>
              
              <div className="progress-container">
                <div className="progress-bar" style={{ width: `${loadingProgress}%` }}></div>
                <div className="progress-text">{Math.round(loadingProgress)}%</div>
              </div>
              
              <div className="loading-message">
                <p className="loading-title">Analyzing UI Design</p>
                <p className="loading-subtitle">
                  {loadingProgress < 20 && "Processing image..."}
                  {loadingProgress >= 20 && loadingProgress < 40 && "Identifying UI elements..."}
                  {loadingProgress >= 40 && loadingProgress < 60 && "Evaluating layout structure..."}
                  {loadingProgress >= 60 && loadingProgress < 80 && "Assessing accessibility..."}
                  {loadingProgress >= 80 && "Generating recommendations..."}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="error-message">
          <p>{error}</p>
          <button onClick={resetForm}>Try Again</button>
        </div>
      )}

      {analysisData && !isLoading && (
        <div className="results-container">
          <UXAnalysisDashboard resultData={analysisData} />
          <button onClick={resetForm} className="reset-btn mb-4">Analyze Another UI</button>
        </div>
      )}
    </div>
  );
}

export default App;