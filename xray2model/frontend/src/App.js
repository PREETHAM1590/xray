import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Viewer from './components/Viewer';

function App() {
  const [file, setFile] = useState(null);
  const [modelUrl, setModelUrl] = useState(''); // Store the Blob URL for the model
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Clean up Blob URL when component unmounts or modelUrl changes
  useEffect(() => {
    return () => {
      if (modelUrl) {
        URL.revokeObjectURL(modelUrl);
      }
    };
  }, [modelUrl]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setModelUrl(''); // Clear previous model when new file is selected
    setError(''); // Clear previous errors
  };

  const handleReconstruct = async () => {
    if (!file) {
      setError('Please select an image file first.');
      return;
    }
    setError('');
    setLoading(true);
    setModelUrl(''); // Clear previous model

    // Clean up any existing blob URL before creating a new one
    if (modelUrl) {
      URL.revokeObjectURL(modelUrl);
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Send file to backend /reconstruct endpoint
      const response = await axios.post('http://localhost:5000/reconstruct', formData, {
        responseType: 'blob', // Important: expect binary data (the GLB file)
      });

      // Create a Blob URL from the response data
      const blob = new Blob([response.data], { type: 'model/gltf-binary' });
      const url = URL.createObjectURL(blob);
      setModelUrl(url);

    } catch (err) {
      console.error("Reconstruction failed:", err);
      let errorMsg = 'Reconstruction failed. Check server logs.';
      if (err.response && err.response.data) {
        // Try to read error message if backend sent JSON error within the blob response handling
        try {
            // Read the blob as text to see if it contains a JSON error message
            const errorJson = JSON.parse(await err.response.data.text());
            if (errorJson && errorJson.error) {
                errorMsg = `Reconstruction failed: ${errorJson.error}`;
            }
        } catch (parseError) {
            // Blob wasn't JSON, stick to generic error
            console.error("Could not parse error response:", parseError);
        }
      }
      setError(errorMsg);
      setModelUrl(''); // Ensure no model URL is set on error
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 32, fontFamily: 'sans-serif' }}>
      <h1>2D X-ray to 3D Reconstruction</h1>
      <div style={{ marginBottom: '1rem' }}>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        <button onClick={handleReconstruct} disabled={!file || loading} style={{ marginLeft: '1rem' }}>
          {loading ? 'Processing...' : 'Upload & Reconstruct 3D'}
        </button>
      </div>
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {/* Pass the Blob URL to the Viewer */}
      {modelUrl && !error && <Viewer modelUrl={modelUrl} />}
      {!modelUrl && !loading && !error && <p>Select an X-ray image and click "Upload & Reconstruct 3D".</p>}
    </div>
  );
}

export default App;
