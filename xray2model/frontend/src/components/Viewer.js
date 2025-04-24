import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, useGLTF, Html } from '@react-three/drei';

// Component to load and display the GLTF model
function Model({ url }) {
  // useGLTF hook handles loading the GLB/GLTF file
  // It supports Suspense for fallback during loading
  const { scene } = useGLTF(url);
  // You might need to scale or position the loaded scene
  return <primitive object={scene} scale={1} />; // Adjust scale as needed
}

// Loading fallback component
function Loader() {
  return <Html center>Loading 3D model...</Html>;
}

// Main Viewer component
function Viewer({ modelUrl }) {
  if (!modelUrl) {
    return <p>No model URL provided.</p>;
  }

  return (
    <div style={{ width: '80vw', height: '60vh', marginTop: '1rem', border: '1px solid #ccc' }}>
      <Canvas camera={{ position: [2, 2, 2], fov: 50 }}>
        {/* Suspense handles the loading state of the model */}
        <Suspense fallback={<Loader />}>
          <ambientLight intensity={0.8} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <directionalLight position={[-10, 10, 5]} intensity={0.5} />
          {/* Render the loaded model */}
          <Model url={modelUrl} />
          <OrbitControls />
        </Suspense>
      </Canvas>
      {/* Optional: Display the blob URL for debugging */}
      {/* <p style={{ fontSize: '0.8em', wordBreak: 'break-all' }}>Model URL: {modelUrl}</p> */}
    </div>
  );
}

export default Viewer;
