import os
import io
import numpy as np
import torch
import trimesh
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from skimage.measure import marching_cubes

# Use relative import since 'models' is a sibling directory to 'backend'
from ..models.unet_reconstruction import ReconstructionUNet

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Configuration ---
# Define expected input size for the model (adjust as needed)
# These might match the training data dimensions
TARGET_IMG_SIZE = (256, 256)
TARGET_VOXEL_DEPTH = 64 # Example depth for the 3D volume

# Define the isovalue for Marching Cubes (needs tuning based on model output)
MARCHING_CUBES_ISOVALUE = 0.5

# Define mesh simplification target (e.g., target number of faces)
# Set to None to disable simplification
MESH_SIMPLIFICATION_TARGET_FACES = 50000

# --- Model Loading ---
# In a real application, load trained weights here.
# For now, instantiate the model directly.
# Ensure parameters match the expected input/output dimensions after pre-processing.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model (adjust parameters as needed)
# The input channels should match the pre-processed input tensor's channel count.
# The spatial_dims should be 3 for 3D output.
reconstruction_model = ReconstructionUNet(
    spatial_dims=3,
    in_channels=1, # After pre-processing, we'll likely have 1 channel
    out_channels=1,
    channels=(16, 32, 64, 128), # Example channels
    strides=(2, 2, 2),
    num_res_units=1
).to(device)
reconstruction_model.eval() # Set model to evaluation mode

# --- Helper Functions ---
def allowed_file(filename):
    """Checks if the filename has an allowed extension."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocesses the input PIL image for the model."""
    # 1. Convert to grayscale
    img = image.convert('L')
    # 2. Resize
    img = img.resize(TARGET_IMG_SIZE, Image.Resampling.LANCZOS)
    # 3. Convert to numpy array and normalize
    img_np = np.array(img, dtype=np.float32) / 255.0
    # 4. Convert to PyTorch tensor -> [H, W]
    img_tensor = torch.from_numpy(img_np)

    # --- Adapt 2D input for 3D UNet ---
    # This is a placeholder strategy and needs refinement based on the actual model training.
    # Strategy: Unsqueeze to [C, H, W], then repeat across depth D, then add batch dim.
    # Resulting shape: [1, 1, D, H, W]
    img_tensor = img_tensor.unsqueeze(0) # Add channel dim -> [1, H, W]
    # Repeat the slice across the depth dimension
    img_tensor_3d = img_tensor.unsqueeze(1).repeat(1, TARGET_VOXEL_DEPTH, 1, 1) # -> [1, D, H, W]
    # Add batch dimension
    img_tensor_batch = img_tensor_3d.unsqueeze(0) # -> [1, 1, D, H, W]

    return img_tensor_batch.to(device)

def convert_voxels_to_glb(voxel_grid: np.ndarray, isovalue: float, simplify_target: int = None) -> bytes:
    """Converts a voxel grid to a GLB byte stream using Marching Cubes."""
    if voxel_grid.ndim != 3:
        raise ValueError(f"Voxel grid must be 3D, but got shape {voxel_grid.shape}")

    print(f"Running Marching Cubes with isovalue={isovalue}...")
    try:
        # Ensure voxel grid is float64 for marching_cubes
        voxel_grid_float = voxel_grid.astype(np.float64)
        verts, faces, normals, values = marching_cubes(
            volume=voxel_grid_float,
            level=isovalue,
            spacing=(1.0, 1.0, 1.0) # Adjust spacing if needed
        )
    except Exception as e:
        print(f"Marching Cubes failed: {e}")
        # Check if it's due to flat volume
        if np.all(voxel_grid > isovalue) or np.all(voxel_grid < isovalue):
             print("Marching Cubes error likely due to the volume being entirely above or below the isovalue.")
        raise # Re-raise the exception

    if len(verts) == 0 or len(faces) == 0:
         raise ValueError(f"Marching Cubes resulted in an empty mesh (0 vertices or faces) for isovalue {isovalue}. Try adjusting the isovalue.")

    print(f"Generated mesh with {len(verts)} vertices and {len(faces)} faces.")

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # Optional: Simplify mesh
    if simplify_target is not None:
        current_faces = mesh.faces.shape[0]
        # Only simplify if the current face count is greater than the target
        # and the target is greater than 0.
        if current_faces > simplify_target and simplify_target > 0:
            # Calculate the required reduction factor (proportion of faces to remove)
            target_reduction = 1.0 - (simplify_target / float(current_faces))
            # Ensure target_reduction is strictly between 0.0 and 1.0 as required by some versions/backends
            target_reduction = max(1e-6, min(1.0 - 1e-6, target_reduction)) 

            print(f"Simplifying mesh from {current_faces} faces to approximately {simplify_target} faces...")
            try:
                # Pass the target face count directly
                mesh = mesh.simplify_quadric_decimation(face_count=simplify_target)
                print(f"Simplified mesh has {mesh.vertices.shape[0]} vertices and {mesh.faces.shape[0]} faces.")
            except Exception as simplify_error:
                print(f"Error during mesh simplification: {simplify_error}")
                # Optionally, decide whether to proceed with the unsimplified mesh or raise an error
                # For now, we'll proceed with the unsimplified mesh if simplification fails
                print("Proceeding with unsimplified mesh.")
        elif current_faces <= simplify_target:
            print(f"Mesh already has {current_faces} faces, which is less than or equal to the target {simplify_target}. No simplification needed.")
        else: # simplify_target <= 0
             print(f"Invalid simplify_target ({simplify_target}). Must be greater than 0. Skipping simplification.")

    # Optional: Smooth mesh (example using Taubin smoothing)
    # print("Smoothing mesh...")
    # trimesh.smoothing.filter_taubin(mesh, iterations=10)

    print("Exporting mesh to GLB format...")
    # Export to GLB format in memory
    with io.BytesIO() as f:
        mesh.export(f, file_type='glb')
        f.seek(0)
        glb_data = f.read()
    print("GLB export complete.")
    return glb_data

# --- API Endpoints ---
@app.route('/', methods=['GET'])
def index():
    """Basic route to confirm server is running."""
    return jsonify({'status': 'XRay2Model Backend Server is running. Use the /reconstruct endpoint.'})

@app.route('/reconstruct', methods=['POST'])
def reconstruct_xray():
    """
    Receives an X-ray image, runs reconstruction, and returns a 3D model in GLB format.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        try:
            print(f"Received file: {file.filename}")
            # Read image
            img = Image.open(file.stream)

            # Preprocess image for the model
            print("Pre-processing image...")
            input_tensor = preprocess_image(img)
            print(f"Input tensor shape: {input_tensor.shape}")

            # Perform inference
            print("Running model inference...")
            with torch.no_grad():
                output_tensor = reconstruction_model(input_tensor)
            print(f"Output tensor shape: {output_tensor.shape}") # Should be [1, 1, D, H, W]

            # Post-process output tensor
            # Remove batch and channel dimensions, move to CPU, convert to numpy
            voxel_output = output_tensor.squeeze().cpu().numpy()
            print(f"Output voxel grid shape: {voxel_output.shape}") # Should be [D, H, W]

            # Convert voxel grid to GLB mesh
            glb_data = convert_voxels_to_glb(
                voxel_output,
                isovalue=MARCHING_CUBES_ISOVALUE,
                simplify_target=MESH_SIMPLIFICATION_TARGET_FACES
            )

            # Return the GLB data
            print("Sending GLB data...")
            return send_file(
                io.BytesIO(glb_data),
                mimetype='model/gltf-binary',
                as_attachment=True,
                download_name=f'{os.path.splitext(file.filename)[0]}_reconstruction.glb'
            )

        except ValueError as ve:
             print(f"Value Error during reconstruction: {ve}")
             return jsonify({'error': str(ve)}), 400
        except Exception as e:
            print(f"An error occurred during reconstruction: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback to server logs
            return jsonify({'error': 'An internal error occurred during reconstruction.'}), 500
    else:
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg, bmp, gif, tiff'}), 400

# --- Main Execution ---
if __name__ == '__main__':
    # Note: Running with debug=True is not recommended for production
    app.run(debug=True, host='0.0.0.0', port=5000) # Use port 5000 by default
