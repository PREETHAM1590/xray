# Script for generating Digitally Reconstructed Radiographs (DRRs)
# from 3D CT volumes using torchdr for ray-casting simulation.

# TODO:
# - Verify torchdr API calls and parameter conventions (volume shape/spacing order, geometry).
# - Implement more realistic projection geometry (source/detector positions, angles, SID, SDD).
# - Add geometric jitter/variations for robustness during training.
# - Add more sophisticated intensity normalization/windowing to DRR output.
# - Implement CT preprocessing (resampling to isotropic spacing, intensity clipping) before projection.
# - Store projection metadata (angle, geometry params, patient ID) alongside DRRs.
# - Add optional noise/artifact simulation.

import argparse
import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
from tqdm import tqdm # For progress bar
import math
import torch
import torchdr # Make sure torchdr is installed (pip install torchdr)

def simulate_drr_torchdr(
    ct_volume_sitk: sitk.Image, 
    sdd: float, # Source-to-Detector Distance
    sod: float, # Source-to-Origin Distance (origin is center of CT volume)
    detector_height_mm: float, 
    detector_width_mm: float, 
    pixel_spacing_mm: float, 
    angle_deg: float, 
    device: torch.device,
    output_size_px: tuple = None # Optional output pixel dimensions (width, height)
    ) -> np.ndarray:
    """
    Simulates a DRR using torchdr's Siddon projector.

    Args:
        ct_volume_sitk: SimpleITK image object of the CT volume.
        sdd: Source-to-Detector Distance (mm).
        sod: Source-to-Origin Distance (mm).
        detector_height_mm: Height of the detector panel (mm).
        detector_width_mm: Width of the detector panel (mm).
        pixel_spacing_mm: Pixel spacing on the detector (mm). Assumed square.
        angle_deg: Rotation angle around the Z-axis (superior-inferior) in degrees.
        device: PyTorch device ('cpu' or 'cuda').
        output_size_px (tuple, optional): Desired output pixel size (width, height). Resizes if provided.

    Returns:
        A 2D numpy array representing the DRR, normalized to uint8, or None if error.
    """
    try:
        # --- Get CT Volume Info ---
        ct_spacing = ct_volume_sitk.GetSpacing()   # (sx, sy, sz) - Assuming mm
        ct_size = ct_volume_sitk.GetSize()       # (nx, ny, nz) - Number of voxels
        # ct_origin = ct_volume_sitk.GetOrigin() # Physical coord of first voxel center? Not directly used by torchdr projector init?

        # Convert CT volume to PyTorch tensor
        # Ensure CT data represents attenuation coefficients (e.g., HU + 1024, scaled)
        # Placeholder: Assuming raw HU values for now, may need conversion/scaling
        ct_volume_np = sitk.GetArrayFromImage(ct_volume_sitk).astype(np.float32) # Shape: (nz, ny, nx)
        # Add batch and channel dimensions: [1, 1, nz, ny, nx]
        volume = torch.from_numpy(ct_volume_np).unsqueeze(0).unsqueeze(0).to(device)
        
        # --- Define Geometry using torchdr ---
        # Detector pixel dimensions
        height_px = int(round(detector_height_mm / pixel_spacing_mm))
        width_px = int(round(detector_width_mm / pixel_spacing_mm))
        
        # Convert angle to radians
        angle_rad = math.radians(angle_deg)

        # Calculate source position (rotating around Z-axis in XY plane)
        # Assumes origin (0,0,0) is the center of rotation (isocenter)
        # torchdr geometry might assume different coordinate systems - VERIFY DOCS
        source_x = sod * math.sin(angle_rad)
        source_y = -sod * math.cos(angle_rad) 
        source_z = 0.0 
        # torchdr expects source position relative to volume center? Check docs.
        # For now, assuming world coordinates with volume centered at origin.
        
        # --- Initialize torchdr Projector ---
        # Note: torchdr API might change between versions. This is based on general principles.
        # We need to define the detector plane and source position relative to the volume.
        # torchdr often uses parameter objects for geometry.
        
        # Volume parameters for torchdr (check expected order: ZYX or XYZ?)
        # Assuming ZYX based on numpy array shape from SimpleITK
        volume_spacing_zyx = torch.tensor([ct_spacing[2], ct_spacing[1], ct_spacing[0]], device=device)
        # volume_shape_zyx = torch.tensor([ct_size[2], ct_size[1], ct_size[0]], device=device) # Shape derived from volume tensor
        
        # Detector parameters
        detector_spacing_yx = torch.tensor([pixel_spacing_mm, pixel_spacing_mm], device=device)
        # detector_shape_yx = torch.tensor([height_px, width_px], device=device) # Shape passed directly

        # Define source position tensor for torchdr
        source_position = torch.tensor([source_x, source_y, source_z], device=device)

        # Define detector parameters for torchdr.DRR
        # Need rotation and translation relative to the source or world origin
        # Let's define rotation around Z axis
        rotation = torchdr.utils.Rotation.from_euler("Z", torch.tensor([angle_rad], device=device))
        
        # Initialize the DRR module
        # Using default step size, assuming volume origin is center
        # Need to verify volume origin handling and coordinate system in torchdr
        drr_simulator = torchdr.DRR(
            volume, 
            spacing=volume_spacing_zyx, 
            sdd=sdd, 
            height=height_px, 
            width=width_px,
            detector_spacing=detector_spacing_yx,
            origin=None, # Let torchdr handle origin based on volume shape/spacing? Verify.
            device=device
        )

        # --- Perform Projection ---
        # Pass the rotation and source position to the forward call
        # Note: torchdr might require batch dimensions for geometry too
        print(f"  Simulating DRR with torchdr for angle {angle_deg} deg...")
        with torch.no_grad(): # Ensure no gradients are computed if not needed
             drr_tensor = drr_simulator(rotation=rotation, source=source_position.unsqueeze(0)) # Add batch dim to source

        # Output tensor is likely line integrals (attenuation). Apply Beer-Lambert law.
        # Clamp negative values which can occur due to interpolation
        drr_tensor = torch.exp(-torch.clamp(drr_tensor, min=0.0)) 
        
        # Remove batch/channel dims, move to CPU, convert to numpy
        drr_np = drr_tensor.squeeze().cpu().numpy()

        # --- Post-process DRR ---
        # Normalize intensity (e.g., 0-1 range) - adjust as needed
        min_val = np.min(drr_np)
        max_val = np.max(drr_np)
        if max_val > min_val:
            drr_normalized = (drr_np - min_val) / (max_val - min_val)
        else:
            drr_normalized = np.zeros_like(drr_np)

        # Convert to PIL Image for potential resizing and saving
        drr_image_pil = Image.fromarray((drr_normalized * 255.0).astype(np.uint8))

        # Resize if output_size_px is specified
        if output_size_px is not None:
            target_size_wh = output_size_px # Assume (width, height)
            drr_image_pil = drr_image_pil.resize(target_size_wh, Image.Resampling.LANCZOS)

        return np.array(drr_image_pil)

    except Exception as e:
        print(f"  Error during DRR simulation for angle {angle_deg}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main(args):
    print("--- DRR Generation Script (using torchdr) ---")
    print(f"CT Input Directory: {args.ct_dir}")
    print(f"DRR Output Directory: {args.output_dir}")
    print(f"Projection Angles (degrees): {args.angles}")
    print(f"Output DRR Size (pixels): {args.size_px if args.size_px else 'Native Detector Size'}")
    print(f"Detector Height (mm): {args.det_h_mm}, Width (mm): {args.det_w_mm}, Pixel Spacing (mm): {args.pix_spacing}")
    print(f"Geometry: SDD={args.sdd}mm, SOD={args.sod}mm")

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")

    if not os.path.isdir(args.ct_dir):
        print(f"Error: CT directory not found: {args.ct_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    # TODO: Create a metadata file/structure

    ct_files = [f for f in os.listdir(args.ct_dir) if f.endswith(('.nii', '.nii.gz', '.mha', '.mhd'))]
    print(f"Found {len(ct_files)} potential CT files.")

    if not ct_files:
        print("No CT files found in the input directory. Exiting.")
        return

    output_size_pixels = tuple(args.size_px) if args.size_px else None

    print("\nProcessing CT files...")
    files_processed = 0
    files_skipped = 0
    for ct_file in tqdm(ct_files, desc="Generating DRRs"):
        ct_path = os.path.join(args.ct_dir, ct_file)
        base_filename = ct_file
        while '.' in base_filename:
            base, ext = os.path.splitext(base_filename)
            known_exts = ['.nii', '.gz', '.mha', '.mhd']
            if ext.lower() in known_exts: base_filename = base
            else: break

        try:
            # Load CT volume
            ct_volume = sitk.ReadImage(ct_path)
            
            # TODO: Add CT preprocessing (resampling, clipping) before simulation

            # Generate DRR for each specified angle
            for angle in args.angles:
                drr_filename = f"{base_filename}_drr_{angle}deg.png"
                drr_output_path = os.path.join(args.output_dir, drr_filename)

                # Generate DRR using torchdr
                drr_image_np = simulate_drr_torchdr(
                    ct_volume_sitk=ct_volume, 
                    sdd=args.sdd, 
                    sod=args.sod,
                    detector_height_mm=args.det_h_mm,
                    detector_width_mm=args.det_w_mm,
                    pixel_spacing_mm=args.pix_spacing,
                    angle_deg=angle, 
                    device=device,
                    output_size_px=output_size_pixels
                )

                if drr_image_np is not None:
                    drr_image_pil = Image.fromarray(drr_image_np)
                    drr_image_pil.save(drr_output_path)
                else:
                    print(f"  Skipped saving DRR for angle {angle} due to generation error.")
            
            # TODO: Save metadata

            files_processed += 1

        except Exception as e:
            print(f"\nError processing {ct_file}: {e}")
            files_skipped += 1

    print(f"\n--- DRR Generation Finished ---")
    print(f"Successfully processed: {files_processed} CT files.")
    print(f"Skipped due to errors: {files_skipped} CT files.")
    print(f"Generated DRRs saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DRRs from 3D CT scans using torchdr.")
    parser.add_argument("--ct_dir", type=str, required=True, help="Directory containing input 3D CT scans.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated DRR images (as PNG).")
    parser.add_argument("--angles", type=float, nargs='+', default=[0.0, 90.0], help="Space-separated list of projection angles in degrees.")
    parser.add_argument("--size_px", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=None, help="Optional output pixel size (width height) for DRR images.")
    # Basic Geometry Arguments
    parser.add_argument("--sdd", type=float, default=1000.0, help="Source-to-Detector Distance (mm).")
    parser.add_argument("--sod", type=float, default=500.0, help="Source-to-Origin (isocenter) Distance (mm).")
    parser.add_argument("--det_h_mm", type=float, default=400.0, help="Detector height (mm).")
    parser.add_argument("--det_w_mm", type=float, default=400.0, help="Detector width (mm).")
    parser.add_argument("--pix_spacing", type=float, default=0.5, help="Detector pixel spacing (mm).")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU if available.")
    
    args = parser.parse_args()
    main(args)
