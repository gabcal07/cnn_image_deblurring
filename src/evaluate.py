import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from src.dataset import GoProDataset
from src.utils.metrics import calculate_psnr
from torch.utils.data import DataLoader

def predict_tiled(model, image_tensor, tile_size=512, overlap=64, device='mps'):
    """
    Performs inference on a large image by splitting it into overlapping tiles.
    Reconstructs the full image by blending predictions.
    
    Args:
        model: Trained Pytorch model
        image_tensor: Input image of shape (1, C, H, W)
        tile_size: Size of the sliding window (e.g., 512)
        overlap: Overlap between tiles to avoid edge artifacts (e.g., 64)
        device: 'mps', 'cuda', or 'cpu'
    """
    model.eval()
    b, c, h, w = image_tensor.shape
    
    # Initialize output canvas and count map (to average overlapping regions)
    output = torch.zeros((b, c, h, w), device=device)
    count_map = torch.zeros((b, 1, h, w), device=device)
    
    stride = tile_size - overlap
    
    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Calculate tile boundaries
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Adjust start position to handle image borders (prevent padding)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)
                
                # Extract tile and move to GPU
                input_tile = image_tensor[:, :, y_start:y_end, x_start:x_end].to(device)
                
                # Predict
                pred_tile = model(input_tile)
                
                # Accumulate prediction and update count map
                output[:, :, y_start:y_end, x_start:x_end] += pred_tile
                count_map[:, :, y_start:y_end, x_start:x_end] += 1.0

    # Normalize by the count map (averaging overlapping regions)
    final_output = output / count_map
    
    return final_output

def evaluate_full_resolution(model, dataset_paths, tile_size=512, overlap=64, device='mps'):
    """
    Evaluates the model on the full validation dataset using tiling strategy.
    Computes detailed statistics (Mean, Median, Min, Max, Std).
    """
    val_blur_paths, val_sharp_paths = dataset_paths
    
    # Create a temporary dataset in full resolution mode (no cropping)
    # Ensure GoProDataset is defined in your scope
    eval_dataset = GoProDataset(
        val_blur_paths, val_sharp_paths,
        is_train=False,
        full_image=True 
    )
    
    print(f"🚀 Starting Full Resolution Evaluation on {len(eval_dataset)} images...")
    print(f"   Tiling strategy: {tile_size}x{tile_size} (Overlap: {overlap})")
    
    psnr_values = []
    
    # Iterate through the dataset
    for i in tqdm(range(len(eval_dataset)), desc="Evaluating"):
        # Load raw tensors (CPU)
        blur_tensor, sharp_tensor = eval_dataset[i]
        
        # Add batch dimension (1, C, H, W)
        # blur_input stays on CPU initially, predict_tiled handles GPU transfer per tile
        blur_input = blur_tensor.unsqueeze(0) 
        sharp_tensor = sharp_tensor.unsqueeze(0).to(device) # GT on GPU for metric calc
        
        # Inference
        prediction = predict_tiled(model, blur_input, tile_size=tile_size, overlap=overlap, device=device)
        
        # Clamp values to valid range [0, 1] for metric calculation
        prediction = prediction.clamp(0, 1)
        
        # Calculate PSNR
        # Ensure calculate_psnr is defined in your scope
        psnr = calculate_psnr(prediction, sharp_tensor).item()
        psnr_values.append(psnr)
        
    # Convert to numpy for stats
    psnr_array = np.array(psnr_values)
    
    print("\n" + "="*40)
    print("📊 FINAL RESULTS (FULL RESOLUTION)")
    print("="*40)
    print(f"Total Images    : {len(psnr_array)}")
    print(f"Mean PSNR       : {np.mean(psnr_array):.2f} dB")
    print(f"Median PSNR     : {np.median(psnr_array):.2f} dB")
    print(f"Min PSNR        : {np.min(psnr_array):.2f} dB")
    print(f"Max PSNR        : {np.max(psnr_array):.2f} dB")
    print(f"Std Dev         : {np.std(psnr_array):.2f} dB")
    print("="*40)
    
    return psnr_array