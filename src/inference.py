import torch
import torchvision.transforms.functional as TF
from PIL import Image
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.utils.metrics import calculate_psnr


def get_image_files(root_dir):
    """Recursively find all image files in a directory."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    files_list = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, root_dir)
                files_list.append((full_path, rel_path))
    return sorted(files_list)


def load_image(path):
    img = Image.open(path).convert("RGB")
    return TF.to_tensor(img).unsqueeze(0)  # (1, C, H, W)


def save_image(tensor, path):
    img = TF.to_pil_image(tensor.squeeze(0).cpu().clamp(0, 1))
    img.save(path)
    print(f"Image sauvegardée : {path}")


def save_comparison(input_tensor, output_tensor, path):
    input_img = input_tensor.squeeze(0).cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    output_img = output_tensor.squeeze(0).cpu().permute(1, 2, 0).clamp(0, 1).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_img)
    axes[0].set_title("Blurred Input")
    axes[0].axis("off")

    axes[1].imshow(output_img)
    axes[1].set_title("Deblurred Output")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Comparison saved : {path}")


def predict_tiled(model, image_tensor, tile_size=512, overlap=64, device="cpu"):
    """Inférence intelligente par tuilage pour la HD/4K"""
    model.eval()
    b, c, h, w = image_tensor.shape

    # Optimization: If image is 512x512, predict in one go
    if h == 512 and w == 512:
        with torch.no_grad():
            return model(image_tensor.to(device))

    output = torch.zeros((b, c, h, w), device=device)
    count_map = torch.zeros((b, 1, h, w), device=device)
    stride = tile_size - overlap

    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                input_tile = image_tensor[:, :, y_start:y_end, x_start:x_end].to(device)
                pred_tile = model(input_tile)

                output[:, :, y_start:y_end, x_start:x_end] += pred_tile
                count_map[:, :, y_start:y_end, x_start:x_end] += 1.0

    return output / count_map


def main():
    parser = argparse.ArgumentParser(
        description="Deblur images using Lightweight U-Net"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to blurred image"
    )
    parser.add_argument(
        "--output", type=str, default="result.png", help="Path to save result"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="experiments/unet_light_v3_42/best_model.pth",
        help="Path to .pth checkpoint",
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Device (cuda, mps, cpu)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Save comparison figure instead of just output",
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=None,
        help="Path to sharp images (folder or file) for PSNR calculation",
    )
    args = parser.parse_args()

    # 1. Device Setup
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, switching to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2. Load Model
    print("Loading model...")
    try:
        from src.models.lightunet import LightweightUNet

        model = LightweightUNet(
            in_channels=3, out_channels=3, global_residual=True, start_filters=48
        )

        # Gestion du chargement (parfois le state_dict est imbriqué sous 'model_state_dict')
        checkpoint = torch.load(args.model, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Erreur chargement modèle : {e}")
        return

    # 3. Process
    if os.path.isdir(args.input):
        print(f"Processing folder: {args.input}")
        
        image_files = get_image_files(args.input)
        print(f"Found {len(image_files)} images.")
        
        psnr_values = []

        for in_path, rel_path in tqdm(image_files, desc="Deblurring"):
            # Construct output path preserving structure
            out_path = os.path.join(args.output, rel_path)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            img = load_image(in_path)
            res = predict_tiled(model, img, device=device)
            
            # Calculate PSNR if ground truth is provided
            if args.ground_truth:
                # Try to find GT with same relative path
                gt_path = os.path.join(args.ground_truth, rel_path)
                
                if os.path.exists(gt_path):
                    gt_img = load_image(gt_path).to(device)
                    # Ensure dimensions match
                    if res.shape != gt_img.shape:
                            print(f"Warning: Shape mismatch for {rel_path}. Res: {res.shape}, GT: {gt_img.shape}")
                    else:
                        current_psnr = calculate_psnr(res, gt_img).item()
                        psnr_values.append(current_psnr)
                        # print(f"Processed {rel_path} | PSNR: {current_psnr:.2f} dB") # Too verbose for tqdm
                else:
                    # Fallback: check if GT folder is flat but input is nested (or vice versa)
                    # This is a simple heuristic: check if filename exists in GT root
                    filename = os.path.basename(rel_path)
                    flat_gt_path = os.path.join(args.ground_truth, filename)
                    if os.path.exists(flat_gt_path):
                         gt_img = load_image(flat_gt_path).to(device)
                         if res.shape == gt_img.shape:
                            current_psnr = calculate_psnr(res, gt_img).item()
                            psnr_values.append(current_psnr)
                    else:
                        pass # GT not found
            
            # Always save the result
            if args.test:
                save_comparison(img, res, out_path)
            else:
                save_image(res, out_path)
        
        # Summary statistics
        if psnr_values:
            mean_psnr = np.mean(psnr_values)
            print(f"\n{'='*40}")
            print(f"Average PSNR: {mean_psnr:.2f} dB")
            print(f"Min PSNR: {np.min(psnr_values):.2f} dB")
            print(f"Max PSNR: {np.max(psnr_values):.2f} dB")
            print(f"{'='*40}")

            # Plot distribution
            plt.figure(figsize=(10, 6))
            plt.hist(psnr_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            plt.title(f"PSNR Distribution (Mean: {mean_psnr:.2f} dB)")
            plt.xlabel("PSNR (dB)")
            plt.ylabel("Count")
            plt.axvline(mean_psnr, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_psnr:.2f}')
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            dist_path = os.path.join(args.output, "psnr_distribution.png")
            plt.savefig(dist_path)
            print(f"PSNR distribution saved to {dist_path}")

    else:
        # Image unique
        print(f"Processing file: {args.input}")
        img = load_image(args.input)
        res = predict_tiled(model, img, device=device)
        
        if args.ground_truth:
             if os.path.isdir(args.ground_truth):
                 # Try to find file with same name
                 filename = os.path.basename(args.input)
                 gt_path = os.path.join(args.ground_truth, filename)
             else:
                 gt_path = args.ground_truth
            
             if os.path.exists(gt_path):
                gt_img = load_image(gt_path).to(device)
                psnr = calculate_psnr(res, gt_img).item()
                print(f"PSNR: {psnr:.2f} dB")
        
        if args.test:
            save_comparison(img, res, args.output)
        else:
            save_image(res, args.output)


if __name__ == "__main__":
    main()
