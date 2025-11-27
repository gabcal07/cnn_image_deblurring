import torch
import torchvision.transforms.functional as TF
from PIL import Image
import argparse
import os
import sys

# Assure-toi que Python trouve tes modules src
sys.path.append(os.getcwd())

# Importe ta classe modèle (adapte le chemin selon ta structure)
# from src.model import LightweightUNet 
# Si tu n'as pas séparé le fichier, tu peux coller la classe LightweightUNet ici directement.

def load_image(path):
    img = Image.open(path).convert('RGB')
    return TF.to_tensor(img).unsqueeze(0) # (1, C, H, W)

def save_image(tensor, path):
    img = TF.to_pil_image(tensor.squeeze(0).cpu().clamp(0, 1))
    img.save(path)
    print(f"✅ Image sauvegardée : {path}")

def predict_tiled(model, image_tensor, tile_size=512, overlap=64, device='cpu'):
    """Inférence intelligente par tuilage pour la HD/4K"""
    model.eval()
    b, c, h, w = image_tensor.shape
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
    parser = argparse.ArgumentParser(description="Deblur images using Lightweight U-Net")
    parser.add_argument('--input', type=str, required=True, help="Path to blurred image")
    parser.add_argument('--output', type=str, default='result.png', help="Path to save result")
    parser.add_argument('--model', type=str, default='experiments/best_model.pth', help="Path to .pth checkpoint")
    parser.add_argument('--device', type=str, default='mps', help="Device (cuda, mps, cpu)")
    args = parser.parse_args()

    # 1. Device Setup
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("⚠️ MPS not available, switching to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 2. Load Model
    # Important : Il faut que les paramètres (48 filtres vs 32) correspondent à ton entraînement !
    # Si tu as changé start_filters, mets-le à jour ici.
    print("Loading model...")
    try:
        # Pense à importer ta classe LightweightUNet ou à la définir ici
        from src.model import LightweightUNet 
        model = LightweightUNet(in_channels=3, out_channels=3) 
        
        # Gestion du chargement (parfois le state_dict est imbriqué sous 'model_state_dict')
        checkpoint = torch.load(args.model, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Erreur chargement modèle : {e}")
        return

    # 3. Process
    if os.path.isdir(args.input):
        # Si c'est un dossier, on traite tout
        print(f"Processing folder: {args.input}")
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                in_path = os.path.join(args.input, filename)
                out_path = os.path.join(args.output, f"deblurred_{filename}")
                
                img = load_image(in_path)
                res = predict_tiled(model, img, device=device)
                save_image(res, out_path)
    else:
        # Image unique
        print(f"Processing file: {args.input}")
        img = load_image(args.input)
        res = predict_tiled(model, img, device=device)
        save_image(res, args.output)

if __name__ == '__main__':
    main()