import os
import torch
import timm
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data2', help='Root for data2')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_name', type=str, default='vit_small_patch16_224_dino', help='timm model name')
    args = parser.parse_args()

    # Paths
    image_root = os.path.join(args.data_root, "svg_png")
    feature_root = os.path.join(args.data_root, "features")
    if not os.path.exists(feature_root):
        os.makedirs(feature_root)
        
    # Load Model
    print(f"Loading model {args.model_name}...")
    try:
        model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Available models:", timm.list_models('*dino*'))
        return

    model.cuda()
    model.eval()
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Get all image folders
    # Structure: data2/svg_png/0000/00000007/
    print("Scanning directories...")
    subfolders = sorted(os.listdir(image_root)) # 0000, 0001...
    
    all_ids = []
    for sub in subfolders:
        sub_path = os.path.join(image_root, sub)
        if os.path.isdir(sub_path):
            ids = sorted(os.listdir(sub_path)) # 00000007, ...
            for id_ in ids:
                all_ids.append((sub, id_))
                
    print(f"Found {len(all_ids)} samples.")
    
    views = ["Front", "FrontTopRight", "Right", "Top"]
    
    # Processing Loop
    # We process sample by sample (loading 4 images)
    # Ideally we should batch this, but for simplicity and memory safety we do it simply first
    # Or we can batch the 4 images -> (4, 3, 224, 224) -> model -> (4, 384)
    
    for sub, id_ in tqdm(all_ids):
        save_path = os.path.join(feature_root, sub, f"{id_}.npy")
        
        # Skip if already exists
        if os.path.exists(save_path):
            continue
            
        # Ensure subfolder exists in feature_root
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        img_dir = os.path.join(image_root, sub, id_)
        
        images = []
        valid = True
        for view in views:
            img_name = f"{id_}_{view}.png"
            img_path = os.path.join(img_dir, img_name)
            try:
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                images.append(img)
            except Exception as e:
                # print(f"Error loading {img_path}: {e}")
                valid = False
                break
        
        if not valid:
            # Save zeros or skip? Better to save zeros to keep alignment with vectors
            features = np.zeros((4, 384), dtype=np.float32)
        else:
            # Stack: (4, 3, 224, 224)
            batch = torch.stack(images).cuda()
            
            with torch.no_grad():
                # Forward
                feats = model(batch) # (4, 384)
                
            features = feats.cpu().numpy().astype(np.float32)
            
        # Save
        np.save(save_path, features)

if __name__ == '__main__':
    main()
