import os
import sys
from datasets import load_dataset
from tqdm import tqdm
import shutil

def download_and_organize_hmdb51():
    print("=" * 60)
    print("HMDB51 Dataset Download (via Hugging Face)")
    print("=" * 60)
    
    data_dir = "data/hmdb51"
    
    if os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0:
        print(f"✓ HMDB51 dataset already exists at {data_dir}")
        action_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"  Found {len(action_dirs)} action categories")
        return
    
    print("\nDownloading HMDB51 dataset from Hugging Face...")
    print("This may take a while (~2GB download)...")
    
    try:
        ds = load_dataset("jili5044/hmdb51", trust_remote_code=True)
        print("✓ Dataset downloaded successfully")
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nYou may need to login to Hugging Face first:")
        print("  huggingface-cli login")
        print("\nOr set your token:")
        print("  export HF_TOKEN=your_token_here")
        sys.exit(1)
    
    os.makedirs(data_dir, exist_ok=True)
    
    print("\nOrganizing dataset into action categories...")
    
    splits = ['train', 'test']
    for split in splits:
        if split not in ds:
            continue
            
        print(f"\nProcessing {split} split...")
        split_data = ds[split]
        
        for idx, item in enumerate(tqdm(split_data, desc=f"Organizing {split}")):
            label = item.get('label', item.get('category', 'unknown'))
            video = item.get('video', None)
            
            if isinstance(label, int) and 'label_names' in split_data.features:
                label_names = split_data.features['label'].names
                label = label_names[label]
            
            action_dir = os.path.join(data_dir, str(label))
            os.makedirs(action_dir, exist_ok=True)
            
            video_filename = item.get('filename', f'{split}_{idx:05d}.avi')
            if not video_filename.endswith(('.avi', '.mp4')):
                video_filename = f"{video_filename}.avi"
            
            video_path = os.path.join(action_dir, video_filename)
            
            if video is not None and hasattr(video, 'save'):
                try:
                    video.save(video_path)
                except:
                    pass
            elif isinstance(video, str) and os.path.exists(video):
                shutil.copy(video, video_path)
    
    action_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    print("\n" + "=" * 60)
    print("✓ HMDB51 dataset setup complete!")
    print("=" * 60)
    print(f"Dataset location: {os.path.abspath(data_dir)}")
    print(f"Action categories: {len(action_dirs)}")
    
    if action_dirs:
        print("\nCategories found:")
        for action in sorted(action_dirs)[:10]:
            video_count = len([f for f in os.listdir(os.path.join(data_dir, action)) 
                              if f.endswith(('.avi', '.mp4'))])
            print(f"  - {action}: {video_count} videos")
        if len(action_dirs) > 10:
            print(f"  ... and {len(action_dirs) - 10} more categories")

if __name__ == "__main__":
    download_and_organize_hmdb51()

