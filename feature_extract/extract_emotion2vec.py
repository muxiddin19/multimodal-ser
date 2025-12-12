"""
Extract emotion2vec features for IEMOCAP
emotion2vec achieves SOTA on speech emotion recognition
"""

import torch
import numpy as np
import pickle
import os
from pathlib import Path
from funasr import AutoModel

# Initialize emotion2vec model
print("Loading emotion2vec model...")
model = AutoModel(model="iic/emotion2vec_plus_large", model_revision="v2.0.4")

def extract_emotion2vec_features(audio_path):
    """Extract emotion2vec features from audio file"""
    try:
        # emotion2vec returns both emotion predictions and features
        result = model.generate(audio_path, output_dir=None, granularity="utterance")
        
        # Get the hidden features (768-dim for large model)
        if isinstance(result, list) and len(result) > 0:
            feats = result[0].get('feats', None)
            if feats is not None:
                return feats  # [768] dimensional
        return None
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def process_iemocap_dataset(
    original_pkl_path,
    audio_base_dir,
    output_pkl_path
):
    """
    Process IEMOCAP dataset and replace ECAPA features with emotion2vec
    
    Args:
        original_pkl_path: Path to your existing pkl file with BERT+ECAPA features
        audio_base_dir: Base directory containing IEMOCAP audio files
        output_pkl_path: Output path for new pkl file
    """
    
    # Load original data
    print(f"Loading original data from {original_pkl_path}")
    with open(original_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Total samples: {len(data)}")
    
    # Process each sample
    new_data = []
    failed = 0
    
    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"Processing {i}/{len(data)}...")
        
        # Get audio path from the item
        # Adjust this based on your data structure
        if isinstance(item, dict):
            audio_path = item.get('audio_path', item.get('wav_path', None))
            text_embed = item.get('text_embed')
            label = item.get('label')
        else:
            # If it's a tuple/list format, adjust accordingly
            continue
            
        if audio_path is None:
            # Try to construct path from session/utterance ID
            # You may need to adjust this based on your data structure
            failed += 1
            continue
        
        # Make sure path exists
        if not os.path.exists(audio_path):
            # Try with base directory
            audio_path = os.path.join(audio_base_dir, audio_path)
            if not os.path.exists(audio_path):
                failed += 1
                continue
        
        # Extract emotion2vec features
        audio_embed = extract_emotion2vec_features(audio_path)
        
        if audio_embed is None:
            failed += 1
            continue
        
        new_item = {
            'text_embed': text_embed,
            'audio_embed': audio_embed,  # Now 768-dim emotion2vec
            'label': label,
            'audio_path': audio_path
        }
        new_data.append(new_item)
    
    print(f"Successfully processed: {len(new_data)}")
    print(f"Failed: {failed}")
    
    # Save new data
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(new_data, f)
    
    print(f"Saved to {output_pkl_path}")
    
    return new_data


def extract_from_audio_list(audio_paths, output_path):
    """
    Extract emotion2vec features from a list of audio paths
    Returns dict mapping audio_path -> features
    """
    features_dict = {}
    
    for i, path in enumerate(audio_paths):
        if i % 50 == 0:
            print(f"Processing {i}/{len(audio_paths)}...")
        
        feats = extract_emotion2vec_features(path)
        if feats is not None:
            features_dict[path] = feats
    
    with open(output_path, 'wb') as f:
        pickle.dump(features_dict, f)
    
    return features_dict


# ============================================================
# Alternative: If you have raw audio paths in your pkl
# ============================================================

def upgrade_existing_pkl(
    input_pkl_path,
    output_pkl_path,
    audio_dir=None
):
    """
    Upgrade existing pkl by adding emotion2vec features
    Keeps original structure but replaces/adds audio_embed
    """
    
    with open(input_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {len(data)} samples")
    print(f"Sample structure: {type(data[0])}")
    
    if isinstance(data[0], dict):
        print(f"Keys: {data[0].keys()}")
    
    # Check if we have audio paths
    # You need to know your data structure
    # Common patterns:
    # 1. data[i]['audio_path']
    # 2. data[i]['wav_path'] 
    # 3. data[i]['utterance_id'] -> need to construct path
    
    upgraded_data = []
    
    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"Processing {i}/{len(data)}...")
            
        # Determine audio path based on your data structure
        # MODIFY THIS based on your actual data format
        audio_path = None
        
        if isinstance(item, dict):
            # Try common keys
            for key in ['audio_path', 'wav_path', 'path', 'audio']:
                if key in item:
                    audio_path = item[key]
                    break
                    
            # If path doesn't exist, try with audio_dir
            if audio_path and audio_dir and not os.path.exists(audio_path):
                audio_path = os.path.join(audio_dir, os.path.basename(audio_path))
        
        if audio_path and os.path.exists(audio_path):
            # Extract emotion2vec features
            e2v_feats = extract_emotion2vec_features(audio_path)
            
            if e2v_feats is not None:
                new_item = dict(item) if isinstance(item, dict) else {
                    'text_embed': item[0],
                    'audio_embed': item[1],
                    'label': item[2]
                }
                new_item['audio_embed'] = e2v_feats  # Replace with emotion2vec
                new_item['audio_embed_dim'] = 768
                upgraded_data.append(new_item)
                continue
        
        # Keep original if emotion2vec extraction failed
        upgraded_data.append(item)
    
    # Save
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(upgraded_data, f)
    
    print(f"Saved {len(upgraded_data)} samples to {output_pkl_path}")


if __name__ == "__main__":
    # Test with a single audio file first
    test_audio = "path/to/test.wav"  # Replace with actual path
    
    if os.path.exists(test_audio):
        feats = extract_emotion2vec_features(test_audio)
        print(f"Feature shape: {feats.shape if feats is not None else 'None'}")
    
    # Then process your dataset
    # Uncomment and modify paths as needed:
    
    # upgrade_existing_pkl(
    #     input_pkl_path="features/IEMOCAP_BERT_ECAPA_train.pkl",
    #     output_pkl_path="features/IEMOCAP_BERT_E2V_train.pkl",
    #     audio_dir="/path/to/IEMOCAP/audio"
    # )