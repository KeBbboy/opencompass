import json
import os
import glob
import numpy as np
from collections import defaultdict

def load_entropy_data(log_dir):
    """
    Load entropy data from JSON files
    Returns a dictionary: {sample_idx: {layer_idx: entropy_mean}}
    """
    data = defaultdict(dict)
    
    # Find all JSON files matching the pattern
    json_files = glob.glob(os.path.join(log_dir, "entropy_layer*_sample*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                content = json.load(f)
                
            layer_idx = content['layer_idx']
            sample_idx = content['sample_idx']
            entropy_mean = content['entropy_mean']
            
            data[sample_idx][layer_idx] = entropy_mean
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    return data

def calculate_layer_averages(data):
    """
    Calculate average entropy mean for each layer across all samples
    Returns a dictionary: {layer_idx: statistics}
    """
    layer_averages = {}
    
    # Get all unique layers
    all_layers = set()
    for sample_data in data.values():
        all_layers.update(sample_data.keys())
    
    # Calculate average for each layer
    for layer_idx in sorted(all_layers):
        entropy_values = []
        for sample_idx in data.keys():
            if layer_idx in data[sample_idx]:
                entropy_values.append(data[sample_idx][layer_idx])
        
        if entropy_values:
            layer_averages[layer_idx] = {
                'layer_idx': layer_idx,
                'average_entropy_mean': float(np.mean(entropy_values))
            }
    
    return layer_averages

def save_layer_averages(layer_averages, output_path):
    """
    Save layer averages to a JSON file
    """
    # Convert to a list sorted by layer index
    output_data = {
        'description': 'Average entropy mean for each layer across all samples',
        'layers': [layer_averages[layer_idx] for layer_idx in sorted(layer_averages.keys())]
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Layer averages saved to {output_path}")

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # JSON files are in the 'json' subdirectory
    json_dir = os.path.join(script_dir, 'json')
    
    # Load data
    print("Loading entropy data...")
    data = load_entropy_data(json_dir)
    
    # Print summary
    print(f"\nLoaded data for {len(data)} samples")
    for sample_idx in sorted(data.keys()):
        print(f"  Sample {sample_idx}: {len(data[sample_idx])} layers")
    
    # Calculate layer averages
    print("\nCalculating layer averages...")
    layer_averages = calculate_layer_averages(data)
    
    # Print layer averages
    print(f"\nLayer averages across all samples:")
    for layer_idx in sorted(layer_averages.keys()):
        stats = layer_averages[layer_idx]
        print(f"  Layer {layer_idx}: avg={stats['average_entropy_mean']:.4f}")
    
    # Save to JSON file
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'layer_averages.json')
    save_layer_averages(layer_averages, output_path)

if __name__ == "__main__":
    main()
