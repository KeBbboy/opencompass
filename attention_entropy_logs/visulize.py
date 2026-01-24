import json
import os
import glob
import matplotlib.pyplot as plt
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

def plot_entropy_by_layer(data, output_path='entropy_by_layer.png'):
    """
    Plot entropy mean vs layer for each sample
    """
    plt.figure(figsize=(12, 8))
    
    # Define colors for different samples
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Sort samples for consistent plotting
    samples = sorted(data.keys())
    
    for idx, sample_idx in enumerate(samples):
        layers = sorted(data[sample_idx].keys())
        entropy_means = [data[sample_idx][layer] for layer in layers]
        
        plt.plot(layers, entropy_means, 
                marker='o', 
                markersize=4,
                linewidth=2,
                label=f'Sample {sample_idx}',
                color=colors[idx],
                alpha=0.8)
    
    plt.xlabel('Layer Index', fontsize=14, fontweight='bold')
    plt.ylabel('Entropy Mean', fontsize=14, fontweight='bold')
    plt.title('Attention Entropy Mean Across Layers', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()

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
    
    # Create visualization
    print("\nGenerating plot...")
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'entropy_by_layer.png')
    plot_entropy_by_layer(data, output_path)

if __name__ == "__main__":
    main()
