import matplotlib.pyplot as plt
import pandas as pd
import re
from collections import defaultdict

# Configuration - modify these paths as needed
LOG_FILE_PATH = './exp/normal2supernet/logs/valid_console.txt'  # Path to your log file
SAVE_PATH = './plots/training_plot_n2s.png'  # Set to None to show interactively
SAVE_DPI = 300  # Resolution when saving

def parse_log_file(file_path):
    data_points = []
    point_counter = 0  # Sequential counter for x-axis
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line.startswith("Valid"):
                continue
                
            try:
                # Split into metrics and model data
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                    
                metrics_part = parts[0]
                model_part = parts[1]
                
                # Extract metrics
                metrics = {
                    'point': point_counter,
                    'valid_loss': float(re.search(r'loss=([\d.]+)', metrics_part).group(1)),
                    'valid_top1': float(re.search(r'top-1=([\d.]+)', metrics_part).group(1)),
                    'train_top1': float(re.search(r'Train top-1 ([\d.]+)', metrics_part).group(1)),
                    'train_loss': float(re.search(r'Train loss ([\d.]+)', metrics_part).group(1))
                }
                
                # Extract model metrics (handles any D*-WM*-H* pattern)
                model_metrics = re.findall(r'(D\d+-WM\d+-H\d+) \(([\d.]+)\)', model_part)
                for model, value in model_metrics:
                    metrics[model] = float(value)
                
                data_points.append(metrics)
                point_counter += 1
                
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(f"Error: {e}")
                continue
    
    return pd.DataFrame(data_points)

def visualize_data(df, save_path=SAVE_PATH, dpi=SAVE_DPI):
    plt.figure(figsize=(16, 12))
    
    # Plot losses
    plt.subplot(3, 1, 1)
    plt.plot(df['point'], df['valid_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.plot(df['point'], df['train_loss'], 'b--', label='Training Loss', linewidth=2)
    plt.xlabel('Epoch (10/step)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Plot accuracies
    plt.subplot(3, 1, 2)
    plt.plot(df['point'], df['valid_top1'], 'g-', label='Validation Top-1', linewidth=2)
    plt.plot(df['point'], df['train_top1'], 'm--', label='Training Top-1', linewidth=2)
    plt.xlabel('Epoch (10/step)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Top-1 Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Plot model metrics
    plt.subplot(3, 1, 3)
    model_cols = [col for col in df.columns if re.match(r'D\d+-WM\d+-H\d+', col)]
    
    # Group by depth (D value) for better visualization
    depth_groups = defaultdict(list)
    for col in model_cols:
        depth = int(re.search(r'D(\d+)-', col).group(1))
        depth_groups[depth].append(col)
    
    # Create a colormap for different depths
    cmap = plt.get_cmap('tab20')
    for i, (depth, cols) in enumerate(sorted(depth_groups.items())):
        color = cmap(i / len(depth_groups))
        for col in cols:
            # Extract head number for label
            heads = int(re.search(r'H(\d+)', col).group(1))
            plt.plot(df['point'], df[col], 
                    color=color, 
                    linestyle='-' if heads == 12 else '--',
                    label=f'D{depth}-H{heads}',
                    alpha=0.8)
    
    plt.xlabel('Epoch (10/step)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Component Performance by Depth and Heads', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# Main execution
if __name__ == "__main__":
    try:
        df = parse_log_file(LOG_FILE_PATH)
        if df.empty:
            print(f"No valid data points found in {LOG_FILE_PATH}")
        else:
            visualize_data(df)
    except FileNotFoundError:
        print(f"Error: File {LOG_FILE_PATH} not found. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")