"""
Quick script to visualize training metrics
Run this after training to see episode returns and other metrics
"""
import sys
from visualize_training import load_tensorboard_data, plot_training_metrics, plot_episode_returns_detailed, print_summary_statistics

# Set the log directory (change this if needed)
LOGDIR = 'dsp2/logs/improved_adaptive'

print("=" * 80)
print("TRAINING VISUALIZATION")
print("=" * 80)
print(f"Loading data from: {LOGDIR}\n")

try:
    # Load data from TensorBoard logs
    data = load_tensorboard_data(LOGDIR)
    
    # Print summary statistics
    print_summary_statistics(data)
    
    # Create main visualization
    print("\nGenerating plots...")
    plot_training_metrics(data, save_path='D:\\FPTU\\Term8\\DSP391m\\DSP391m\\dsp2\\visualization\\images\\training_metrics.png')
    
    # Create detailed episode return analysis
    plot_episode_returns_detailed(data, save_path='D:\\FPTU\\Term8\\DSP391m\\DSP391m\\dsp2\\visualization\\images\\episode_returns_detailed.png')
    
    print("\n" + "=" * 80)
    print("Visualization complete!")
    print("Generated files:")
    print("  - training_metrics.png")
    print("  - episode_returns_detailed.png")
    print("=" * 80)
    
except Exception as e:
    print(f"Error: {e}")
    print("\nMake sure you have:")
    print("  1. Completed at least one training run")
    print("  2. TensorBoard log files in the specified directory")
    print(f"  3. The directory exists: {LOGDIR}")
    sys.exit(1)
