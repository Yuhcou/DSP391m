"""
Visualize training metrics from TensorBoard logs
Shows episode rewards, losses, and other metrics during training
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path

def load_tensorboard_data(logdir):
    """Load data from TensorBoard event files."""
    event_files = list(Path(logdir).glob('events.out.tfevents.*'))
    
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {logdir}")
    
    # Use the most recent event file
    event_file = sorted(event_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"Loading data from: {event_file.name}")
    
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    
    # Get available tags
    available_tags = ea.Tags()['scalars']
    print(f"Available metrics: {available_tags}")
    
    data = {}
    for tag in available_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': np.array(steps), 'values': np.array(values)}
    
    return data

def smooth_curve(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

def plot_training_metrics(data, save_path=None):
    """Create comprehensive visualization of training metrics."""
    
    # Determine subplot layout based on available metrics
    metrics_to_plot = []
    
    if 'episode_return' in data:
        metrics_to_plot.append(('episode_return', 'Episode Return', 'Total Return'))
    if 'episode_avg_return' in data:
        metrics_to_plot.append(('episode_avg_return', 'Episode Avg Return', 'Avg Return per Step'))
    if 'loss' in data:
        metrics_to_plot.append(('loss', 'Training Loss', 'Loss'))
    if 'q_mean' in data:
        metrics_to_plot.append(('q_mean', 'Mean Q-Value', 'Q-Value'))
    if 'n_waiting' in data:
        metrics_to_plot.append(('n_waiting', 'Waiting Passengers', 'Count'))
    if 'awt' in data:
        metrics_to_plot.append(('awt', 'Average Waiting Time', 'Time Steps'))
    if 'ajt' in data:
        metrics_to_plot.append(('ajt', 'Average Journey Time', 'Time Steps'))
    if 'base_reward' in data:
        metrics_to_plot.append(('base_reward', 'Base Reward', 'Reward'))
    if 'adaptive_reward_bonus' in data:
        metrics_to_plot.append(('adaptive_reward_bonus', 'Adaptive Reward Bonus', 'Bonus'))
    
    n_metrics = len(metrics_to_plot)
    if n_metrics == 0:
        print("No metrics found to plot!")
        return
    
    # Create figure with subplots
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle('Training Metrics Over Time', fontsize=16, fontweight='bold')
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (metric_key, title, ylabel) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        steps = data[metric_key]['steps']
        values = data[metric_key]['values']
        
        # Plot raw values
        ax.plot(steps, values, alpha=0.3, linewidth=0.5, color='blue', label='Raw')
        
        # Plot smoothed values
        if len(values) > 10:
            smoothed = smooth_curve(values, weight=0.6)
            ax.plot(steps, smoothed, linewidth=2, color='red', label='Smoothed (0.6)')
        
        ax.set_xlabel('Training Steps', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        final_val = values[-1]
        
        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nFinal: {final_val:.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()

def plot_episode_returns_detailed(data, save_path=None):
    """Create detailed plot focused on episode returns."""
    
    if 'episode_return' not in data:
        print("No episode_return data found!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Episode Return Analysis', fontsize=16, fontweight='bold')
    
    steps = data['episode_return']['steps']
    values = data['episode_return']['values']
    
    # Plot 1: Raw and smoothed returns
    ax1 = axes[0, 0]
    ax1.plot(steps, values, alpha=0.3, linewidth=0.5, color='blue', label='Raw Returns')
    smoothed_90 = smooth_curve(values, weight=0.6)
    smoothed_95 = smooth_curve(values, weight=0.8)
    ax1.plot(steps, smoothed_90, linewidth=2, color='red', label='Smoothed (0.6)')
    ax1.plot(steps, smoothed_95, linewidth=2, color='green', label='Smoothed (0.8)')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Episode Return')
    ax1.set_title('Episode Returns Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of returns
    ax2 = axes[0, 1]
    ax2.hist(values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.2f}')
    ax2.axvline(x=np.median(values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(values):.2f}')
    ax2.set_xlabel('Episode Return')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Episode Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling statistics
    ax3 = axes[1, 0]
    window_size = 100
    if len(values) >= window_size:
        rolling_mean = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
        rolling_std = np.array([np.std(values[max(0, i-window_size):i]) 
                               for i in range(window_size, len(values)+1)])
        valid_steps = steps[window_size-1:]
        
        ax3.plot(valid_steps, rolling_mean, linewidth=2, color='blue', label=f'Rolling Mean ({window_size})')
        ax3.fill_between(valid_steps, 
                        rolling_mean - rolling_std, 
                        rolling_mean + rolling_std, 
                        alpha=0.3, color='blue', label='±1 Std Dev')
    else:
        ax3.plot(steps, values, linewidth=2, color='blue')
    
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Episode Return')
    ax3.set_title(f'Rolling Statistics (Window={window_size})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Progress over time (quartiles)
    ax4 = axes[1, 1]
    n_bins = 20
    bin_size = len(values) // n_bins
    if bin_size > 0:
        bin_means = []
        bin_q25 = []
        bin_q75 = []
        bin_centers = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = min((i + 1) * bin_size, len(values))
            bin_data = values[start_idx:end_idx]
            
            if len(bin_data) > 0:
                bin_means.append(np.mean(bin_data))
                bin_q25.append(np.percentile(bin_data, 25))
                bin_q75.append(np.percentile(bin_data, 75))
                bin_centers.append(steps[start_idx + len(bin_data)//2])
        
        ax4.plot(bin_centers, bin_means, linewidth=3, color='red', marker='o', label='Mean')
        ax4.fill_between(bin_centers, bin_q25, bin_q75, alpha=0.3, color='blue', label='25-75 Percentile')
        ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Episode Return')
        ax4.set_title('Learning Progress (Binned Statistics)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Detailed plot saved to: {save_path}")
    
    plt.show()

def print_summary_statistics(data):
    """Print summary statistics of key metrics."""
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY STATISTICS")
    print("=" * 80)
    
    metrics_of_interest = {
        'episode_return': 'Episode Return',
        'episode_avg_return': 'Episode Avg Return',
        'awt': 'Average Waiting Time',
        'ajt': 'Average Journey Time',
        'n_waiting': 'Waiting Passengers',
        'loss': 'Training Loss',
        'q_mean': 'Mean Q-Value'
    }
    
    for metric_key, metric_name in metrics_of_interest.items():
        if metric_key in data:
            values = data[metric_key]['values']
            
            print(f"\n{metric_name}:")
            print(f"  Mean:   {np.mean(values):>10.2f}")
            print(f"  Std:    {np.std(values):>10.2f}")
            print(f"  Min:    {np.min(values):>10.2f}")
            print(f"  Max:    {np.max(values):>10.2f}")
            print(f"  Median: {np.median(values):>10.2f}")
            
            # First 10% vs Last 10%
            split_idx = len(values) // 10
            if split_idx > 0:
                first_10 = values[:split_idx]
                last_10 = values[-split_idx:]
                improvement = np.mean(last_10) - np.mean(first_10)
                print(f"  First 10% Mean: {np.mean(first_10):>10.2f}")
                print(f"  Last 10% Mean:  {np.mean(last_10):>10.2f}")
                print(f"  Improvement:    {improvement:>10.2f} ({improvement/np.abs(np.mean(first_10))*100:+.1f}%)")
    
    print("\n" + "=" * 80)

def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training metrics from TensorBoard logs')
    parser.add_argument('--logdir', type=str, default='dsp2/logs/improved_adaptive',
                       help='Path to TensorBoard log directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for saving plots (optional)')
    parser.add_argument('--detailed', action='store_true',
                       help='Create detailed episode return analysis')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.logdir}")
    data = load_tensorboard_data(args.logdir)
    
    # Print summary statistics
    print_summary_statistics(data)
    
    # Create visualizations
    output_path = args.output if args.output else 'D:\\FPTU\\Term8\\DSP391m\\DSP391m\\dsp2\\visualization\\images\\training_metrics.png'
    plot_training_metrics(data, save_path=output_path)
    
    if args.detailed:
        detailed_path = output_path.replace('.png', '_detailed.png')
        plot_episode_returns_detailed(data, save_path=detailed_path)

if __name__ == "__main__":
    main()
