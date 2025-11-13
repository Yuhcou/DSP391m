"""
Baseline Evaluator for Traditional EGCS Algorithms

This script evaluates traditional elevator dispatching algorithms:
- Random (baseline)
- Collective Control
- Nearest Car
- Sectoring/Zoning
- Fixed Priority

Metrics collected:
- Average Waiting Time (AWT): Time passengers wait in hall queues
- Average Journey Time (AJT): Total time from arrival to destination
- Average System Time: Waiting + In-car time
- Total Passengers Served
- Episode Return (cumulative reward)
"""
import os
import sys
import yaml
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dsp2.env.egcs_env import EGCSEnv
from dsp2.agents.traditional_algorithms import TraditionalAlgorithmAdapter


@dataclass
class EpisodeMetrics:
    """Metrics collected during an episode."""
    waiting_times: List[float] = field(default_factory=list)
    journey_times: List[float] = field(default_factory=list)
    total_wait_steps: int = 0
    total_incar_steps: int = 0
    passengers_served: int = 0
    passengers_generated: int = 0
    episode_return: float = 0.0
    steps: int = 0
    
    @property
    def avg_waiting_time(self) -> float:
        """Average waiting time per passenger (in time steps)."""
        if not self.waiting_times:
            return 0.0
        return np.mean(self.waiting_times)
    
    @property
    def avg_journey_time(self) -> float:
        """Average journey time per passenger (in time steps)."""
        if not self.journey_times:
            return 0.0
        return np.mean(self.journey_times)
    
    @property
    def avg_system_time(self) -> float:
        """Average total system time per passenger."""
        return self.avg_waiting_time + self.avg_journey_time
    
    @property
    def avg_waiting_queue(self) -> float:
        """Average number of passengers waiting per step."""
        if self.steps == 0:
            return 0.0
        return self.total_wait_steps / self.steps
    
    @property
    def avg_incar_count(self) -> float:
        """Average number of passengers in cars per step."""
        if self.steps == 0:
            return 0.0
        return self.total_incar_steps / self.steps


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_env(cfg: Dict[str, Any], seed: int = None) -> EGCSEnv:
    """Create environment from configuration."""
    return EGCSEnv(
        n_floors=cfg['n_floors'],
        m_elevators=cfg['m_elevators'],
        capacity=cfg['capacity'],
        dt=cfg['dt'],
        lambda_fn=lambda t: cfg['lambda'],
        t_max=cfg['t_max'],
        seed=seed if seed is not None else cfg['seed'],
        w_wait=cfg['w_wait'],
        w_incar=cfg['w_incar'],
        r_alight=cfg['r_alight'],
        r_board=cfg['r_board'],
        penalty_normalize=cfg['penalty_normalize']
    )


class PassengerTracker:
    """Track individual passenger metrics."""
    
    def __init__(self):
        self.waiting_passengers = {}  # {passenger_id: arrival_time}
        self.boarded_passengers = {}  # {passenger_id: (arrival_time, board_time)}
        self.completed_waiting = []
        self.completed_journey = []
        self.next_id = 0
    
    def register_arrivals(self, n_arrivals: int, current_time: int):
        """Register new passenger arrivals."""
        for _ in range(n_arrivals):
            self.waiting_passengers[self.next_id] = current_time
            self.next_id += 1
    
    def register_boarding(self, n_boarded: int, current_time: int):
        """Register passengers boarding elevators."""
        # Take oldest waiting passengers
        boarded_ids = sorted(self.waiting_passengers.keys())[:n_boarded]
        for pid in boarded_ids:
            arrival_time = self.waiting_passengers.pop(pid)
            self.boarded_passengers[pid] = (arrival_time, current_time)
            waiting_time = current_time - arrival_time
            self.completed_waiting.append(waiting_time)
    
    def register_alighting(self, n_alighted: int, current_time: int):
        """Register passengers alighting (reaching destination)."""
        # Take oldest boarded passengers
        alighted_ids = sorted(self.boarded_passengers.keys())[:n_alighted]
        for pid in alighted_ids:
            arrival_time, board_time = self.boarded_passengers.pop(pid)
            journey_time = current_time - board_time
            self.completed_journey.append(journey_time)


def evaluate_algorithm(algorithm_name: str, cfg: Dict[str, Any], 
                       n_episodes: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluate a traditional algorithm.
    
    Args:
        algorithm_name: Name of algorithm ('random', 'collective', 'nearest_car', 
                       'sectoring', 'fixed_priority')
        cfg: Configuration dictionary
        n_episodes: Number of episodes to run
        verbose: Print progress
    
    Returns:
        Dictionary with aggregated metrics
    """
    all_metrics = []
    
    for ep in range(n_episodes):
        env = make_env(cfg, seed=100 + ep)
        
        # Create algorithm adapter (if not random)
        if algorithm_name != 'random':
            adapter = TraditionalAlgorithmAdapter(
                algorithm_name, 
                env.N, 
                env.M, 
                env.capacity
            )
        
        # Reset tracking
        tracker = PassengerTracker()
        metrics = EpisodeMetrics()
        
        state = env.reset()
        done = False
        
        while not done:
            # Select action
            if algorithm_name == 'random':
                mask = env.action_mask()
                action = np.zeros(env.M, dtype=int)
                for i in range(env.M):
                    legal = [a for a in range(4) if mask[i, a]]
                    action[i] = np.random.choice(legal) if legal else 0
            else:
                action = adapter.select_action(state, env)
            
            # Step environment
            state, reward, done, info = env.step(action)
            
            # Track metrics
            metrics.episode_return += reward
            metrics.steps += 1
            metrics.total_wait_steps += info['n_waiting']
            metrics.total_incar_steps += info['n_incar']
            metrics.passengers_generated += info['arrivals']
            
            # Track passenger journeys
            tracker.register_arrivals(info['arrivals'], env.t)
            if info['boarded'] > 0:
                tracker.register_boarding(info['boarded'], env.t)
            if info['alighted'] > 0:
                tracker.register_alighting(info['alighted'], env.t)
                metrics.passengers_served += info['alighted']
        
        # Collect episode metrics
        metrics.waiting_times = tracker.completed_waiting
        metrics.journey_times = tracker.completed_journey
        all_metrics.append(metrics)
        
        if verbose:
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"Return={metrics.episode_return:.1f}, "
                  f"AWT={metrics.avg_waiting_time:.2f}, "
                  f"AJT={metrics.avg_journey_time:.2f}, "
                  f"Served={metrics.passengers_served}")
    
    # Aggregate results
    results = {
        'algorithm': algorithm_name,
        'n_episodes': n_episodes,
        'mean_return': np.mean([m.episode_return for m in all_metrics]),
        'std_return': np.std([m.episode_return for m in all_metrics]),
        'mean_awt': np.mean([m.avg_waiting_time for m in all_metrics]),
        'std_awt': np.std([m.avg_waiting_time for m in all_metrics]),
        'mean_ajt': np.mean([m.avg_journey_time for m in all_metrics]),
        'std_ajt': np.std([m.avg_journey_time for m in all_metrics]),
        'mean_system_time': np.mean([m.avg_system_time for m in all_metrics]),
        'mean_waiting_queue': np.mean([m.avg_waiting_queue for m in all_metrics]),
        'mean_incar_count': np.mean([m.avg_incar_count for m in all_metrics]),
        'mean_passengers_served': np.mean([m.passengers_served for m in all_metrics]),
        'mean_passengers_generated': np.mean([m.passengers_generated for m in all_metrics]),
        'service_rate': np.mean([m.passengers_served / max(1, m.passengers_generated) 
                                for m in all_metrics]),
    }
    
    return results


def print_results_table(results_list: List[Dict[str, Any]]):
    """Print results in a formatted table."""
    print("\n" + "=" * 120)
    print("TRADITIONAL EGCS ALGORITHMS - PERFORMANCE COMPARISON")
    print("=" * 120)
    
    # Header
    print(f"{'Algorithm':<20} {'Return':>12} {'AWT':>10} {'AJT':>10} "
          f"{'System':>10} {'Queue':>10} {'InCar':>10} {'Served':>10} {'Rate':>8}")
    print("-" * 120)
    
    # Data rows
    for res in results_list:
        print(f"{res['algorithm']:<20} "
              f"{res['mean_return']:>12.2f} "
              f"{res['mean_awt']:>10.2f} "
              f"{res['mean_ajt']:>10.2f} "
              f"{res['mean_system_time']:>10.2f} "
              f"{res['mean_waiting_queue']:>10.2f} "
              f"{res['mean_incar_count']:>10.2f} "
              f"{res['mean_passengers_served']:>10.1f} "
              f"{res['service_rate']:>8.2%}")
    
    print("=" * 120)
    
    # Find best performers
    best_return = max(results_list, key=lambda x: x['mean_return'])
    best_awt = min(results_list, key=lambda x: x['mean_awt'])
    best_ajt = min(results_list, key=lambda x: x['mean_ajt'])
    
    print(f"\n🏆 Best Return:      {best_return['algorithm']:<20} ({best_return['mean_return']:.2f})")
    print(f"🏆 Best AWT:         {best_awt['algorithm']:<20} ({best_awt['mean_awt']:.2f} steps)")
    print(f"🏆 Best AJT:         {best_ajt['algorithm']:<20} ({best_ajt['mean_ajt']:.2f} steps)")
    print("=" * 120)
    
    # Detailed statistics
    print("\nDETAILED STATISTICS:")
    print("-" * 120)
    for res in results_list:
        print(f"\n{res['algorithm'].upper()}:")
        print(f"  Return:           {res['mean_return']:.2f} ± {res['std_return']:.2f}")
        print(f"  Avg Wait Time:    {res['mean_awt']:.2f} ± {res['std_awt']:.2f} steps")
        print(f"  Avg Journey Time: {res['mean_ajt']:.2f} ± {res['std_ajt']:.2f} steps")
        print(f"  Avg System Time:  {res['mean_system_time']:.2f} steps")
        print(f"  Passengers:       {res['mean_passengers_served']:.1f} / {res['mean_passengers_generated']:.1f} "
              f"({res['service_rate']:.2%})")


def save_baseline_metrics(results_list: List[Dict[str, Any]], config_name: str):
    """Save baseline metrics for adaptive reward calculation."""
    output_dir = 'dsp2/baselines'
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate baseline statistics - convert all numpy types to Python native types
    baseline = {
        'config': config_name,
        'algorithms': {},
        'aggregate': {
            'mean_awt': float(np.mean([r['mean_awt'] for r in results_list])),
            'min_awt': float(min([r['mean_awt'] for r in results_list])),
            'max_awt': float(max([r['mean_awt'] for r in results_list])),
            'mean_ajt': float(np.mean([r['mean_ajt'] for r in results_list])),
            'min_ajt': float(min([r['mean_ajt'] for r in results_list])),
            'max_ajt': float(max([r['mean_ajt'] for r in results_list])),
            'mean_return': float(np.mean([r['mean_return'] for r in results_list])),
        }
    }
    
    for res in results_list:
        baseline['algorithms'][res['algorithm']] = {
            'mean_awt': float(res['mean_awt']),
            'std_awt': float(res['std_awt']),
            'mean_ajt': float(res['mean_ajt']),
            'std_ajt': float(res['std_ajt']),
            'mean_return': float(res['mean_return']),
            'std_return': float(res['std_return']),
        }
    
    # Save to YAML
    output_path = os.path.join(output_dir, f'{config_name}_baseline.yaml')
    with open(output_path, 'w') as f:
        yaml.dump(baseline, f, default_flow_style=False)
    
    print(f"\n✅ Baseline metrics saved to: {output_path}")
    print(f"\n📊 Aggregate Baseline Metrics:")
    print(f"  Mean AWT across all algorithms: {baseline['aggregate']['mean_awt']:.2f} steps")
    print(f"  Best AWT: {baseline['aggregate']['min_awt']:.2f} steps")
    print(f"  Worst AWT: {baseline['aggregate']['max_awt']:.2f} steps")
    print(f"  Mean AJT across all algorithms: {baseline['aggregate']['mean_ajt']:.2f} steps")


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Traditional EGCS Algorithms')
    parser.add_argument('--config', type=str, default='simple',
                       help='Config name (simple, mini, quick, default, improved)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes per algorithm')
    parser.add_argument('--algorithms', type=str, nargs='+',
                       default=['random', 'collective', 'nearest_car', 'sectoring', 'fixed_priority'],
                       help='Algorithms to evaluate')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = os.path.join('dsp2', 'configs', f'{args.config}.yaml')
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return
    
    cfg = load_config(config_path)
    
    print("=" * 120)
    print(f"EVALUATING TRADITIONAL EGCS ALGORITHMS - {args.config.upper()} CONFIG")
    print("=" * 120)
    print(f"Environment: {cfg['n_floors']} floors, {cfg['m_elevators']} elevators, "
          f"capacity={cfg['capacity']}")
    print(f"Traffic: λ={cfg['lambda']}, t_max={cfg['t_max']}s")
    print(f"Episodes: {args.episodes} per algorithm")
    print("=" * 120)
    
    # Evaluate each algorithm
    results = []
    for algo in args.algorithms:
        print(f"\n📊 Evaluating {algo.upper()}...")
        try:
            result = evaluate_algorithm(algo, cfg, n_episodes=args.episodes, verbose=True)
            results.append(result)
        except Exception as e:
            print(f"❌ Error evaluating {algo}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison table
    if results:
        print_results_table(results)
        
        # Save baseline metrics
        save_baseline_metrics(results, args.config)
    else:
        print("\n❌ No results to display")


if __name__ == "__main__":
    main()
