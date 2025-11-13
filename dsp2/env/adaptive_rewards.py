"""
Adaptive reward calculation based on traditional algorithm baselines
"""
from __future__ import annotations
import os
import yaml
import numpy as np
from typing import Dict, Any, Optional, Tuple


class AdaptiveRewardCalculator:
    """
    Calculate adaptive rewards based on comparison with traditional algorithms.
    
    Implements multiple reward adaptation strategies:
    1. Dynamic Baseline Thresholding
    2. Comparative Reward Shaping
    3. Multi-Metric Performance Index
    4. Algorithm-Specific Competitive Rewards
    5. Staged Difficulty Training
    """
    
    def __init__(self, config_name: str = 'simple', baseline_weight: float = 0.5,
                 performance_bonus_scale: float = 1.0, 
                 comparative_penalty_scale: float = 2.0,
                 curriculum_stage: int = 0):
        """
        Initialize adaptive reward calculator.
        
        Args:
            config_name: Which baseline config to load (simple, mini, default)
            baseline_weight: Weight for baseline comparison rewards (0-1)
            performance_bonus_scale: Multiplier for performance bonuses
            comparative_penalty_scale: Multiplier for comparative penalties
            curriculum_stage: Current curriculum stage (0=random, 1=collective, 2=nearest, 3=sectoring)
        """
        self.config_name = config_name
        self.baseline_weight = baseline_weight
        self.performance_bonus_scale = performance_bonus_scale
        self.comparative_penalty_scale = comparative_penalty_scale
        self.curriculum_stage = curriculum_stage
        
        # Load baseline metrics
        self.baseline_metrics = self._load_baseline_metrics(config_name)
        
        # Extract key thresholds from baselines
        if self.baseline_metrics:
            self._setup_thresholds()
        else:
            # Fallback defaults if baseline not found
            self._setup_default_thresholds()
    
    def _load_baseline_metrics(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load baseline metrics from YAML file."""
        baseline_path = os.path.join('dsp2', 'baselines', f'{config_name}_baseline.yaml')
        
        if not os.path.exists(baseline_path):
            print(f"Warning: Baseline metrics not found at {baseline_path}")
            return None
        
        try:
            with open(baseline_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading baseline metrics: {e}")
            return None
    
    def _setup_thresholds(self):
        """Setup performance thresholds from loaded baselines."""
        algos = self.baseline_metrics['algorithms']
        agg = self.baseline_metrics['aggregate']
        
        # Extract AWT thresholds (convert from numpy scalars if needed)
        self.awt_best = float(agg['min_awt'])
        self.awt_worst = float(agg['max_awt'])
        self.awt_mean = float(agg['mean_awt'])
        
        # Extract AJT thresholds
        self.ajt_best = float(agg['min_ajt'])
        self.ajt_worst = float(agg['max_ajt'])
        self.ajt_mean = float(agg['mean_ajt'])
        
        # Algorithm-specific targets
        self.algo_targets = {
            'random': {
                'awt': float(algos['random']['mean_awt']),
                'ajt': float(algos['random']['mean_ajt']),
                'return': float(algos['random']['mean_return'])
            },
            'collective': {
                'awt': float(algos['collective']['mean_awt']),
                'ajt': float(algos['collective']['mean_ajt']),
                'return': float(algos['collective']['mean_return'])
            },
            'nearest_car': {
                'awt': float(algos['nearest_car']['mean_awt']),
                'ajt': float(algos['nearest_car']['mean_ajt']),
                'return': float(algos['nearest_car']['mean_return'])
            },
            'sectoring': {
                'awt': float(algos['sectoring']['mean_awt']),
                'ajt': float(algos['sectoring']['mean_ajt']),
                'return': float(algos['sectoring']['mean_return'])
            },
            'fixed_priority': {
                'awt': float(algos['fixed_priority']['mean_awt']),
                'ajt': float(algos['fixed_priority']['mean_ajt']),
                'return': float(algos['fixed_priority']['mean_return'])
            }
        }
        
        # Performance tiers (based on traditional algorithms)
        self.performance_tiers = {
            'excellent': self.awt_best,  # Beat best algorithm
            'good': (self.awt_best + self.awt_mean) / 2,  # Between best and mean
            'average': self.awt_mean,  # Around mean performance
            'poor': self.awt_worst  # Approaching worst
        }
        
        # Curriculum stage targets
        self.curriculum_targets = [
            ('random', self.algo_targets['random']['awt']),
            ('collective', self.algo_targets['collective']['awt']),
            ('nearest_car', self.algo_targets['nearest_car']['awt']),
            ('sectoring', self.algo_targets['sectoring']['awt'])
        ]
    
    def _setup_default_thresholds(self):
        """Setup default thresholds when baselines not available."""
        print("Warning: Using default thresholds (baselines not loaded)")
        self.awt_best = 2.0
        self.awt_worst = 30.0
        self.awt_mean = 10.0
        self.ajt_best = 3.0
        self.ajt_worst = 70.0
        self.ajt_mean = 20.0
        
        self.performance_tiers = {
            'excellent': 2.0,
            'good': 5.0,
            'average': 10.0,
            'poor': 30.0
        }
        
        self.curriculum_targets = [
            ('random', 30.0),
            ('collective', 5.0),
            ('nearest_car', 3.5),
            ('sectoring', 2.5)
        ]
        
        self.algo_targets = {}
    
    def get_performance_tier(self, awt: float) -> str:
        """Determine performance tier based on AWT."""
        if awt <= self.performance_tiers['excellent']:
            return 'excellent'
        elif awt <= self.performance_tiers['good']:
            return 'good'
        elif awt <= self.performance_tiers['average']:
            return 'average'
        else:
            return 'poor'
    
    def calculate_comparative_reward(self, current_awt: float, current_ajt: float,
                                    n_waiting: int, n_incar: int) -> float:
        """
        Strategy 2: Comparative Reward Shaping
        Adjust reward based on performance relative to baselines.
        """
        if current_awt == 0:  # No data yet
            return 0.0
        
        # Calculate performance ratio vs mean traditional algorithm
        awt_ratio = current_awt / max(0.1, self.awt_mean)
        ajt_ratio = current_ajt / max(0.1, self.ajt_mean)
        
        # Use tanh to bound the bonus/penalty to reasonable range
        awt_bonus = np.tanh(1.0 - awt_ratio) * self.performance_bonus_scale
        ajt_bonus = np.tanh(1.0 - ajt_ratio) * self.performance_bonus_scale * 0.5  # AJT less important
        
        total_bonus = (awt_bonus + ajt_bonus) * self.baseline_weight
        
        # Clip to prevent extreme values
        return float(np.clip(total_bonus, -2.0, 2.0))
    
    def calculate_performance_tier_bonus(self, awt: float) -> float:
        """
        Strategy 1: Dynamic Baseline Thresholding
        Provide graduated bonuses based on performance tier.
        """
        tier = self.get_performance_tier(awt)
        
        tier_bonuses = {
            'excellent': 2.0,  # Beating best traditional algorithm
            'good': 1.0,       # Above average
            'average': 0.0,    # Neutral
            'poor': -1.0       # Below average
        }
        
        return tier_bonuses[tier] * self.performance_bonus_scale
    
    def calculate_curriculum_bonus(self, awt: float) -> float:
        """
        Strategy 5: Staged Difficulty Training
        Provide bonus based on current curriculum stage target.
        """
        if self.curriculum_stage >= len(self.curriculum_targets):
            return 0.0
        
        stage_name, stage_target = self.curriculum_targets[self.curriculum_stage]
        
        # Bonus for beating current stage target
        if awt <= stage_target:
            # Scaled bonus - better performance = higher bonus
            improvement = (stage_target - awt) / max(0.1, stage_target)
            # Clip bonus to prevent extreme values
            return min(improvement, 1.0) * self.performance_bonus_scale
        else:
            # Penalty for not meeting target - HEAVILY CLIPPED
            # Use gentler penalty that doesn't explode with poor performance
            deficit_ratio = (awt - stage_target) / max(0.1, stage_target)
            # Clip to max penalty of -1.0 regardless of how bad performance is
            penalty = -min(deficit_ratio * 0.5, 1.0) * self.comparative_penalty_scale
            return penalty
    
    def calculate_algorithm_competitive_bonus(self, awt: float, ajt: float) -> float:
        """
        Strategy 4: Algorithm-Specific Competitive Rewards
        Compare against specific algorithms.
        """
        if not self.algo_targets:
            return 0.0
        
        # Count how many algorithms we beat
        algorithms_beaten = 0
        total_algorithms = len(self.algo_targets)
        
        for algo_name, targets in self.algo_targets.items():
            if awt <= targets['awt']:
                algorithms_beaten += 1
        
        # Bonus for beating multiple algorithms
        beat_ratio = algorithms_beaten / total_algorithms
        
        if beat_ratio == 1.0:
            return 3.0 * self.performance_bonus_scale  # Beat all!
        elif beat_ratio >= 0.75:
            return 2.0 * self.performance_bonus_scale  # Beat most
        elif beat_ratio >= 0.5:
            return 1.0 * self.performance_bonus_scale  # Beat half
        elif beat_ratio >= 0.25:
            return 0.5 * self.performance_bonus_scale  # Beat some
        else:
            return 0.0  # Beat none or very few
    
    def calculate_multi_metric_index(self, awt: float, ajt: float, 
                                    service_rate: float) -> float:
        """
        Strategy 3: Multi-Metric Performance Index
        Combine multiple metrics into single performance score.
        """
        # Target metrics from best traditional algorithms
        target_service_rate = 0.99  # 99% from baselines
        
        # Normalize each metric with HEAVY CLIPPING to prevent explosion
        # Use sigmoid-like scaling instead of raw ratios
        awt_ratio = awt / max(0.1, self.awt_mean)  # Compare to mean, not best
        ajt_ratio = ajt / max(0.1, self.ajt_mean)
        
        # Sigmoid-like scoring: good performance = positive, bad = negative, but bounded
        # Score range: approximately -1 to +1
        awt_score = np.tanh(1.0 - awt_ratio)  # tanh bounds to [-1, 1]
        ajt_score = np.tanh(1.0 - ajt_ratio)
        service_score = (service_rate - target_service_rate)  # Already bounded [-1, 1]
        
        # Weighted combination with smaller coefficients
        combined_score = (awt_score * 0.2 + ajt_score * 0.1 + service_score * 0.05)
        
        # Final clipping to ensure bounded output
        return np.clip(combined_score * self.performance_bonus_scale, -1.0, 1.0)
    
    def get_dynamic_penalty_weights(self, awt: float) -> Tuple[float, float]:
        """
        Adjust penalty weights based on current performance.
        Returns: (w_wait, w_incar) adjusted weights
        """
        # If performing well, reduce penalties (trust the agent)
        # If performing poorly, increase penalties (guide more strongly)
        
        if awt <= self.performance_tiers['excellent']:
            # Excellent performance - minimal penalties
            return (0.5, 0.1)
        elif awt <= self.performance_tiers['good']:
            # Good performance - standard penalties
            return (1.0, 0.2)
        elif awt <= self.performance_tiers['average']:
            # Average performance - increased penalties
            return (1.5, 0.3)
        else:
            # Poor performance - strong penalties to guide learning
            return (2.0, 0.5)
    
    def calculate_total_adaptive_reward(self, base_reward: float, awt: float, 
                                       ajt: float, n_waiting: int, n_incar: int,
                                       service_rate: float = 1.0) -> float:
        """
        Calculate total adaptive reward combining all strategies.
        
        Args:
            base_reward: Original environment reward
            awt: Current average waiting time
            ajt: Current average journey time
            n_waiting: Number waiting
            n_incar: Number in car
            service_rate: Percentage of passengers served
        
        Returns:
            Total reward with adaptive bonuses/penalties
        """
        if awt == 0:  # No metrics yet
            return base_reward
        
        # Apply all strategies with individual bounds
        comparative = self.calculate_comparative_reward(awt, ajt, n_waiting, n_incar)
        tier_bonus = self.calculate_performance_tier_bonus(awt)
        curriculum = self.calculate_curriculum_bonus(awt)
        competitive = self.calculate_algorithm_competitive_bonus(awt, ajt)
        multi_metric = self.calculate_multi_metric_index(awt, ajt, service_rate)
        
        # Combine with REDUCED weights to prevent accumulation
        total_adaptive = (
            comparative * 0.3 +      # Reduced from 1.0
            tier_bonus * 0.2 +       # Reduced from 0.5
            curriculum * 0.3 +       # Reduced from 1.0
            competitive * 0.1 +      # Reduced from 0.3
            multi_metric * 0.1       # Reduced from 0.2
        )
        
        # CRITICAL: Clip total adaptive bonus to prevent extreme rewards
        # This ensures adaptive rewards stay in reasonable range
        total_adaptive_clipped = np.clip(total_adaptive, -5.0, 5.0)
        
        final_reward = base_reward + total_adaptive_clipped
        
        # Final safety clip: prevent any single step reward from being extreme
        # Base rewards are typically in range [-1, 1], so allow adaptive to extend this
        return float(np.clip(final_reward, -10.0, 10.0))
    
    def get_info_dict(self, awt: float, ajt: float) -> Dict[str, Any]:
        """Get detailed information for logging."""
        tier = self.get_performance_tier(awt)
        
        info = {
            'performance_tier': tier,
            'awt_vs_best': awt / self.awt_best if self.awt_best > 0 else 0,
            'awt_vs_mean': awt / self.awt_mean if self.awt_mean > 0 else 0,
            'ajt_vs_best': ajt / self.ajt_best if self.ajt_best > 0 else 0,
            'curriculum_stage': self.curriculum_stage,
        }
        
        if self.curriculum_stage < len(self.curriculum_targets):
            stage_name, stage_target = self.curriculum_targets[self.curriculum_stage]
            info['curriculum_target'] = stage_target
            info['curriculum_target_name'] = stage_name
            info['meets_curriculum_target'] = awt <= stage_target
        
        return info
