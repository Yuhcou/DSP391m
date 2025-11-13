"""
Real-time passenger tracking for AWT/AJT calculation during training
"""
from __future__ import annotations
from typing import List, Tuple
from collections import deque


class PassengerTracker:
    """
    Track individual passenger metrics in real-time during episodes.
    Used to calculate Average Waiting Time (AWT) and Average Journey Time (AJT).
    """
    
    def __init__(self):
        # Track passengers in different states
        self.waiting_passengers = {}  # {passenger_id: arrival_time}
        self.boarded_passengers = {}  # {passenger_id: (arrival_time, board_time)}
        
        # Completed metrics
        self.completed_waiting_times = []
        self.completed_journey_times = []
        
        # Running statistics
        self.total_passengers_arrived = 0
        self.total_passengers_boarded = 0
        self.total_passengers_served = 0
        
        self.next_passenger_id = 0
    
    def register_arrivals(self, n_arrivals: int, current_time: int):
        """Register new passenger arrivals."""
        for _ in range(n_arrivals):
            self.waiting_passengers[self.next_passenger_id] = current_time
            self.next_passenger_id += 1
            self.total_passengers_arrived += 1
    
    def register_boarding(self, n_boarded: int, current_time: int):
        """
        Register passengers boarding elevators.
        Assumes FIFO - oldest waiting passengers board first.
        """
        if n_boarded == 0:
            return
        
        # Take oldest waiting passengers (FIFO)
        boarded_ids = sorted(self.waiting_passengers.keys())[:n_boarded]
        
        for pid in boarded_ids:
            arrival_time = self.waiting_passengers.pop(pid)
            waiting_time = current_time - arrival_time
            
            # Store boarding info
            self.boarded_passengers[pid] = (arrival_time, current_time)
            
            # Record completed waiting time
            self.completed_waiting_times.append(waiting_time)
            self.total_passengers_boarded += 1
    
    def register_alighting(self, n_alighted: int, current_time: int):
        """
        Register passengers alighting (reaching destination).
        Assumes FIFO - oldest boarded passengers alight first.
        """
        if n_alighted == 0:
            return
        
        # Take oldest boarded passengers (FIFO)
        alighted_ids = sorted(self.boarded_passengers.keys())[:n_alighted]
        
        for pid in alighted_ids:
            arrival_time, board_time = self.boarded_passengers.pop(pid)
            journey_time = current_time - board_time
            
            # Record completed journey time
            self.completed_journey_times.append(journey_time)
            self.total_passengers_served += 1
    
    def get_current_awt(self) -> float:
        """Get current Average Waiting Time."""
        if not self.completed_waiting_times:
            return 0.0
        return sum(self.completed_waiting_times) / len(self.completed_waiting_times)
    
    def get_current_ajt(self) -> float:
        """Get current Average Journey Time."""
        if not self.completed_journey_times:
            return 0.0
        return sum(self.completed_journey_times) / len(self.completed_journey_times)
    
    def get_recent_awt(self, window: int = 100) -> float:
        """Get Average Waiting Time over recent window."""
        if not self.completed_waiting_times:
            return 0.0
        recent = self.completed_waiting_times[-window:]
        return sum(recent) / len(recent)
    
    def get_recent_ajt(self, window: int = 100) -> float:
        """Get Average Journey Time over recent window."""
        if not self.completed_journey_times:
            return 0.0
        recent = self.completed_journey_times[-window:]
        return sum(recent) / len(recent)
    
    def get_system_time(self) -> float:
        """Get average total system time (waiting + journey)."""
        return self.get_current_awt() + self.get_current_ajt()
    
    def get_statistics(self) -> dict:
        """Get all tracking statistics."""
        return {
            'awt': self.get_current_awt(),
            'ajt': self.get_current_ajt(),
            'system_time': self.get_system_time(),
            'total_arrived': self.total_passengers_arrived,
            'total_boarded': self.total_passengers_boarded,
            'total_served': self.total_passengers_served,
            'currently_waiting': len(self.waiting_passengers),
            'currently_in_car': len(self.boarded_passengers),
            'samples_awt': len(self.completed_waiting_times),
            'samples_ajt': len(self.completed_journey_times),
        }
    
    def reset(self):
        """Reset all tracking for new episode."""
        self.waiting_passengers.clear()
        self.boarded_passengers.clear()
        self.completed_waiting_times.clear()
        self.completed_journey_times.clear()
        self.total_passengers_arrived = 0
        self.total_passengers_boarded = 0
        self.total_passengers_served = 0
        self.next_passenger_id = 0
