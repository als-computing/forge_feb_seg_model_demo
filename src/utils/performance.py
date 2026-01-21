import logging
import time

logger = logging.getLogger(__name__)


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, name: str, rank: int = 0, enabled: bool = True):
        self.name = name
        self.rank = rank
        self.enabled = enabled and (rank == 0)
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        if self.enabled:
            self.start_time = time.time()
            logger.info(f"{self.name}...")
        return self
        
    def __exit__(self, *args):
        if self.enabled:
            self.end_time = time.time()
            elapsed = self.end_time - self.start_time
            logger.info(f"✓ {self.name} [{self._format_time(elapsed)}]")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds < 1:
            return f"{seconds*1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
    
    def elapsed(self) -> float:
        """Get elapsed time."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0


class PerformanceTracker:
    """Track performance metrics across the pipeline."""
    
    def __init__(self, rank: int = 0):
        self.rank = rank
        self.metrics = {}
        self.start_time = time.time()
        
    def record(self, name: str, duration: float):
        """Record a timing metric."""
        self.metrics[name] = duration
    
    def log_summary(self, world_size: int, num_images: int, num_patches: int, num_prompts: int):
        """Log comprehensive performance summary."""
        if self.rank != 0:
            return
            
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60, flush=True)
        print("Performance Summary:", flush=True)
        print("=" * 60, flush=True)
        
        # Breakdown
        print("\nTiming Breakdown:", flush=True)
        for name, duration in self.metrics.items():
            pct = (duration / total_time) * 100
            print(f"  {name:.<30} {self._fmt(duration)} ({pct:.1f}%)", flush=True)
        
        print(f"  {'─' * 30}", flush=True)
        print(f"  {'Total time':.<30} {self._fmt(total_time)}", flush=True)
        
        # Throughput
        print("\nThroughput:", flush=True)
        images_per_min = (num_images / total_time) * 60
        time_per_image = total_time / num_images
        time_per_patch = total_time / num_patches if num_patches > 0 else 0
        time_per_prompt_patch = time_per_patch / num_prompts if num_prompts > 0 else 0
        
        print(f"  Images/min:          {images_per_min:.2f}", flush=True)
        print(f"  Time per image:      {self._fmt(time_per_image)}", flush=True)
        print(f"  Time per patch:      {self._fmt(time_per_patch)}", flush=True)
        print(f"  Time per prompt/patch: {self._fmt(time_per_prompt_patch)}", flush=True)
        
        # GPU stats
        if world_size > 1:
            inference_time = self.metrics.get('inference', 0)
            if inference_time > 0:
                # Estimate single GPU time (approximate)
                estimated_single_gpu = inference_time * world_size
                speedup = estimated_single_gpu / total_time
                efficiency = (speedup / world_size) * 100
                
                print(f"\nScaling Performance:", flush=True)
                print(f"  GPUs used:           {world_size}", flush=True)
                print(f"  Speedup:             {speedup:.2f}x", flush=True)
                print(f"  Scaling efficiency:  {efficiency:.1f}%", flush=True)
                print(f"  Effective batch size: {world_size}", flush=True)  # batch_size=1 per GPU
        
        print("=" * 60 + "\n", flush=True)
    
    @staticmethod
    def _fmt(seconds: float) -> str:
        """Format seconds."""
        if seconds < 1:
            return f"{seconds*1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.1f}s"
