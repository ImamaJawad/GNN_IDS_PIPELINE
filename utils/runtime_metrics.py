"""
Runtime Metrics Module
Handles timing, throughput, and cost estimation for GNN models
"""
import time
import torch

class RuntimeTracker:
    def __init__(self, gpu_cost_per_hour=0.45):
        self.start_train = None
        self.end_train = None
        self.start_infer = None
        self.end_infer = None
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.metrics_history = []
    
    def start_training_timer(self):
        """Start training time tracking"""
        self.start_train = time.time()
        print("🏁 Training timer started...")
    
    def end_training_timer(self):
        """End training time tracking"""
        self.end_train = time.time()
        print("✅ Training timer stopped...")
    
    def start_inference_timer(self):
        """Start inference time tracking"""
        self.start_infer = time.time()
        print("🏁 Inference timer started...")
    
    def end_inference_timer(self):
        """End inference time tracking"""
        self.end_infer = time.time()
        print("✅ Inference timer stopped...")
    
    def calculate_runtime_metrics(self, total_samples=None, results_dict=None, model_name="Model"):
        """
        Calculate and print comprehensive runtime metrics
        
        Args:
            total_samples: Total number of samples processed (if known)
            results_dict: Dictionary containing evaluation results (must have 'threshold' key)
            model_name: Name of the model for display purposes
        """
        if self.start_train is None or self.end_train is None:
            print("❌ Training times not recorded!")
            return None
        
        if self.start_infer is None or self.end_infer is None:
            print("❌ Inference times not recorded!")
            return None
        
        # Calculate durations
        train_duration = self.end_train - self.start_train
        infer_duration = self.end_infer - self.start_infer
        
        # Calculate throughput if samples provided
        throughput = total_samples / infer_duration if total_samples and infer_duration > 0 else 0
        
        # Calculate GPU cost
        total_gpu_time = (train_duration + infer_duration) / 3600  # Convert to hours
        gpu_cost = total_gpu_time * self.gpu_cost_per_hour
        
        # Create metrics dict
        metrics = {
            'model_name': model_name,
            'train_duration': train_duration,
            'infer_duration': infer_duration,
            'total_duration': train_duration + infer_duration,
            'throughput': throughput,
            'gpu_cost': gpu_cost,
            'total_samples': total_samples or 0,
            'timestamp': time.time()
        }
        
        # Print metrics
        print(f"\n--- Runtime Metrics for {model_name} ---")
        print(f"Training Time: {train_duration:.2f}s ({train_duration/60:.2f} minutes)")
        print(f"Inference Time: {infer_duration:.2f}s")
        print(f"Total Runtime: {train_duration + infer_duration:.2f}s")
        if throughput > 0:
            print(f"Throughput: {throughput:.2f} samples/sec")
        print(f"Estimated GPU Cost: ${gpu_cost:.4f}")
        
        if results_dict and 'threshold' in results_dict:
            print(f"Selected threshold: {results_dict['threshold']:.6f}")
            metrics['threshold'] = results_dict['threshold']
        
        # Store in history
        self.metrics_history.append(metrics)
        
        return metrics
    
    def reset_timers(self):
        """Reset all timers for new experiment"""
        self.start_train = None
        self.end_train = None
        self.start_infer = None
        self.end_infer = None
    
    def get_summary_report(self):
        """Generate summary report of all experiments"""
        if not self.metrics_history:
            print("No metrics recorded yet.")
            return
        
        print(f"\n{'='*80}")
        print("RUNTIME METRICS SUMMARY")
        print(f"{'='*80}")
        
        total_cost = sum(m['gpu_cost'] for m in self.metrics_history)
        total_time = sum(m['total_duration'] for m in self.metrics_history)
        
        print(f"Total Experiments: {len(self.metrics_history)}")
        print(f"Total Runtime: {total_time/60:.2f} minutes")
        print(f"Total GPU Cost: ${total_cost:.4f}")
        print(f"Average Cost per Experiment: ${total_cost/len(self.metrics_history):.4f}")
        
        print(f"\nPer-Dataset Breakdown:")
        for i, metrics in enumerate(self.metrics_history, 1):
            print(f"  {i}. {metrics['model_name']}: {metrics['total_duration']:.1f}s, ${metrics['gpu_cost']:.4f}")


# Standalone function version (for direct integration)
def calculate_and_print_runtime_metrics(start_train, end_train, start_infer, end_infer, 
                                       total_samples=None, results=None, model_name="Model"):
    """
    Standalone function to calculate runtime metrics from timestamps
    
    Args:
        start_train: Training start timestamp
        end_train: Training end timestamp  
        start_infer: Inference start timestamp
        end_infer: Inference end timestamp
        total_samples: Total number of samples processed 
        results: Results dictionary with threshold info
        model_name: Name for display
    """
    train_duration = end_train - start_train
    infer_duration = end_infer - start_infer
    
    # Calculate metrics
    throughput = total_samples / infer_duration if total_samples and infer_duration > 0 else 0
    gpu_cost = ((train_duration + infer_duration) / 3600) * 0.45
    
    print(f"\n--- Runtime Metrics for {model_name} ---")
    print(f"Training Time: {train_duration:.2f}s")
    print(f"Inference Time: {infer_duration:.2f}s")
    if throughput > 0:
        print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Estimated GPU Cost: ${gpu_cost:.4f}")
    
    if results and 'threshold' in results:
        print(f"Selected threshold: {results['threshold']:.6f}")
    
    return {
        'train_duration': train_duration,
        'infer_duration': infer_duration,
        'throughput': throughput,
        'gpu_cost': gpu_cost,
        'total_samples': total_samples or 0
    }


def extract_runtime_from_scalers(scalers):
    """
    Helper function to extract timing info from scalers dict
    
    Args:
        scalers: Dictionary returned from training (contains 'timing' key)
    
    Returns:
        Dictionary with timing info
    """
    if 'timing' not in scalers:
        return None
    
    return scalers['timing']