"""
Comparison script for P2PFL SVM vs Scikit-learn SVM
This script runs both algorithms and provides detailed comparison metrics.
"""

import time
import subprocess
import sys
import json
from datetime import datetime

def run_p2pfl_experiment(nodes=2, rounds=1, epochs=1, batch_size=16):
    """Run P2PFL experiment and capture metrics"""
    print("🚀 Running P2PFL SVM Experiment...")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run the P2PFL experiment
    cmd = [
        sys.executable, "mnist.py",
        "--nodes", str(nodes),
        "--rounds", str(rounds), 
        "--epochs", str(epochs),
        "--batch_size", str(batch_size)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        p2pfl_metrics = {
            'algorithm': 'P2PFL SVM',
            'nodes': nodes,
            'rounds': rounds,
            'epochs': epochs,
            'batch_size': batch_size,
            'total_time': end_time - start_time,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print("✅ P2PFL experiment completed successfully")
        else:
            print("❌ P2PFL experiment failed")
            print(f"Error: {result.stderr}")
            
        return p2pfl_metrics
        
    except subprocess.TimeoutExpired:
        print("⏰ P2PFL experiment timed out")
        return {
            'algorithm': 'P2PFL SVM',
            'success': False,
            'error': 'Timeout'
        }

def run_scikit_experiment():
    """Run Scikit-learn SVM experiment and capture metrics"""
    print("\n🚀 Running Scikit-learn SVM Experiment...")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, "primalSVMScikit.py"], 
                              capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        scikit_metrics = {
            'algorithm': 'Scikit-learn SVM',
            'total_time': end_time - start_time,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
        if result.returncode == 0:
            print("✅ Scikit-learn experiment completed successfully")
        else:
            print("❌ Scikit-learn experiment failed")
            print(f"Error: {result.stderr}")
            
        return scikit_metrics
        
    except subprocess.TimeoutExpired:
        print("⏰ Scikit-learn experiment timed out")
        return {
            'algorithm': 'Scikit-learn SVM',
            'success': False,
            'error': 'Timeout'
        }

def extract_metrics_from_output(output_text):
    """Extract key metrics from algorithm output"""
    metrics = {}
    
    # Extract timing information
    if "Total experiment time:" in output_text:
        for line in output_text.split('\n'):
            if "Total experiment time:" in line:
                time_str = line.split(":")[1].strip().split()[0]
                metrics['experiment_time'] = float(time_str)
                break
    
    # Extract accuracy information
    if "Test Accuracy:" in output_text:
        for line in output_text.split('\n'):
            if "Test Accuracy:" in line:
                acc_str = line.split(":")[1].strip().split()[0]
                metrics['test_accuracy'] = float(acc_str)
                break
                
    return metrics

def print_comparison_report(p2pfl_results, scikit_results):
    """Print detailed comparison report"""
    print("\n" + "="*60)
    print("📊 ALGORITHM COMPARISON REPORT")
    print("="*60)
    print(f"🕐 Comparison timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n🔬 P2PFL SVM Results:")
    print(f"  ✅ Success: {p2pfl_results['success']}")
    if p2pfl_results['success']:
        print(f"  ⏱️  Total Time: {p2pfl_results['total_time']:.2f} seconds")
        print(f"  🏗️  Configuration: {p2pfl_results['nodes']} nodes, {p2pfl_results['rounds']} rounds, {p2pfl_results['epochs']} epochs")
    else:
        print(f"  ❌ Error: {p2pfl_results.get('error', 'Unknown error')}")
    
    print(f"\n🔬 Scikit-learn SVM Results:")
    print(f"  ✅ Success: {scikit_results['success']}")
    if scikit_results['success']:
        print(f"  ⏱️  Total Time: {scikit_results['total_time']:.2f} seconds")
    else:
        print(f"  ❌ Error: {scikit_results.get('error', 'Unknown error')}")
    
    # Performance comparison
    if p2pfl_results['success'] and scikit_results['success']:
        print(f"\n⚡ Performance Comparison:")
        time_diff = p2pfl_results['total_time'] - scikit_results['total_time']
        if time_diff > 0:
            print(f"  🏃 Scikit-learn is {time_diff:.2f}s faster ({time_diff/p2pfl_results['total_time']*100:.1f}% faster)")
        else:
            print(f"  🏃 P2PFL is {abs(time_diff):.2f}s faster ({abs(time_diff)/scikit_results['total_time']*100:.1f}% faster)")
    
    print(f"\n📋 Detailed Results:")
    print(f"  P2PFL Output: {len(p2pfl_results['stdout'])} characters")
    print(f"  Scikit Output: {len(scikit_results['stdout'])} characters")
    
    # Save results to file
    results = {
        'timestamp': datetime.now().isoformat(),
        'p2pfl_results': p2pfl_results,
        'scikit_results': scikit_results
    }
    
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to 'comparison_results.json'")

def main():
    """Main comparison function"""
    print("🔬 SVM Algorithm Comparison Tool")
    print("=" * 50)
    print("This tool compares P2PFL SVM vs Scikit-learn SVM")
    print("on the same MNIST dataset with detailed metrics.")
    
    # Run P2PFL experiment
    p2pfl_results = run_p2pfl_experiment(nodes=2, rounds=1, epochs=1, batch_size=16)
    
    # Run Scikit-learn experiment  
    scikit_results = run_scikit_experiment()
    
    # Print comparison report
    print_comparison_report(p2pfl_results, scikit_results)
    
    print(f"\n🎉 Comparison completed!")
    print(f"📁 Check 'comparison_results.json' for detailed results")

if __name__ == "__main__":
    main()
