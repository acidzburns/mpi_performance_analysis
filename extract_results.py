import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import os

def extract_timing(filename):
    """Extract execution time from result file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        match = re.search(r'Total Execution Time: ([\d.]+) seconds', content)
        if match:
            return float(match.group(1))
        
        match = re.search(r'Total MPI Execution Time: ([\d.]+) seconds', content)
        if match:
            return float(match.group(1))
            
        return None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def extract_mpi_timing(filename):
    """Extract MPI timing details"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        total_time = None
        transfer_time = None
        comp_time = None
        
        match = re.search(r'Total MPI Execution Time: ([\d.]+) seconds', content)
        if match:
            total_time = float(match.group(1))
        
        match = re.search(r'Data Transfer: ([\d.]+) seconds', content)
        if match:
            transfer_time = float(match.group(1))
        
        match = re.search(r'Computation: ([\d.]+) seconds', content)
        if match:
            comp_time = float(match.group(1))
            
        return total_time, transfer_time, comp_time
    except:
        return None, None, None

def main():
    print("=" * 60)
    print("MPI PERFORMANCE RESULTS EXTRACTOR")
    print("=" * 60)
    print()
    
    # Check if results folder exists
    if not os.path.exists('results'):
        print("ERROR: results folder not found!")
        print("Please run run_all_tests.bat first.")
        return
    
    # Extract sequential time
    seq_time = extract_timing('results/sequential_results.txt')
    
    # Extract MPI times
    mpi_results = []
    nodes = [2, 4, 6, 8]
    
    for n in nodes:
        total, transfer, comp = extract_mpi_timing(f'results/mpi_{n}_nodes.txt')
        if total:
            mpi_results.append({
                'Nodes': n,
                'MPI_Time': total,
                'Transfer_Time': transfer or 0,
                'Comp_Time': comp or 0
            })
    
    if seq_time is None:
        print("ERROR: Could not find sequential timing")
        print("Make sure sequential_results.txt exists in results folder")
        return
    
    print(f"Sequential Execution Time: {seq_time:.6f} seconds")
    print()
    
    # Create comparison table
    print("=" * 80)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Nodes':<10} {'Seq Time (s)':<15} {'MPI Time (s)':<15} {'Speedup':<12} {'Efficiency (%)':<15} {'Reduction (%)'}")
    print("-" * 80)
    
    results_data = []
    
    for result in mpi_results:
        n = result['Nodes']
        mpi_time = result['MPI_Time']
        speedup = seq_time / mpi_time
        efficiency = (speedup / n) * 100
        reduction = (1 - mpi_time / seq_time) * 100
        
        print(f"{n:<10} {seq_time:<15.6f} {mpi_time:<15.6f} {speedup:<12.2f} {efficiency:<15.1f} {reduction:<10.1f}")
        
        results_data.append({
            'Dataset Size': '10M',
            'Nodes': n,
            'Sequential Time (s)': seq_time,
            'MPI Time (s)': mpi_time,
            'Speedup': speedup,
            'Efficiency (%)': efficiency,
            'Time Reduction (%)': reduction,
            'Transfer Time (s)': result['Transfer_Time'],
            'Compute Time (s)': result['Comp_Time']
        })
    
    # Save to CSV
    df = pd.DataFrame(results_data)
    df.to_csv('results/performance_summary.csv', index=False)
    print()
    print("✓ Results saved to results/performance_summary.csv")
    
    # Create visualizations
    print()
    print("Generating performance charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Chart 1: Time Comparison
    ax = axes[0, 0]
    x = np.arange(len(nodes))
    width = 0.35
    bars1 = ax.bar(x - width/2, [seq_time] * len(nodes), width, label='Sequential', color='navy', alpha=0.8)
    bars2 = ax.bar(x + width/2, [r['MPI_Time'] for r in mpi_results], width, label='MPI', color='coral', alpha=0.8)
    ax.set_xlabel('Number of Virtual Nodes')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Execution Time Comparison\n(10 Million Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # Chart 2: Speedup
    ax = axes[0, 1]
    speedups = [seq_time / r['MPI_Time'] for r in mpi_results]
    ax.bar(nodes, speedups, color='steelblue', alpha=0.7)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No Speedup')
    ax.plot(nodes, nodes, 'g--', alpha=0.5, label='Ideal Speedup')
    ax.set_xlabel('Number of Virtual Nodes')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup vs Virtual Nodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, v in enumerate(speedups):
        ax.text(nodes[i], v + 0.1, f'{v:.2f}x', ha='center', va='bottom', fontsize=10)
    
    # Chart 3: Efficiency
    ax = axes[1, 0]
    efficiencies = [(speedup/n)*100 for speedup, n in zip(speedups, nodes)]
    bars = ax.bar(nodes, efficiencies, color='teal', alpha=0.7)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% Efficiency')
    ax.set_xlabel('Number of Virtual Nodes')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Parallel Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, v in enumerate(efficiencies):
        ax.text(nodes[i], v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Chart 4: Time Reduction
    ax = axes[1, 1]
    reduction = [(1 - r['MPI_Time']/seq_time) * 100 for r in mpi_results]
    bars = ax.bar(nodes, reduction, color='purple', alpha=0.7)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Number of Virtual Nodes')
    ax.set_ylabel('Time Reduction (%)')
    ax.set_title('Time Reduction vs Sequential')
    ax.grid(True, alpha=0.3)
    
    for i, v in enumerate(reduction):
        ax.text(nodes[i], v + 1, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('MPI Performance Analysis - 10 Million Dataset\nVirtual Nodes vs Sequential', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('results/performance_charts.png', dpi=300, bbox_inches='tight')
    print("✓ Charts saved to results/performance_charts.png")
    
    # Detailed breakdown chart
    fig2, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(nodes))
    width = 0.6
    
    transfer_times = [r['Transfer_Time'] for r in mpi_results]
    comp_times = [r['Comp_Time'] for r in mpi_results]
    
    ax.bar(x, comp_times, width, label='Computation', color='steelblue', alpha=0.8)
    ax.bar(x, transfer_times, width, bottom=comp_times, label='Data Transfer', color='coral', alpha=0.8)
    
    ax.set_xlabel('Number of Virtual Nodes')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('MPI Execution Time Breakdown\n(10 Million Dataset)')
    ax.set_xticks(x)
    ax.set_xticklabels(nodes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    for i, r in enumerate(mpi_results):
        total = r['MPI_Time']
        ax.text(i, total + 0.5, f'{total:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/mpi_breakdown.png', dpi=300, bbox_inches='tight')
    print("✓ Breakdown chart saved to results/mpi_breakdown.png")
    
    # Print summary
    print()
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    best_idx = np.argmax(speedups)
    best_nodes = nodes[best_idx]
    best_speedup = speedups[best_idx]
    best_efficiency = efficiencies[best_idx]
    best_reduction = reduction[best_idx]
    
    print(f"\nBest Speedup: {best_speedup:.2f}x with {best_nodes} nodes")
    print(f"Best Efficiency: {best_efficiency:.1f}% with {best_nodes} nodes")
    print(f"Maximum Time Reduction: {best_reduction:.1f}% with {best_nodes} nodes")
    
    # Amdahl's Law analysis
    print("\n" + "=" * 60)
    print("AMDAHL'S LAW ANALYSIS")
    print("=" * 60)
    
    f_parallel = (best_speedup - 1) / (best_speedup / best_nodes - 1) if best_speedup / best_nodes > 1 else 0.95
    f_parallel = max(0.5, min(1.0, f_parallel))
    
    print(f"\nEstimated Parallel Fraction: {f_parallel:.2f}")
    print("\nTheoretical Speedup with Amdahl's Law:")
    print(f"{'Nodes':<10} {'Theoretical':<15} {'Actual':<15} {'Achieved (%)'}")
    print("-" * 50)
    
    for n, actual in zip(nodes, speedups):
        theoretical = 1 / ((1 - f_parallel) + (f_parallel / n))
        achieved = (actual / theoretical) * 100
        print(f"{n:<10} {theoretical:<15.2f} {actual:<15.2f} {achieved:<10.1f}")
    
    print("\n" + "=" * 60)
    print("COMPLETE! Check results folder for all outputs.")
    print("=" * 60)

if __name__ == "__main__":
    main()