from setup_experiments import SetupExperiments
import argparse
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run TPWGA(co) vs RWRGA(co) comparison experiments with fixed target functions'
    )
    parser.add_argument('-s', '--seed', default=0, type=int, help='Random seed (default: 0)')
    parser.add_argument('-n', '--num_exp', default=100, type=int, help='Number of experiments (default: 10)')
    parser.add_argument('-i', '--iterations', default=150, type=int, help='Iterations per run (default: 150)')
    parser.add_argument('--skip-comparison', action='store_true', help='Skip algorithm comparison plots')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip time analysis plots (deprecated)')
    parser.add_argument('--fast', action='store_true', help='Fast test mode (reduce exp/iter)')
    args = parser.parse_args()
    return args.seed, args.num_exp, args.iterations, args.skip_comparison, args.skip_analysis, args.fast


def print_core_info():
    print("=== Fixed Target Functions (5 total) ===")
    targets = [
        ("E1 (f¹)", "Log-Cosh loss, φ₁-φ₁₀, norm=1.0"),
        ("E2 (f²)", "Quadratic loss, φ₁₁-φ₂₀, norm=1.0"),
        ("E3 (f³)", "Huber loss (δ=0.5), φ₂₁-φ₃₀, norm=1.0"),
        ("E4 (f⁴)", "Gaussian-weighted quadratic, φ₃₁-φ₄₀, norm=1.0"),
        ("E5 (f⁵)", "Log-quadratic loss, φ₄₁-φ₅₀, norm=1.0")
    ]
    for name, desc in targets:
        print(f"  {name}: {desc}")
    
    print("\n=== Algorithm Comparison ===")
    print("  TPWGA(co): 1 run for 5 objectives (parallel, shared dict)")
    print("  RWRGA(co): 5 runs for 5 objectives (sequential, independent dict)")


if __name__ == '__main__':
    seed, num_exp, max_iterations, skip_comparison, skip_analysis, fast_mode = parse_arguments()
    
    if fast_mode:
        num_exp, max_iterations = min(num_exp, 5), min(max_iterations, 100)
        print(f"[Fast Mode] Reduced params: {num_exp} experiments, {max_iterations} iterations")
    
    print_core_info()
    
    print("\n=== Experiment Parameters ===")
    print(f"  Seed: {seed} | Experiments: {num_exp} | Iterations: {max_iterations}")
    print(f"  Max sparsity: 100 | Target functions: 5")
    print(f"  Dictionary size: 200 atoms | Dimension: 100")
    
    print("\n=== Running Experiments ===")
    exp = SetupExperiments(seed=seed)
    exp.run(num_exp=num_exp, max_iterations=max_iterations)
    
    if not skip_comparison:
        print("\n=== Generating Plots ===")
        print("  1. Error vs Iterations plot (5 targets)")
        exp.plot_algorithm_comparison()
        
    if not skip_analysis and not fast_mode:
        print("\nNote: Detailed time analysis plots are removed (as requested)")
    
    print("\n=== Key Statistics ===")
    if hasattr(exp, 'tpwga_times') and exp.tpwga_times:
        tpwga_avg = np.mean(exp.tpwga_times)
        rwrga_avg = np.mean(exp.rwrga_times)
        speedup = rwrga_avg / tpwga_avg if tpwga_avg > 0 else np.inf
        
        print(f"  TPWGA(co) avg time: {tpwga_avg:.3f}s (per obj: {tpwga_avg/5:.3f}s)")
        print(f"  RWRGA(co) avg time: {rwrga_avg:.3f}s (per obj: {rwrga_avg/5:.3f}s)")
        print(f"  Speedup factor: {speedup:.2f}x {'(TPWGA faster)' if speedup>1 else '(RWRGA faster)'}")
        
        print("\n  Final Error Values (RWRGA/TPWGA ratio):")
        for i in range(5):
            target_name = f'E{i+1}'
            tpwga_errs = [exp.err_tpwga[target_name][e][-1] for e in range(num_exp)]
            rwrga_errs = [exp.err_rwrga[target_name][e][-1] for e in range(num_exp)]
            tpwga_mean = np.mean(tpwga_errs)
            rwrga_mean = np.mean(rwrga_errs)
            ratio = rwrga_mean / tpwga_mean if tpwga_mean > 0 else np.inf
            
            if ratio < 0.8:
                status = "RWRGA better"
            elif ratio > 1.2:
                status = "TPWGA better"
            else:
                status = "Similar"
                
            print(f"    {target_name}: {ratio:.3f} ({status})")
    
    print("\n=== Generated Files ===")
    print("  Results: ./results/algorithm_comparison.csv, ./results/rwrga_per_target_times.csv")
    print("  Plots: ./images/algorithm_comparison_5targets.pdf")  
    
    print("\n=== Usage Examples ===")
    print("  python main.py -n 20 -i 200    # 20 experiments, 200 iterations")
    print("  python main.py --fast          # Quick test mode")
    print("  python main.py --skip-comparison  # Skip plots")
    
    print("\n=== Experiments Completed Successfully ===")
