import os
import time
import warnings
import numpy as np
import pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize

warnings.filterwarnings('ignore', category=UserWarning, message='Tight layout not applied')

from algorithms import TPWGA_co, RWRGA_co

np.set_printoptions(precision=4)
sns.set_theme(style='darkgrid', palette='Set2', font='monospace', font_scale=1.2)

class SetupExperiments:
    def __init__(self, seed=0):
        self.seed = seed
        os.makedirs('./images/', exist_ok=True)
        os.makedirs('./results/', exist_ok=True)
        self.max_sparsity = 120
        self._init_fixed_targets()
        self._define_function_forms()
        self.function_names = []

    def _init_fixed_targets(self):
        self.fixed_coefficients = [
            np.array([-0.3503, 7.4337, 3.1915, 3.2564, 5.1087, -6.1464, 0.1871, -4.3729, 6.0989, -4.7985]) / np.sqrt(68.0745),
            np.array([-8.8490, 5.9533, -3.8216, 6.0842, 6.2776, -1.1615, -3.0417, 0.6057, 3.2787, -7.5800]) / np.sqrt(84.8494),
            np.array([4.0988, -7.3654, -0.5787, -2.4201, 3.4830, 7.0936, 5.7626, 8.9905, -0.7825, -7.6696]) / np.sqrt(91.3015),
            np.array([4.3269, -9.0398, -4.2818, 7.3232, 6.5825, -8.8706, 4.6076, -8.4932, 1.3888, -7.9668]) / np.sqrt(93.5379),
            np.array([-8.0936, 2.3424, -5.3692, -7.5148, 1.0766, 7.2496, -6.6960, 4.2379, -2.2778, -4.2073]) / np.sqrt(82.3101)
        ]
        
        self.fixed_indices = [
            list(range(0, 10)), list(range(10, 20)),
            list(range(20, 30)), list(range(30, 40)),
            list(range(40, 50))
        ]
        
        print("=== Fixed Target Functions Initialization ===")
        for i, coef in enumerate(self.fixed_coefficients):
            print(f"f^{i+1}: norm = {np.linalg.norm(coef):.6f} (target: 1.0)")

    def _define_function_forms(self):
        self.function_definitions = [
            {
                'name': 'E1: Log-Cosh Loss',
                'E_func': lambda x, f: np.sum(np.log(np.cosh(x - f))),
                'dE_func': lambda x, y, f: np.sum(np.tanh(x - f) * y)
            },
            {
                'name': 'E2: Quadratic Loss',
                'E_func': lambda x, f: 0.5 * np.dot(x - f, x - f),
                'dE_func': lambda x, y, f: np.dot(x - f, y)
            },
            {
                'name': 'E3: Huber Loss',
                'E_func': lambda x, f: np.sum(np.where(np.abs(x - f) <= 0.5, 
                                                       0.5 * (x - f)**2 / 0.5, 
                                                       np.abs(x - f) - 0.5/2)),
                'dE_func': lambda x, y, f: np.sum(np.where(np.abs(x - f) <= 0.5, 
                                                           (x - f)/0.5 * y, 
                                                           np.sign(x - f) * y))
            },
            {
                'name': 'E4: Gaussian-weighted Quadratic',
                'E_func': lambda x, f: np.sum((x - f)**2 * np.exp(-0.3 * (x - f)**2)),
                'dE_func': lambda x, y, f: np.sum(2 * (x - f) * np.exp(-0.3 * (x - f)**2) * 
                                                  (1 - 0.3 * (x - f)**2) * y)
            },
            {
                'name': 'E5: Log-Quadratic Loss',
                'E_func': lambda x, f: np.sum(np.log(1 + 0.5 * (x - f)**2)),
                'dE_func': lambda x, y, f: np.sum(((x - f)/(1 + 0.5 * (x - f)**2 + 1e-10)) * y)
            }
        ]

    def generate_dictionary(self, dim=100, num_d=300, type_d='gauss'):
        np.random.seed(self.seed)
        if type_d == 'gauss':
            self.D = np.random.randn(num_d, dim)
        else:
            self.D = 2 * np.random.rand(num_d, dim) - 1
        
        self.D /= np.linalg.norm(self.D, 2, axis=1, keepdims=True)
        if self.D.shape[0] < 50:
            raise ValueError("Dictionary must have at least 50 basis vectors")

    def generate_function(self):
        self.generate_dictionary()
        self.original_targets = []
        
        for i in range(5):
            coef = self.fixed_coefficients[i]
            indices = self.fixed_indices[i]
            f_target = np.sum([coef[j] * self.D[idx] for j, idx in enumerate(indices)], axis=0)
            f_target += 0.01 * np.random.randn(len(f_target))
            self.original_targets.append(f_target)
        
        self.targets = []
        self.targets.extend(self.original_targets[:5])
        
        self.E_list = []
        self.dE_list = []
        self.function_names = []
        
        # 只生成前5个函数（E1-E5）
        for i in range(5):
            func_def = self.function_definitions[i]
            f_target = self.targets[i]
            
            E_raw = lambda x, f=f_target: func_def['E_func'](x, f)
            dE_raw = lambda x, y, f=f_target: func_def['dE_func'](x, y, f)
            
            self._normalize_and_store(E_raw, dE_raw, f_target, i)
            self.function_names.append(func_def['name'])
        
        print(f"\nGenerated {len(self.E_list)} functions:")
        for i, name in enumerate(self.function_names):
            print(f"  {name}")

    def _normalize_and_store(self, E_raw, dE_raw, f_target, idx):
        x_zero = np.zeros_like(f_target)
        x_target = f_target.copy()
        
        E_at_zero = E_raw(x_zero)
        E_at_target = E_raw(x_target)
        
        try:
            res = minimize(E_raw, f_target, method='BFGS', tol=1e-6)
            if res.success:
                x_target = res.x
                E_at_target = res.fun
        except:
            pass
        
        diff = max(abs(E_at_zero - E_at_target), 1e-10)
        scale = 1.0 / diff
        
        E_norm = lambda x: (E_raw(x) - E_at_target) * scale
        dE_norm = lambda x, y: dE_raw(x, y) * scale
        
        self.E_list.append(E_norm)
        self.dE_list.append(dE_norm)

    def get_target_function(self, target_idx):
        if target_idx < 0 or target_idx >= len(self.E_list):
            raise ValueError(f"Invalid target index: {target_idx}")
        return lambda x: self.E_list[target_idx](x), lambda x, y: self.dE_list[target_idx](x, y)

    def run(self, num_exp, max_iterations=200):
        self.num_exp = num_exp
        self.max_iterations = max_iterations
        
        num_targets = 5  
        self.err_tpwga = {f'E{i+1}': {} for i in range(num_targets)}
        self.err_rwrga = {f'E{i+1}': {} for i in range(num_targets)}
        self.tpwga_times = []
        self.rwrga_times = []
        self.rwrga_per_target_times = {f'E{i+1}': [] for i in range(num_targets)}
        
        np.random.seed(self.seed)
        test_seeds = np.random.randint(1e9, size=num_exp)
        
        print(f"\n=== Running {num_exp} Comparison Experiments ===")
        print(f"Iterations per run: {max_iterations}, Max sparsity: {self.max_sparsity}")
        
        if not hasattr(self, 'function_names') or len(self.function_names) == 0:
            print("Warning: function_names not initialized, using default names")
            self.function_names = [f'E{i+1}' for i in range(num_targets)]
        
        print(f"Testing {num_targets} functions:")
        for i, name in enumerate(self.function_names):
            print(f"  {name}")
        
        for t in range(num_exp):
            np.random.seed(test_seeds[t])
            self.generate_function()
            
            tpwga_start = time.perf_counter()
            err_list_tpwga, _ = TPWGA_co(self.E_list, self.dE_list, self.D, 
                                       self.max_sparsity, t_m=0.9, max_iterations=max_iterations)
            tpwga_time = time.perf_counter() - tpwga_start
            self.tpwga_times.append(tpwga_time)
            
            rwrga_start = time.perf_counter()
            err_list_rwrga = []
            for i in range(num_targets):
                E_single, dE_single = self.get_target_function(i)
                rwrga_iter_start = time.perf_counter()
                err_rwrga, _ = RWRGA_co(E_single, dE_single, self.D, 
                                      self.max_sparsity, t_m=0.9, max_iterations=max_iterations)
                self.rwrga_per_target_times[f'E{i+1}'].append(time.perf_counter() - rwrga_iter_start)
                err_list_rwrga.append(err_rwrga)
            rwrga_time = time.perf_counter() - rwrga_start
            self.rwrga_times.append(rwrga_time)
            
            for i in range(num_targets):
                self.err_tpwga[f'E{i+1}'][t] = err_list_tpwga[i]
                self.err_rwrga[f'E{i+1}'][t] = err_list_rwrga[i]
            
            print(f"Experiment {t+1}/{num_exp}: TPWGA={tpwga_time:.4f}s, RWRGA={rwrga_time:.4f}s")
        
        self._save_comparison_results()
        print("\n=== Experiments Completed ===")

    def _save_comparison_results(self):
        data = []
        for exp_idx in range(self.num_exp):
            row = {
                'Experiment': exp_idx + 1,
                'TPWGA_Time': self.tpwga_times[exp_idx],
                'RWRGA_Total_Time': self.rwrga_times[exp_idx],
                'Speedup_Factor': self.rwrga_times[exp_idx] / self.tpwga_times[exp_idx],
                'TPWGA_per_Objective': self.tpwga_times[exp_idx] / 5,  
                'RWRGA_per_Objective': self.rwrga_times[exp_idx] / 5  
            }
            
            for i in range(5): 
                target_name = f'E{i+1}'
                tpwga_final = self.err_tpwga[target_name][exp_idx][-1]
                rwrga_final = self.err_rwrga[target_name][exp_idx][-1]
                row[f'{target_name}_TPWGA'] = tpwga_final
                row[f'{target_name}_RWRGA'] = rwrga_final
                row[f'{target_name}_Error_Ratio'] = rwrga_final / tpwga_final if tpwga_final > 0 else np.inf
            
            data.append(row)
        
        pd.DataFrame(data).to_csv('./results/algorithm_comparison.csv', index=False)
        
        target_time_data = []
        for exp_idx in range(self.num_exp):
            for i in range(5):  # 只保存5个目标的时间
                target_name = f'E{i+1}'
                if i < len(self.function_names):
                    func_name = self.function_names[i]
                else:
                    func_name = f'E{i+1}'
                    
                target_time_data.append({
                    'Experiment': exp_idx + 1,
                    'Target': target_name,
                    'Function_Name': func_name,
                    'RWRGA_Time': self.rwrga_per_target_times[target_name][exp_idx]
                })
        pd.DataFrame(target_time_data).to_csv('./results/rwrga_per_target_times.csv', index=False)
        print("✓ Results saved to ./results/")

    def plot_algorithm_comparison(self):
        if not hasattr(self, 'function_names') or len(self.function_names) == 0:
            self.function_names = [f'E{i+1}' for i in range(5)]  
            
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))  
        axes = axes.ravel()
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        exp_idx = 0
        
        for i in range(5):  # 5
            ax = axes[i]
            target_name = f'E{i+1}'
            if i < len(self.function_names):
                display_name = self.function_names[i].split(':')[0] if ':' in self.function_names[i] else self.function_names[i]
            else:
                display_name = target_name
            
            if target_name not in self.err_tpwga or exp_idx not in self.err_tpwga[target_name]:
                ax.text(0.5, 0.5, f"No data for {target_name}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{target_name}: {display_name}', fontsize=10, fontweight='bold')
                continue
                
            err_tpwga = np.array(self.err_tpwga[target_name][exp_idx])
            err_rwrga = np.array(self.err_rwrga[target_name][exp_idx])
            
            err_tpwga = np.maximum(err_tpwga, 1e-10)
            err_rwrga = np.maximum(err_rwrga, 1e-10)
            
            iterations_tpwga = np.arange(len(err_tpwga))
            iterations_rwrga = np.arange(len(err_rwrga))
           
            ax.plot(iterations_tpwga, err_tpwga, 
                   linewidth=2, label='TPWGA(co)', color=colors[i], linestyle='-')
          
            ax.plot(iterations_rwrga, err_rwrga,
                   linewidth=2, label='RWRGA(co)', color=colors[i], linestyle='--')
            
            ax.scatter(len(err_tpwga)-1, err_tpwga[-1], s=80, color=colors[i],
                      marker='o', edgecolors='black', linewidth=1.5, zorder=5)
            
            ax.scatter(len(err_rwrga)-1, err_rwrga[-1], s=80, color=colors[i],
                      marker='s', edgecolors='black', linewidth=1.5, zorder=5)
            
            y_pos_tp = err_tpwga[-1] * 1.5 if err_tpwga[-1] > 1e-3 else err_tpwga[-1] * 0.5
            y_pos_rw = err_rwrga[-1] * 1.5 if err_rwrga[-1] > 1e-3 else err_rwrga[-1] * 0.5
            
            ax.text(len(err_tpwga)-1, y_pos_tp,
                   f'{err_tpwga[-1]:.2e}', fontsize=8, ha='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
           
            ax.text(len(err_rwrga)-1, y_pos_rw,
                   f'{err_rwrga[-1]:.2e}', fontsize=8, ha='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            ax.set_yscale('log')
            ax.set_xlabel('Iterations', fontsize=10)
            ax.set_ylabel('Function Value', fontsize=10)
            
            ax.set_title(f'{target_name}', fontsize=10)
            
            ax.legend(loc='lower left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
        
        # 6
        ax_time = axes[5]
        
        if len(self.tpwga_times) == 0 or len(self.rwrga_times) == 0:
            ax_time.text(0.5, 0.5, "No timing data available", 
                        ha='center', va='center', transform=ax_time.transAxes)
            ax_time.set_title('Runtime Comparison', fontsize=12, fontweight='bold')
        else:
            tpwga_mean_time = np.mean(self.tpwga_times)
            tpwga_std_time = np.std(self.tpwga_times)
            rwrga_mean_time = np.mean(self.rwrga_times)
            rwrga_std_time = np.std(self.rwrga_times)
            
            algorithms = ['TPWGA(co)', 
                         '5-fold RWRGA(co)']  
            times = [tpwga_mean_time, rwrga_mean_time]
            stds = [tpwga_std_time, rwrga_std_time]
            
            bars = ax_time.bar(algorithms, times, yerr=stds, 
                               color=['red', 'blue'], 
                               alpha=0.8, width=0.6, capsize=10, edgecolor='black')
            
            for bar, time_val, std_val in zip(bars, times, stds):
                ax_time.text(bar.get_x() + bar.get_width()/2., time_val + std_val + 0.05,
                       f'{time_val:.3f}s\n±{std_val:.3f}s', ha='center', va='bottom', 
                       fontsize=10, fontweight='bold')
            
            ax_time.set_ylabel('Average Runtime (seconds)', fontsize=11)
            ax_time.set_title('Runtime Comparison', fontsize=12)
            ax_time.grid(True, alpha=0.3, axis='y')
            ax_time.set_ylim(0, max(times)*1.3)
            ax_time.tick_params(axis='both', which='major', labelsize=9)
        
        
        for i in range(6, len(axes)):
            axes[i].set_visible(False)
        
        plt.subplots_adjust(
            left=0.06,
            right=0.98,
            bottom=0.05,
            top=0.95,
            wspace=0.25,
            hspace=0.4
        )
        
        plt.suptitle('', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig('./images/algorithm_comparison_5targets.pdf', format='pdf', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Plot saved to ./images/algorithm_comparison_5targets.pdf")
