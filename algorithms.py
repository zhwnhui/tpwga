import numpy as np
from scipy.optimize import fminbound, minimize
from tqdm import tqdm
np.seterr(all='ignore')

def TPWGA_co(E_list, dE_list, D, max_sparsity, t_m=0.9, max_iterations=None):
    N = len(E_list)
    num_d, dim = D.shape
    max_iter = max_iterations if max_iterations is not None else max_sparsity * 2

    x = np.zeros((N, dim), dtype=np.float64)
    err_list = [[E_list[i](x[i])] for i in range(N)]
    ind = []
    ind_set = set()
    spr = 0

    def lambda_fun(lam, x_curr, phi_scaled):
        if lam < 0:
            return np.inf
        x_tmp = x_curr + lam * phi_scaled
        return sum(E_list[i](x_tmp[i]) for i in range(N))

    def s_fun(s, x_hat):
        x_tmp = s * x_hat
        return sum(E_list[i](x_tmp[i]) for i in range(N))

    desc = lambda: f'  TPWGA(co): sparsity={spr:3d}/{max_sparsity}, iter={len(ind):3d}/{max_iter}'
    with tqdm(total=max_iter, ascii=True, desc=desc(), 
              bar_format='{desc} |{bar}|', leave=False) as pbar:
        itr = 0
        while spr < max_sparsity and itr < max_iter:
            grad_inner_prods = np.array([
                np.fromiter((-dE_list[i](x[i], D[j]) for j in range(num_d)), 
                            dtype=np.float64, count=num_d)
                for i in range(N)
            ])

            grad_sq_sum = np.sum(grad_inner_prods ** 2, axis=0)
            sqrt_grad_sq = np.sqrt(grad_sq_sum)
            sup_val = np.max(sqrt_grad_sq) + 1e-12
            threshold = t_m * sup_val

            candidate_idx = np.where(sqrt_grad_sq >= threshold)[0]
            phi_m_idx = candidate_idx[0] if len(candidate_idx) > 0 else np.argmax(sqrt_grad_sq)
            phi_m = D[phi_m_idx]

            gip_m = grad_inner_prods[:, phi_m_idx]
            beta_denom = np.sqrt(np.sum(gip_m ** 2)) + 1e-12
            beta_m = gip_m / beta_denom
            phi_scaled = np.outer(beta_m, phi_m)

            lambda_m = fminbound(
                lambda_fun, 0, 10,
                args=(x, phi_scaled),
                xtol=1e-4,
                maxfun=20
            )
            if abs(lambda_m) < 1e-8:
                ind.append(phi_m_idx)
                for i in range(N):
                    err_list[i].append(E_list[i](x[i]))
                itr += 1
                pbar.update()
                continue
            x_hat = x + lambda_m * phi_scaled

            s_m = fminbound(
                s_fun, -5, 5,
                args=(x_hat,),
                xtol=1e-4,
                maxfun=20
            )

            x = s_m * x_hat

            if phi_m_idx not in ind_set:
                spr += 1
                ind_set.add(phi_m_idx)
            ind.append(phi_m_idx)
            
            for i in range(N):
                err_list[i].append(E_list[i](x[i]))

            itr += 1
            pbar.update()

    return err_list, ind









def RWRGA_co(E, dE, D, max_sparsity, t_m=0.9, max_iterations=None):
    x = np.zeros(D.shape[-1])
    err = [E(x)]
    ind = []
    spr = 0
    
    max_iter = max_iterations if max_iterations is not None else max_sparsity * 2
    
    def dE_wrapper(x_val, phi_or_D):
        if len(phi_or_D.shape) == 1:
            return -dE(x_val, phi_or_D)
        else:
            return np.array([-dE(x_val, phi) for phi in phi_or_D])
    
    desc = lambda: f'  RWRGA(co): sparsity={spr:3d}/{max_sparsity}, value={err[-1]:.2e}'
    with tqdm(total=max_iter, ascii=True, desc=desc(), 
              bar_format='{desc} |{bar}|', leave=False) as pbar:
        itr = 0
        while (spr < max_sparsity) and (itr < max_iter):
            grad_inner_products = dE_wrapper(x, D)
            sup_val = np.max(grad_inner_products) + 1e-10
            
            selected_idx = next((idx for idx, val in enumerate(grad_inner_products) 
                                if val >= t_m * sup_val), None)
            if selected_idx is None:
                selected_idx = np.argmax(grad_inner_products)
            phi_m = D[selected_idx]
            
            def lambda_obj(lam):
                return E(x + lam * phi_m) if lam >= 0 else np.inf
            
            lambda_res = minimize(lambda_obj, x0=0.0, bounds=[(0, None)], tol=1e-6)
            lambda_m = lambda_res.x[0]
            
            x_hat_m = x + lambda_m * phi_m
            
            def mu_obj(mu):
                return E(mu * x_hat_m)
            
            mu_res = minimize(mu_obj, x0=1.0, tol=1e-6)
            mu_m = mu_res.x[0]
            
            x = mu_m * x_hat_m
            
            if selected_idx not in ind:
                spr += 1
            ind.append(selected_idx)
            err.append(E(x))
            
            itr += 1
            pbar.update()
    
    return err, ind
