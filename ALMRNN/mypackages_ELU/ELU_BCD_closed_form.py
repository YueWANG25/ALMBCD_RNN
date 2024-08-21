#mypackages/BCD_closed form
import numpy as np
from functools import partial
from scipy.optimize import minimize
import mypackages.functions_calculate_ALfunc as ALfunc
import mypackages.tool_functions as toolfunc
import mypackages_ELU.ELU_grad_AL as ELU_gd


#%% Functions to calculate closed-form solutions of BCD
def updateZin_closed_blk(u, ht, x_dataset, xi1, gamma1, lambda2, lambda3,
                         lambda4, Nx, Nh, T, row_i):
    psi = ALfunc.Psi_blk(ht, x_dataset, Nh, T)
    psi_T = psi.T
    index = np.arange(row_i, T * Nh + row_i, Nh)
    P = 2 * np.diag(
        np.concatenate((lambda2 * np.ones(Nh), lambda3 * np.ones(Nx),
                        lambda4 * np.ones(1)))) + gamma1 * np.dot(psi_T, psi)
    g = np.dot(psi_T, xi1[index]) + gamma1 * np.dot(psi_T, u[index])
    try:
        L = np.linalg.cholesky(P)
        y = np.linalg.solve(L, g)
        zin_kj_blk = np.linalg.solve(L.T, y)
        W_kj_blk = zin_kj_blk[:Nh, :].T
        V_kj_blk = zin_kj_blk[Nh:Nh + Nx, :].T
        b_kj_blk = zin_kj_blk[Nh + Nx:2 * Nh + Nx, :].T
        return W_kj_blk, V_kj_blk, b_kj_blk
    except:
        print(
            'Warning: P is not symmetric positive definite. zin cannot be solved by Cholesky'
        )


def updateZin_closed(W, V, b, u, ht, x_dataset, xi1, gamma1, lambda2, lambda3,
                     lambda4, Nx, Nh, T):
    W = np.zeros([Nh, Nh])
    V = np.zeros([Nh, Nx])
    b = np.zeros([Nh, 1])
    for row_i in range(Nh):
        W[row_i, :], V[row_i, :], b[row_i, :] = updateZin_closed_blk(
            u, ht, x_dataset, xi1, gamma1, lambda2, lambda3, lambda4, Nx, Nh,
            T, row_i)
    zin = toolfunc.vectorwise([W, V, b])
    return zin, W, V, b


def updateZout_closed(ht, y_dataset, lambda1, lambda5, Nh, Ny, T):
    Phisum = []
    Phiysum = []
    phi_blk = np.concatenate((ht, np.ones([1, T])))
    Phisum = np.kron(np.dot(phi_blk, phi_blk.T), np.identity(Ny))
    Phiysum = ALfunc.PhiTy(ht, y_dataset, Nh, Ny, T)
    Q = (2 / T) * Phisum + 2 * np.diag(
        np.hstack((lambda1 * np.ones(Nh * Ny), lambda5 * np.ones(Ny))))
    f = (2 / T) * Phiysum
    # Method1.1: If Q is not positive definite
    # %   zout_kj=pinv(Q)*f;
    # %   zout_k=zout_kj;
    # Method1.2: If Q is positive definite
    try:
        L = np.linalg.cholesky(Q)
        y = np.linalg.solve(L, f)
        zout_kj = np.linalg.solve(L.T, y)
        return zout_kj
    except:
        # dbstop
        return 'Warning: Q is not symmetric positive definite. zout cannot be solved by Cholesky'


def updateH_closed(A_kj, W_kj, V_kj, b_kj, c_kj, ut_kj, x_trainset, y_trainset,
                   gamma1, gamma2, xi1, xi2, Nx, Nh, Ny, T):
    D1 = gamma1 * W_kj.T @ W_kj + (
        2 / T) * A_kj.T @ A_kj + gamma2 * np.identity(Nh)
    D2 = (2 / T) * A_kj.T @ A_kj + gamma2 * np.identity(Nh)
    D1_inv = np.linalg.inv(D1)
    D2_inv = np.linalg.inv(D2)
    xit1 = xi1.reshape([Nh, T], order="F")
    xit2 = xi2.reshape([Nh, T], order="F")
    ht_ing = np.zeros((Nh, T))  # to record the calculation results
    for t in range(0, T - 1):
        b1 = gamma2 * toolfunc.ELU(
            ut_kj[:, [t]]
        ) - xit2[:, [t]] + W_kj.T @ xit1[:, [t + 1]] + gamma1 * W_kj.T @ (
            ut_kj[:, [t + 1]] - V_kj @ x_trainset[t + 1].reshape(
                [Nx, 1], order="F") - b_kj) + (2 / T) * A_kj.T @ (
                    y_trainset[t].reshape([Ny, 1], order="F") - c_kj)
        ht_ing[:, [t]] = D1_inv @ b1
    b2 = gamma2 * toolfunc.ELU(
        ut_kj[:, [T - 1]]) - xit2[:, [T - 1]] + (2 / T) * A_kj.T @ (
            y_trainset[T - 1].reshape([Ny, 1], order="F") - c_kj)
    ht_ing[:, [T - 1]] = D2_inv @ b2
    ht_kj = ht_ing
    return ht_kj


def ValueLi_u_vec(u, u_last, h, cons_a, gamma1, gamma2, beta1, beta2, sigma,
                  lambda6):
    L = 0.5 * gamma1 * (u - cons_a + beta1)**2 + 0.5 * gamma2 * (
        h + beta2 -
        toolfunc.ELU(u))**2 + 0.5 * sigma * (u - u_last)**2 + lambda6 * u**2
    return L


def projected_gradient_bb(z_in_ini, W_ini, V_ini, b_ini, h_ini, ht_ini, u_ini,
                          ut_ini, u_last, x_dataset, xi1, xi2, gamma1, gamma2,
                          lambda6, sigma, Nx, Nh, T, max_iter, tol,
                          upper_bound, lower_bound, index):

    u = u_ini[index]
    grad_u = ELU_gd.GradALM_u_block(W_ini, V_ini, b_ini, u_ini, ut_ini, u_last,
                                    h_ini, ht_ini, z_in_ini, x_dataset, xi1,
                                    xi2, gamma1, gamma2, T, Nx, Nh, lambda6,
                                    sigma).ravel()
    grad_u = grad_u[index]
    alpha = 1.0

    for kk in range(max_iter):
        # Update and calculate gradient
        u_new = np.minimum(
            np.maximum(u - alpha * grad_u.ravel(), lower_bound[index]),
            upper_bound[index])
        u_new_full = np.zeros(Nh * T)
        u_new_full[index] = u_new
        ut_new_full = u_new_full.reshape([Nh, T], order="F")

        grad_u_new = ELU_gd.GradALM_u_block(W_ini, V_ini, b_ini, u_new_full,
                                            ut_new_full, u_last, h_ini, ht_ini,
                                            z_in_ini, x_dataset, xi1, xi2,
                                            gamma1, gamma2, T, Nx, Nh, lambda6,
                                            sigma).ravel()
        grad_u_new = grad_u_new[index]

        # Calculate diff
        diff_u = u_new - u
        diff_grad_u = grad_u_new - grad_u

        # Stop criterion
        if np.linalg.norm(grad_u_new) < tol:
            break

        # Calculate BB steps
        if (np.dot(diff_u, diff_grad_u)) > 0:
            alpha = np.dot(diff_u, diff_u) / np.dot(diff_u, diff_grad_u)
        # Update variables and gradient

        u = u_new
        grad_u = grad_u_new
    return u


def U_minus_new(z_in_ini, W_ini, V_ini, b_ini, u_last, h_ini, cons_a, gamma1,
                gamma2, beta1, beta2, theta1, theta2, sigma, lambda6, epo_k,
                x_dataset, xi1, xi2, Nx, Nh, T):
    judge_cons = gamma1**2 * (theta2 + 1)**2 - 8 * gamma1 * (sigma + gamma1 +
                                                             2 * lambda6)
    point1 = gamma1 * (theta2 + 1) - (gamma1**2 *
                                      (theta2 + 1)**2 - 8 * gamma1 *
                                      (sigma + gamma1 + 2 * lambda6)**0.5)
    point2 = gamma1 * (theta2 + 1) + (gamma1**2 *
                                      (theta2 + 1)**2 - 8 * gamma1 *
                                      (sigma + gamma1 + 2 * lambda6)**0.5)
    final_result = np.zeros(Nh * T)

    neg_indices = np.where((judge_cons <= 0)
                           | ((judge_cons > 0) & (point1 > 1))
                           | ((judge_cons > 0) & (point2 < 0)))[0]
    posi_indices_1 = np.where((judge_cons > 0) & (point2 > 1) & (point1 <= 1)
                              & (point1 > 0))[0]
    posi_indices_2 = np.where((judge_cons > 0) & (point2 <= 1)
                              & (point1 < point2) & (point1 > 0))[0]
    posi_indices_3 = np.where((judge_cons > 0) & (point2 > 1)
                              & (point1 < 0))[0]
    posi_indices_4 = np.where((judge_cons > 0) & (point2 <= 1) & (point2 > 0)
                              & (point1 < 0))[0]

    ht_ini = h_ini.reshape([Nh, T], order="F")

    max_iter = 50
    # for no root and one root cases
    if neg_indices.size > 0:
        lower = np.full(Nh * T, -np.inf)
        upper = np.zeros(Nh * T)
        u_ini = np.zeros(Nh * T)
        ut_ini = u_ini.reshape([Nh, T], order='F')

        neg_result = projected_gradient_bb(z_in_ini, W_ini, V_ini, b_ini,
                                           h_ini, ht_ini, u_ini, ut_ini,
                                           u_last, x_dataset, xi1, xi2, gamma1,
                                           gamma2, lambda6, sigma, Nx, Nh, T,
                                           max_iter, epo_k, upper, lower,
                                           neg_indices)
        final_result[neg_indices] = neg_result

    # for convex and concave
    if posi_indices_1.size > 0:
        upper = np.log(point1).ravel()
        lower = np.full(Nh * T, -np.inf)
        u_ini = np.log(point1).ravel()
        ut_ini = u_ini.reshape([Nh, T], order='F')

        posi_result_1 = projected_gradient_bb(z_in_ini, W_ini, V_ini, b_ini,
                                              h_ini, ht_ini, u_ini, ut_ini,
                                              u_last, x_dataset, xi1, xi2,
                                              gamma1, gamma2, lambda6, sigma,
                                              Nx, Nh, T, max_iter, epo_k,
                                              upper, lower, posi_indices_1)
        F_posi_result_1 = ValueLi_u_vec(posi_result_1, u_last[posi_indices_1],
                                        h_ini[posi_indices_1],
                                        cons_a[posi_indices_1], gamma1, gamma2,
                                        beta1[posi_indices_1],
                                        beta2[posi_indices_1], sigma, lambda6)
        F_zero = ValueLi_u_vec(np.zeros(len(posi_indices_1)),
                               u_last[posi_indices_1], h_ini[posi_indices_1],
                               cons_a[posi_indices_1], gamma1, gamma2,
                               beta1[posi_indices_1], beta2[posi_indices_1],
                               sigma, lambda6)
        index_1 = np.where(F_posi_result_1 > F_zero)[0]
        posi_result_1[index_1] = np.zeros(len(index_1))
        final_result[posi_indices_1] = posi_result_1

    # for convex concave convex
    if posi_indices_2.size > 0:
        upper = np.log(point1).ravel()
        lower = np.full(Nh * T, -np.inf)
        u_ini = np.log(point1).ravel()
        ut_ini = u_ini.reshape([Nh, T], order='F')

        posi_result_21 = projected_gradient_bb(z_in_ini, W_ini, V_ini, b_ini,
                                               h_ini, ht_ini, u_ini, ut_ini,
                                               u_last, x_dataset, xi1, xi2,
                                               gamma1, gamma2, lambda6, sigma,
                                               Nx, Nh, T, max_iter, epo_k,
                                               upper, lower, posi_indices_2)
        F_posi_result_21 = ValueLi_u_vec(posi_result_21,
                                         u_last[posi_indices_2],
                                         h_ini[posi_indices_2],
                                         cons_a[posi_indices_2], gamma1,
                                         gamma2, beta1[posi_indices_2],
                                         beta2[posi_indices_2], sigma, lambda6)

        upper = np.zeros(Nh * T)
        lower = np.log(point2).ravel()
        u_ini = np.log(point2).ravel()
        ut_ini = u_ini.reshape([Nh, T], order='F')

        posi_result_22 = projected_gradient_bb(z_in_ini, W_ini, V_ini, b_ini,
                                               h_ini, ht_ini, u_ini, ut_ini,
                                               u_last, x_dataset, xi1, xi2,
                                               gamma1, gamma2, lambda6, sigma,
                                               Nx, Nh, T, max_iter, epo_k,
                                               upper, lower, posi_indices_2)
        F_posi_result_22 = ValueLi_u_vec(posi_result_22,
                                         u_last[posi_indices_2],
                                         h_ini[posi_indices_2],
                                         cons_a[posi_indices_2], gamma1,
                                         gamma2, beta1[posi_indices_2],
                                         beta2[posi_indices_2], sigma, lambda6)
        posi_result_21[F_posi_result_22 < F_posi_result_21] = posi_result_22
        final_result[posi_indices_2] = posi_result_21

    if posi_indices_3.size > 0:
        posi_result_3 = np.zeros(len(posi_indices_3))
        final_result[posi_indices_3] = posi_result_3

    if posi_indices_4.size > 0:
        upper = np.zeros(Nh * T)
        lower = np.log(point2).ravel()
        u_ini = np.zeros(Nh * T)
        ut_ini = u_ini.reshape([Nh, T], order='F')

        posi_result_4 = projected_gradient_bb(z_in_ini, W_ini, V_ini, b_ini,
                                              h_ini, ht_ini, u_ini, ut_ini,
                                              u_last, x_dataset, xi1, xi2,
                                              gamma1, gamma2, lambda6, sigma,
                                              Nx, Nh, T, max_iter, epo_k,
                                              upper, lower, posi_indices_4)
        final_result[posi_indices_4] = posi_result_4

    return final_result.reshape([Nh * T, 1])


def updateU_closed_new(z_in, z_out, A, W, V, b, h, ht, u_last, xi1, xi2,
                       gamma1, gamma2, lambda1, lambda2, lambda3, lambda4,
                       lambda6, sigma, epo_k, x_dataset, y_dataset, Nh, Nx, Ny,
                       T):
    # compute constant k
    beta1 = xi1 / gamma1
    beta2 = xi2 / gamma2
    cons_a = ALfunc.PsiX(ht, x_dataset, z_in, Nx, Nh, T)
    theta1 = cons_a - beta1
    theta2 = h + beta2
    theta3 = u_last
    # case1: u > 0
    u_posi = (gamma1 * theta1 + gamma1 * theta2 +
              sigma * theta3) * (1 / (2 * gamma1 + 2 * lambda6 + sigma))
    index_posi = u_posi <= 0
    u_posi[index_posi] = 0
    # case2: u <= 0
    u_neg = U_minus_new(z_in, W, V, b, u_last, h, cons_a, gamma1, gamma2,
                        beta1, beta2, theta1, theta2, sigma, lambda6, epo_k,
                        x_dataset, xi1, xi2, Nx, Nh, T)
    # compare function value of case1 and case2
    L_posi = ValueLi_u_vec(u_posi, u_last, h, cons_a, gamma1, gamma2, beta1,
                           beta2, sigma, lambda6)
    L_neg = ValueLi_u_vec(u_neg, u_last, h, cons_a, gamma1, gamma2, beta1,
                          beta2, sigma, lambda6)
    index_L = L_posi > L_neg
    u_posi[index_L] = u_neg[index_L]
    u_star = u_posi
    return u_star
