# Import and install dependency packages
from dependency_packages import np, pd, scipy, copy, time, random, math, gc

# Import mypackages
import mypackages.tool_functions as toolfunc
import mypackages.initial_functions as inifunc
import mypackages.functions_calculate_ALfunc as ALfunc
import mypackages_ELU.ELU_BCD_closed_form as ELU_BCD_closed
from gen_synthetic_dataset import generate_synthetic_dataset


def load_dataset(dataset_name):
    dataset_ori = pd.read_csv(dataset_name + '.csv', header=0, index_col=0)
    return dataset_ori


def split_dataset(dataset_ori, Ny, sizerate_training=0.9):
    dataset_length, data_dim = dataset_ori.shape
    Nx = data_dim - Ny
    train_length = int(sizerate_training * dataset_length)
    T = train_length
    T_test = dataset_length - T

    x_trainset = dataset_ori.iloc[:train_length, :Nx].to_numpy()
    y_trainset = dataset_ori.iloc[:train_length, :Ny].to_numpy()

    x_testset = dataset_ori.iloc[train_length:, :Nx].to_numpy()
    y_testset = dataset_ori.iloc[train_length:, :Ny].to_numpy()

    return x_trainset, y_trainset, x_testset, y_testset, T, T_test, Nx


def initialize_ALM_params(Nh,
                          Ny,
                          Nx,
                          T,
                          gamma0=1,
                          lambda1=None,
                          lambda2=None,
                          lambda3=None,
                          lambda4=None,
                          lambda5=None,
                          lambda6=1e-8,
                          eta1=0.99,
                          eta2=5 / 6,
                          eta3=0.01,
                          eta4=5 / 6,
                          epo_0=0.1,
                          epo_star=1e-20,
                          sigma=1e-5,
                          ALM_m=1,
                          submaxiter=100):

    # Calculate default values if not provided
    if lambda1 is None:
        lambda1 = 1 / (Nh * Ny)
    if lambda2 is None:
        lambda2 = 1 / (Nh * Nh)
    if lambda3 is None:
        lambda3 = 1 / (Nx * Nh)
    if lambda4 is None:
        lambda4 = 1 / Nh
    if lambda5 is None:
        lambda5 = 1 / Ny

    xi1 = np.zeros([Nh * T, 1])
    xi2 = np.zeros([Nh * T, 1])
    gamma1 = gamma0
    gamma2 = gamma0

    epo_k = epo_0

    return {
        "xi1": xi1,
        "xi2": xi2,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "lambda1": lambda1,
        "lambda2": lambda2,
        "lambda3": lambda3,
        "lambda4": lambda4,
        "lambda5": lambda5,
        "lambda6": lambda6,
        "eta1": eta1,
        "eta2": eta2,
        "eta3": eta3,
        "eta4": eta4,
        "epo_k": epo_k,
        "epo_star": epo_star,
        "sigma": sigma,
        "ALM_m": ALM_m,
        "submaxiter": submaxiter
    }


def initialize_variables(Ny,
                         Nh,
                         Nx,
                         T,
                         T_test,
                         distribution_type="Gaussian",
                         mean=None,
                         std=None,
                         x_trainset=None,
                         y_trainset=None,
                         x_testset=None,
                         y_testset=None,
                         xi1=None,
                         xi2=None,
                         gamma1=None,
                         gamma2=None,
                         lambda1=None,
                         lambda2=None,
                         lambda3=None,
                         lambda4=None,
                         lambda5=None,
                         lambda6=None):

    # Set defaults based on distribution type
    if distribution_type == "Gaussian":
        if mean is None:
            mean = 0
        if std is None:
            std = 1e-3
    else:
        if mean is None:
            mean = 0
        if std is None:
            std = 20

    random.seed(123456)

    # Initialize variables using the specified distribution
    A0, W0, V0, b0, c0, zin_0, zout_0 = inifunc.iniVariable(
        Ny, Nh, Nx, T, mean, std, distribution_type)
    u0, ut0, h0, ht0, y_hat0 = ALfunc.CalculY(x_trainset, A0, W0, V0, b0, c0,
                                              T, Nh, Nx, Ny)
    y0_auxi = ALfunc.CalculY_auxi(zout_0, ht0, Nh, Ny, T)

    # Ensure all necessary parameters are provided
    if lambda1 is None or lambda2 is None or lambda3 is None or lambda4 is None or lambda5 is None or lambda6 is None:
        raise ValueError("All lambda parameters must be provided.")
    if xi1 is None or xi2 is None:
        raise ValueError("xi1 and xi2 must be provided.")
    if gamma1 is None or gamma2 is None:
        raise ValueError("gamma1 and gamma2 must be provided.")

    regvalue0 = ALfunc.ValueRegula(A0, W0, V0, b0, c0, u0, lambda1, lambda2,
                                   lambda3, lambda4, lambda5, lambda6)
    lossALM0 = ALfunc.ValueLoss(
        y_trainset, y0_auxi, Ny, T) + regvalue0 + ALfunc.ValueALMterm(
            x_trainset, ht0, h0, u0, zin_0, xi1, xi2, gamma1, gamma2, Nx, Nh,
            T)

    TrainErr0 = (1 / T) * np.sum((y_hat0 - y_trainset)**2)
    if np.isnan(TrainErr0) == True:
        TrainErr0 = 1e+30

    _, _, _, _, y_test_hat0 = ALfunc.CalculY(x_testset, A0, W0, V0, b0, c0,
                                             T_test, Nh, Nx, Ny)
    TestErr0 = (1 / T_test) * np.sum((y_test_hat0 - y_testset)**2)

    FeasVi_h0 = (1 / T) * np.sum((h0 - toolfunc.ELU(u0))**2)**(1 / 2)
    FeasVi_u0 = (1 / T) * np.linalg.norm(
        u0 - ALfunc.PsiX(ht0, x_trainset, zin_0, Nx, Nh, T))

    return {
        "A0": A0,
        "W0": W0,
        "V0": V0,
        "b0": b0,
        "c0": c0,
        "zin_0": zin_0,
        "zout_0": zout_0,
        "u0": u0,
        "ut0": ut0,
        "h0": h0,
        "ht0": ht0,
        "y_hat0": y_hat0,
        "y0_auxi": y0_auxi,
        "lossALM0": lossALM0,
        "feasvi_h0": FeasVi_h0,
        "feasvi_u0": FeasVi_u0,
        "TrainErr0": TrainErr0,
        "TestErr0": TestErr0
    }


def alm_bcd_ELU_optimization(dataset_name,
                             Ny=1,
                             Nh=20,
                             maxiter=10,
                             submaxiter=100,
                             distribution_type="Gaussian",
                             mean=None,
                             std=None,
                             gamma0=1,
                             lambda1=None,
                             lambda2=None,
                             lambda3=None,
                             lambda4=None,
                             lambda5=None,
                             lambda6=1e-8,
                             eta1=0.99,
                             eta2=5 / 6,
                             eta3=0.01,
                             eta4=5 / 6,
                             epo_0=0.1,
                             epo_star=1e-20,
                             sigma=1e-5,
                             ALM_m=1,
                             print_option=2):
    """
    Perform ALM_BCD to train RNNs.

    Parameters:
    - dataset_name: Name of the dataset to load
    - Ny: Number of output units (default: 1)
    - Nh: Number of hidden units (default: 20)
    - maxiter: Maximum number of iterations for the outer loop (default: 10)
    - submaxiter: Maximum number of iterations for the inner loop (default: 100)
    - distribution_type: Type of distribution for variable initialization ("Gaussian" "He", "Glorot", "LeCun", default: "Gaussian")
    - mean: Mean for the initialization distribution (default: 0 for Gaussian, 0 for other)
    - std: Standard deviation for the initialization distribution (default: 1e-3 for Gaussian, 20 for other)
    - gamma0: Initial value of penalty parameters (default: 1)
    - lambda1, lambda2, lambda3, lambda4, lambda5, lambda6: Regularization parameters
    - eta1, eta2, eta3, eta4: Parameters for updating the algorithm
    - epo_0: Initial value of \epsilon_0 (default: 0.1)
    - epo_star: Stop criterion for epo_k (default: 1e-20)
    - sigma: Penalty term parameter \mu (default: 1e-5)
    - ALM_m: If ALM_m = 1, the ALM algorithm is monotone; if ALM_m > 1, the ALM become a non-monotone algorithm and may accelerate the algorithm (default: 1)
    - print: Control the verbosity of the output (default: 0)
        - 0: No output
        - 1: Print final TrainErr, TestErr, FeasVi_h, FeasVi_u
        - 2: Print TrainErr, TestErr, FeasVi_h, FeasVi_u, and time for each outer iteration

    Returns:
    - results: A dictionary containing lists for TrainErr, TestErr, time, FeasVi_h, and FeasVi_u
    """

    # Print the optimization problem being solved
    if print_option >= 1:
        print(
            "\nWe use an Augmented Lagrangian Method to solve the following RNN training problem with ELU:"
        )
        print("minimize l(s) + p(s)")
        print("subject to:")
        print("u - Psi(h) = 0")
        print("h - ELU(u) = 0\n")

    # Load and split the dataset
    dataset_ori = load_dataset(dataset_name)
    x_trainset, y_trainset, x_testset, y_testset, T, T_test, Nx = split_dataset(
        dataset_ori, Ny)

    # Initialize ALM parameters
    params = initialize_ALM_params(Nh,
                                   Ny,
                                   Nx,
                                   T,
                                   gamma0=gamma0,
                                   lambda1=lambda1,
                                   lambda2=lambda2,
                                   lambda3=lambda3,
                                   lambda4=lambda4,
                                   lambda5=lambda5,
                                   lambda6=lambda6,
                                   eta1=eta1,
                                   eta2=eta2,
                                   eta3=eta3,
                                   eta4=eta4,
                                   epo_0=epo_0,
                                   epo_star=epo_star,
                                   sigma=sigma,
                                   ALM_m=1,
                                   submaxiter=submaxiter)

    # Initialize variables
    variables = initialize_variables(Ny,
                                     Nh,
                                     Nx,
                                     T,
                                     T_test,
                                     distribution_type=distribution_type,
                                     mean=mean,
                                     std=std,
                                     x_trainset=x_trainset,
                                     y_trainset=y_trainset,
                                     x_testset=x_testset,
                                     y_testset=y_testset,
                                     xi1=params["xi1"],
                                     xi2=params["xi2"],
                                     gamma1=params["gamma1"],
                                     gamma2=params["gamma2"],
                                     lambda1=params["lambda1"],
                                     lambda2=params["lambda2"],
                                     lambda3=params["lambda3"],
                                     lambda4=params["lambda4"],
                                     lambda5=params["lambda5"],
                                     lambda6=params["lambda6"])

    # Initialize lists to store results
    FeasVi_h, FeasVi_u = [variables["feasvi_h0"]], [variables["feasvi_u0"]]
    TrainErr, TestErr = [variables["TrainErr0"]], [variables["TestErr0"]]
    gamma1_k = [params["gamma1"]]
    gamma_max = 1e4
    funcval_k = [variables["lossALM0"].item()]
    epo_k_recording = [params["epo_k"]]
    time_ALMRNN = [0]

    # Begin outer loop: ALM
    for k in range(maxiter):
        start_k = time.perf_counter()
        # print(f'Iteration {k+1}/{maxiter}')
        theta = 1e+30
        # Initialize variables for both the problem and subproblem
        if k == 0:
            A_kj, W_kj, V_kj, b_kj, c_kj = variables["A0"], variables[
                "W0"], variables["V0"], variables["b0"], variables["c0"]
            zin_kj, zout_kj, h_kj, u_kj, ht_kj, ut_kj = variables[
                "zin_0"], variables["zout_0"], variables["h0"], variables[
                    "u0"], variables["ht0"], variables["ut0"]
        else:
            if lossALM <= theta:
                A_kj, W_kj, V_kj, b_kj, c_kj = A_km1, W_km1, V_km1, b_km1, c_km1
                zin_kj, zout_kj, h_kj, u_kj, ht_kj, ut_kj = zin_km1, zout_km1, h_km1, u_km1, ht_km1, ut_km1
            else:
                if lossALM_bar <= theta:
                    A_kj, W_kj, V_kj, b_kj, c_kj = A_km1, W_km1, V_km1, b_km1, c_km1
                    zin_kj, zout_kj, h_kj, u_kj, ht_kj, ut_kj = zin_km1, zout_km1, h_bar, u_bar, ht_bar, ut_bar
                else:
                    A_kj, W_kj, V_kj, b_kj, c_kj = A0, W0, V0, b0, c0
                    zin_kj, zout_kj, h_kj, u_kj, ht_kj, ut_kj = zin_0, zout_0, h0, u0, ht0, ut0

        # Begin inner loop: BCD
        for j in range(params["submaxiter"]):
            A_kjm1, W_kjm1, V_kjm1, b_kjm1, c_kjm1, zin_kjm1, zout_kjm1 = A_kj, W_kj, V_kj, b_kj, c_kj, zin_kj, zout_kj
            h_kjm1, ht_kjm1, u_kjm1, ut_kjm1 = h_kj, ht_kj, u_kj, ut_kj

            # Update \bf{w}
            zin_kj, W_kj, V_kj, b_kj = ELU_BCD_closed.updateZin_closed(
                W_kjm1, V_kjm1, b_kjm1, u_kjm1, ht_kjm1, x_trainset,
                params["xi1"], params["gamma1"], lambda2, lambda3, lambda4, Nx,
                Nh, T)

            # Update \bf{a}
            zout_kj = ELU_BCD_closed.updateZout_closed(ht_kjm1, y_trainset,
                                                       lambda1, lambda5, Nh,
                                                       Ny, T)
            A_kj = zout_kj[:Ny * Nh].reshape((Ny, Nh), order="F")
            c_kj = zout_kj[Ny * Nh:Ny * Nh + Ny].reshape((Ny, 1))

            # Update \bf{h}
            ht_kj = ELU_BCD_closed.updateH_closed(A_kj, W_kj, V_kj, b_kj, c_kj,
                                                  ut_kjm1, x_trainset,
                                                  y_trainset, params["gamma1"],
                                                  params["gamma2"],
                                                  params["xi1"], params["xi2"],
                                                  Nx, Nh, Ny, T)
            h_kj = ht_kj.T.flatten()
            h_kj = h_kj.reshape(T * Nh, 1)

            # Update \bf{u}
            u_kj = ELU_BCD_closed.updateU_closed_new(
                zin_kj, zout_kj, A_kj, W_kj, V_kj, b_kj, h_kj, ht_kj, u_kjm1,
                params["xi1"], params["xi2"], params["gamma1"],
                params["gamma1"], lambda1, lambda2, lambda3, lambda4, lambda6,
                sigma, 1e-8, x_trainset, y_trainset, Nh, Nx, Ny, T)
            ut_kj = u_kj.reshape([Nh, T], order="F")

            # Stop criterion for the subproblem
            diff_vari_norm_kj = toolfunc.variableNorm(zin_kj - zin_kjm1,
                                                      zout_kj - zout_kjm1,
                                                      h_kj - h_kjm1,
                                                      u_kj - u_kjm1)
            if diff_vari_norm_kj <= min(
                    1e-100, params["epo_k"]) or j == params["submaxiter"] - 1:
                A_submin, W_submin, V_submin, b_submin, c_submin = A_kj, W_kj, V_kj, b_kj, c_kj
                zin_submin, zout_submin, h_submin, u_submin, ht_submin, ut_submin = zin_kj, zout_kj, h_kj, u_kj, ht_kj, ut_kj
                break

        # Record results of the inner loop
        A_km1, W_km1, V_km1, b_km1, c_km1 = A_submin, W_submin, V_submin, b_submin, c_submin
        zin_km1, zout_km1, h_km1, u_km1, ht_km1, ut_km1 = zin_submin, zout_submin, h_submin, u_submin, ht_submin, ut_submin

        # Feasibility
        FeasVi_h.append(np.linalg.norm(h_submin - toolfunc.ELU(u_submin)))
        FeasVi_u.append(
            np.linalg.norm(
                u_submin -
                ALfunc.PsiX(ht_submin, x_trainset, zin_submin, Nx, Nh, T)))

        # Update Lagrangian multipliers
        params["xi1"] += params["gamma1"] * (u_submin - ALfunc.PsiX(
            ht_submin, x_trainset, zin_submin, Nx, Nh, T))
        params["xi2"] += params["gamma2"] * (h_submin - toolfunc.ELU(u_submin))

        # Determine stop criterion
        if params["epo_k"] < params["epo_star"] or params[
                "gamma1"] > gamma_max or params[
                    "gamma2"] > gamma_max or k == maxiter - 1:
            end_k = time.perf_counter()
            time_ALMRNN.append(end_k - start_k)

            _, _, _, _, y_hat_k = ALfunc.CalculY(x_trainset, A_submin,
                                                 W_submin, V_submin, b_submin,
                                                 c_submin, T, Nh, Nx, Ny)
            TrainErr.append((1 / T) * np.sum((y_hat_k - y_trainset)**2))

            u_test_bar, ut_test_bar, h_test_bar, ht_test_bar, y_test_hat = ALfunc.CalculY(
                x_testset, A_submin, W_submin, V_submin, b_submin, c_submin,
                T_test, Nh, Nx, Ny)
            TestErr.append((1 / T_test) * np.sum((y_test_hat - y_testset)**2))
            y_auxi = ALfunc.CalculY_auxi(zout_submin, ht_submin, Nh, Ny, T)

            regvalue = ALfunc.ValueRegula(A_submin, W_submin, V_submin,
                                          b_submin, c_submin, u_submin,
                                          params["lambda1"], params["lambda2"],
                                          params["lambda3"], params["lambda4"],
                                          params["lambda5"], params["lambda6"])
            lossALM = ALfunc.ValueLoss(
                y_trainset, y_auxi, Ny, T) + regvalue + ALfunc.ValueALMterm(
                    x_trainset, ht_submin, h_submin, u_submin, zin_submin,
                    params["xi1"], params["xi2"], params["gamma1"],
                    params["gamma2"], Nx, Nh, T)
            funcval_k.append(lossALM.item())

            if print_option == 2:
                if k == 0:
                    # Print header only once
                    print(
                        f"\n{'Iteration':<10}{'TrainErr':<15}{'TestErr':<15}{'FeasVi_h':<15}{'FeasVi_u':<15}{'Time (s)':<10}"
                    )
                    print("-" * 70)
                print(
                    f"{k+1:<10}{TrainErr[-1]:<15.4f}{TestErr[-1]:<15.4f}{FeasVi_h[-1]:<15.4f}{FeasVi_u[-1]:<15.4f}{time_ALMRNN[-1]:<10.4f}"
                )

            break

        if k > params["ALM_m"]:
            left = max(FeasVi_h[k], FeasVi_u[k])
            right = max(
                np.append(FeasVi_h[(k - params["ALM_m"]):k],
                          FeasVi_u[(k - params["ALM_m"]):k]))

            if left <= params["eta1"] * right:
                params["epo_k"] *= params["eta4"]
            else:
                params["gamma1"] = max(
                    params["gamma1"] / params["eta2"],
                    np.linalg.norm(params["xi1"])**(1 + params["eta3"]),
                    np.linalg.norm(params["xi2"])**(1 + params["eta3"]))
                params["gamma2"] = params["gamma1"]
                params["epo_k"] *= params["eta4"]

        gamma1_k.append(params["gamma1"])
        epo_k_recording.append(params["epo_k"])

        # Record time
        end_k = time.perf_counter()
        time_ALMRNN.append(end_k - start_k)

        # Record Loss function
        y_auxi = ALfunc.CalculY_auxi(zout_submin, ht_submin, Nh, Ny, T)
        regvalue = ALfunc.ValueRegula(A_submin, W_submin, V_submin, b_submin,
                                      c_submin, u_submin, params["lambda1"],
                                      params["lambda2"], params["lambda3"],
                                      params["lambda4"], params["lambda5"],
                                      params["lambda6"])
        lossALM = ALfunc.ValueLoss(
            y_trainset, y_auxi, Ny, T) + regvalue + ALfunc.ValueALMterm(
                x_trainset, ht_submin, h_submin, u_submin, zin_submin,
                params["xi1"], params["xi2"], params["gamma1"],
                params["gamma2"], Nx, Nh, T)
        funcval_k.append(lossALM.item())

        # Calculate Loss bar
        u_bar, ut_bar, h_bar, ht_bar, y_hat_k = ALfunc.CalculY(
            x_trainset, A_submin, W_submin, V_submin, b_submin, c_submin, T,
            Nh, Nx, Ny)
        valueloss_hat = ALfunc.ValueLoss(y_trainset, y_hat_k, Ny, T)
        lossALM_bar = valueloss_hat + regvalue

        # Calculate Training error
        TrainErr.append((1 / T) * np.sum((y_hat_k - y_trainset)**2))

        # Test error
        u_test_bar, ut_test_bar, h_test_bar, ht_test_bar, y_test_hat = ALfunc.CalculY(
            x_testset, A_submin, W_submin, V_submin, b_submin, c_submin,
            T_test, Nh, Nx, Ny)
        TestErr.append((1 / T_test) * np.sum((y_test_hat - y_testset)**2))

        if print_option == 2:
            if k == 0:
                # Print header only once
                print(
                    f"\n{'Iteration':<10}{'TrainErr':<15}{'TestErr':<15}{'FeasVi_h':<15}{'FeasVi_u':<15}{'Time (s)':<10}"
                )
                print("-" * 70)
            print(
                f"{k+1:<10}{TrainErr[-1]:<15.4f}{TestErr[-1]:<15.4f}{FeasVi_h[-1]:<15.4f}{FeasVi_u[-1]:<15.4f}{time_ALMRNN[-1]:<10.4f}"
            )

    # Prepare results to return
    results = {
        "TrainErr": TrainErr,
        "TestErr": TestErr,
        "time": time_ALMRNN,
        "FeasVi_h": FeasVi_h,
        "FeasVi_u": FeasVi_u
    }
    if print_option == 1:
        print(f"\nFinal Results:\n{'-' * 40}")
        print(f"TrainErr   : {TrainErr[-1]:.4f}")
        print(f"TestErr    : {TestErr[-1]:.4f}")
        print(f"FeasVi_h   : {FeasVi_h[-1]:.4f}")
        print(f"FeasVi_u   : {FeasVi_u[-1]:.4f}")
        print(f"{'-' * 40}\n")

    return results
