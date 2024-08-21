from ALM_BCD_ReLU import alm_bcd_ReLU_optimization
from ALM_BCD_ELU import alm_bcd_ELU_optimization
from gen_synthetic_dataset import generate_synthetic_dataset

#%% Import dataset
# Import real-world dataset
dataset_name = "clean_SP500"
Ny = 1
Nh = 20

# Import synthetic datasets
# Nh = 20
# Nx = 20
# Ny = 10
# T_total = 100
# generate_synthetic_dataset(
#     Nh, Nx, Ny, T_total,
#     mean_true=0, stddev_true=0.05,
#     e_mean=0, e_stddev=1e-3,
#     seed=123456, standardize=True
# )
# dataset_name = f"./SynDataset_Nh{Nh}_Nx{Nx}_Ny{Ny}_T{T_total}"

#%% Use ALM_BCD to train RNNs when setting activation function as ReLU
results = alm_bcd_ReLU_optimization(dataset_name=dataset_name,
                                    Ny=Ny,
                                    Nh=Nh,
                                    maxiter=5,
                                    submaxiter=200,
                                    distribution_type="Gaussian",
                                    mean=0,
                                    std=1e-3,
                                    gamma0=1,
                                    lambda1=0.01,
                                    lambda2=0.01,
                                    lambda3=0.01,
                                    lambda4=0.01,
                                    lambda5=0.01,
                                    lambda6=1e-8,
                                    eta1=0.99,
                                    eta2=5 / 6,
                                    eta3=0.01,
                                    eta4=5 / 6,
                                    epo_0=0.1,
                                    epo_star=1e-20,
                                    sigma=1e-8,
                                    ALM_m=1,
                                    print_option=2)

results = alm_bcd_ELU_optimization(dataset_name=dataset_name,
                                   Ny=Ny,
                                   Nh=Nh,
                                   maxiter=5,
                                   submaxiter=200,
                                   distribution_type="Gaussian",
                                   mean=0,
                                   std=1e-3,
                                   gamma0=1,
                                   lambda1=0.01,
                                   lambda2=0.01,
                                   lambda3=0.01,
                                   lambda4=0.01,
                                   lambda5=0.01,
                                   lambda6=1e-8,
                                   eta1=0.99,
                                   eta2=5 / 6,
                                   eta3=0.01,
                                   eta4=5 / 6,
                                   epo_0=0.1,
                                   epo_star=1e-20,
                                   sigma=1e-8,
                                   ALM_m=1,
                                   print_option=2)
