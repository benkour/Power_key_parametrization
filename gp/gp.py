import gpytorch

class EmbeddingGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        # Mean function, before seeing the data, assume the function is flat
        # GP learns deviations via the kernel
        self.mean_module = gpytorch.means.ConstantMean() 
        # self.mean_module = gpytorch.means.LinearMean(train_x.shape[1]) -> Does not work at all
        # Covariance module, testing different Mean and Kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=train_x.shape[1]
            )
        )
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#     gpytorch.kernels.MaternKernel(
#         nu=2.5,
#         ard_num_dims=train_x.shape[1]
#     )
# )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)