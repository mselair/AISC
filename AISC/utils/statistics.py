import numpy as np


def kl_divergence(mu1, std1, mu2, std2):
    # https://en.wikipedia.org/wiki/Normal_distribution
    return 0.5 * ((std1/std2)**2 + ((mu2-mu1)**2 / std2**2) -1 + 2*np.log(std2/std1))

def kl_divergence_mv(mu1, var1, mu2, var2):
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    # https://en.wikipedia.org/wiki/Trace_(linear_algebra)
    return 0.5 * ((np.trace(np.dot(np.linalg.inv(var2), var1))) + np.dot(np.dot((mu2 - mu1), np.linalg.inv(var2)), (mu2-mu1).T) - mu1.shape[1] + np.log(np.linalg.det(var2)/np.linalg.det(var1)))[0, 0]

def combine_gauss_distributions(mu1, std1, N1, mu2, std2, N2):
    c1 = N1 / (N1 + N2)
    c2 = N2 / (N1 + N2)
    mu_combined = (mu1 * c1) + (mu2 * c2)
    std_combined = np.sqrt(
        (N1*std1**2 + N2*std2**2 + N1*((mu1 - mu_combined)**2) + N2*((mu2 - mu_combined)**2)) / (N1+N2)
    ) # https://www.emathzone.com/tutorials/basic-statistics/combined-variance.html - top; not simplified

    # np.sqrt((N1*(std1**2) + N2*(std2**2) + N1*N2*(mu2-mu1)**2/(N1+N2)) / (N1+N2)) # https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf - 1.4

    return mu_combined, std_combined


def combine_mvgauss_distributions(mu1, var1, N1, mu2, var2, N2):
    c1 = N1 / (N1 + N2)
    c2 = N2 / (N1 + N2)
    mu_combined = (mu1 * c1) + (mu2 * c2)
    var_combined = (N1*(var1) + N2*(var2) + N1*N2*(mu2-mu1)**2/(N1+N2)) / (N1+N2) # np.sqrt((N1*(std1**2) + N2*(std2**2) + N1*N2*(mu2-mu1)**2/(N1+N2)) / (N1+N2)) # https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2008/086212.pdf
    for k1 in range(var_combined.shape[0]):
        for k2 in range(k1+1, var_combined.shape[0]):
            var_combined[k2, k1] = var_combined[k1, k2]
    return mu_combined, var_combined
