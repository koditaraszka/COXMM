# COXMM

This is an implementation of COX proportional hazard Mixed Model (frailty model)

While much of the implemenation follows that of COXMEG and it's notation, it does not utilize any of the speedups.
We note that our method is more scalable and efficient than COXMEG when using semidefinite relatedness matrixes (GRM).
We expect that samples of 30,000 can be ran on 64G of RAM in ~3 hours.

Through simulations, we found that the approximations bias the tau parameter which is fine for GWAS but not for estimating heritability.

We provide a correction 2*tau/(1+tau) for when there there is assumed to be noise (environmental effects) on the liability of having the disease. 

While we provide the standard error directly computed from the second derivative, previous work has shown that this underestimates the true standard error. We therefore provide an option to do block jackknifing.

To see various simulations download the h2_shinyapp and run it.
