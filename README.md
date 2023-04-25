# COXPHMM

This is an implementation of Cox proportional hazard mixed model (frailty model)

While much of the implemenation follows that of COXMEG and it's notation, it does not utilize any of the speedups

Through simulations, we found that the approximations bias the tau parameter which is fine for GWAS but not for estimating heritability.
