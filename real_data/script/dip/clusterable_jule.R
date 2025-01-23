library(clusterability)
library(rhdf5)
library(reticulate)

# model1 = 'UUMist0.0010.9'
# model1 = 'UUMist0.050.2'
np <- import("numpy")
args = as.character(commandArgs(trailingOnly = T))
nargs = length(args)
ds = args[1]
nargs = nargs - 1

pvalues1 <- c()
pvalues2 <- c()

for(i in 1:nargs){
    model = args[i+1]
    feature = h5read(paste0('feature', model,'.h5'), 'feature')
    jeu = t(feature)
    print(dim(jeu))
    print(model)
    jsd=apply(jeu,2,sd)
    if(0 %in% jsd){ 
       clust_result1 <- clusterabilitytest(jeu, "dip", pca_scale = FALSE)
      pvalue1 <- clust_result1[["pvalue"]] 
    }else{
      clust_result1 <- clusterabilitytest(jeu, "dip")
      pvalue1 <- clust_result1[["pvalue"]] 
    }


  pvalues1 <- c(pvalues1, pvalue1)
  names(pvalues1)[length(pvalues1)] <- model
}

np$savez(paste0("dip_", ds, ".npz"),
         pvalues1=pvalues1,
         models=args)