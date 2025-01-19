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
    #data=np$load(paste0('rload/',model,'.npz'))
    #jeu=data$f[["jeu"]]
    data=np$load(model)
    jeu=data$f[['pro_features']]
    print(dim(jeu))
    print(model)
    jsd=apply(jeu,2,sd)
    # res <- get_clust_tendency(jeu, n = nrow(jeu)-1,
#                           graph = FALSE)
# res$hopkins_stat
# clust_result2 <- clusterabilitytest(jeu, "silverman", reduction = "distance",
#                                     s_m = 1000, s_setseed = 12345)
    #if(model%in%c("outputYTF_41_0.01_1.0_1.0.npz","outputYTF_41_0.01_10.0_1.0.npz", "outputYTF_41_0.005_1.0_1.0.npz")){
    if(0 %in% jsd){ 
       clust_result1 <- clusterabilitytest(jeu, "dip", pca_scale = FALSE)
      pvalue1 <- clust_result1[["pvalue"]] 
    }else{
      clust_result1 <- clusterabilitytest(jeu, "dip")
      pvalue1 <- clust_result1[["pvalue"]] 
    }

  #clust_result2 <- clusterabilitytest(jeu, "dip", reduction = "distance", distance_standardize = "NONE")
  #pvalue2 <- clust_result2[["pvalue"]]
  #print(c(model,pvalue1,pvalue2))
  pvalues1 <- c(pvalues1, pvalue1)
  #pvalues2 <- c(pvalues2, pvalue2)
  names(pvalues1)[length(pvalues1)] <- model
  #names(pvalues2)[length(pvalues2)] <- model
}

np$savez(paste0("dip_", ds, ".npz"),
         pvalues1=pvalues1,
         #pvalues2=pvalues2,
         models=args)