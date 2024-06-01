library(lsa)
library(reticulate)
library(fpc)
library(R.utils)

np <- import("numpy")
source('helper.r')

args = as.character(commandArgs(trailingOnly = T))
nargs = length(args)

task = args[1]
m = args[2]
key = args[3]



WBT <- function(x,cl,P,s,vv) {
  result <- tryCatch({
    Indices.WBT(x,cl,P,s,vv)$ccc
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}

SDbw1<-function(x, cl) {
  result <- tryCatch({
    Index.SDbw(x, cl)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}


SDbw<-function(x, cl) {
  result <- tryCatch({
    withTimeout({SDbw1(x, cl)}, timeout = 120)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}

CDbw<-function(x, cl) {
  result <- tryCatch({
    withTimeout({cdbw(x, cl)$cdbw}, timeout = 120)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}


file = paste0(task,'/tmp/',m,'.npz')
print(file)
data = np$load(file)
jeu=data$f[["jeu"]]
print(dim(jeu))
TT=data$f[["TT"]]
ss=as.vector(data$f[["ss"]])
vv=as.vector(data$f[["vv"]])
md=data$f[["md"]]
cmd=data$f[["cmd"]]

file = paste0(task,'/tmp/',m,'_', key, '.npz')
print(file)
data = np$load(file)
labelset = data$f[['labelset']]

cl1 = labelset
unique_cl1 = unique(cl1)
indices = match(cl1, unique_cl1)
cl1 = indices
print(unique(cl1))



ccc = WBT(x=jeu, cl=cl1, P=TT,s=ss,vv=vv) #max
print('ccc')
dunn = Index.dunn(md, cl1, Data=jeu, method=NULL) #max Index.dunn(md, cl1); depend on md matrix
dunn2 = Index.dunn(cmd, cl1, Data=jeu, method=NULL) #max Index.dunn(md, cl1); depend on md matrix
  print('dunn')
cind = - Indice.cindex(d=md, cl=cl1) #min #depend on md matrix
cind2 = - Indice.cindex(d=cmd, cl=cl1) #min #depend on md matrix
  print('cind')
db = - Indice.DB(x=jeu, cl=cl1, d = NULL, centrotypes = "centroids", p = 2, q = 2)$DB #min #need data
  print('db')
sdbw = - SDbw(jeu, cl1) #min # need data
print('sdbw')
ccdbw = CDbw(jeu, cl1) #max # need data
print('cdbw')
  
print('done')



np$savez(paste0(task,"/tmp/rr_", m, "_", key, ".npz"), ccc=ccc,
  dunn=dunn, cind=cind,
  dunn2=dunn2, cind2=cind2,
  db=db,sdbw=sdbw, ccdbw=ccdbw,
  models=args)
