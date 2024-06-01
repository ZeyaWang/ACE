library(lsa)
library(reticulate)
library(fpc)
library(R.utils)

np <- import("numpy")
source('helper.r')

args = as.character(commandArgs(trailingOnly = T))
nargs = length(args)

task = args[1]
key = args[2]


# WBT
WBT1 <- function(x,cl,P,s,vv) {
  result <- tryCatch({
    Indices.WBT(x,cl,P,s,vv)$ccc
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}

WBT <- function(x,cl,P,s,vv) {
  result <- tryCatch({
    withTimeout({WBT1(x,cl,P,s,vv)}, timeout = 600)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}



# SDbw1
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
    withTimeout({SDbw1(x, cl)}, timeout = 600)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}



# CDbw
CDbw1<-function(x, cl) {
  result <- tryCatch({
    cdbw(x, cl)$cdbw
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}
CDbw<-function(x, cl) {
  result <- tryCatch({
    withTimeout({CDbw1(x, cl)}, timeout = 600)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}




# dunn
# Index.dunn(md, cl1, Data=jeu, method=NULL)
Dunn1<-function(md, cl, Data, method) {
  result <- tryCatch({
    Index.dunn(md, cl, Data, method)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}
Dunn<-function(md, cl, Data, method) {
  result <- tryCatch({
    withTimeout({Dunn1(md, cl, Data, method)}, timeout = 600)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}


# cindex
#Indice.cindex(d=md, cl=cl1)
Cindex1<-function(d,cl) {
  result <- tryCatch({
    Indice.cindex(d, cl)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}
Cindex<-function(d,cl) {
  result <- tryCatch({
    withTimeout({Cindex1(d,cl)}, timeout = 600)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}


# db
#Indice.DB(x=jeu, cl=cl1, d = NULL, centrotypes = "centroids", p = 2, q = 2)$DB 
DB1<-function(x,cl) {
  result <- tryCatch({
    Indice.DB(x=x, cl=cl, d = NULL, centrotypes = "centroids", p = 2, q = 2)$DB 
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}
DB<-function(x,cl) {
  result <- tryCatch({
    withTimeout({DB1(x,cl)}, timeout = 600)
  }, error = function(e) {
    cat("Error:", conditionMessage(e), "\n")
    return(NA)
  })
  return(result)
}



file = paste0(task,'/raw_tmp/key_',key, '.npz')
data = np$load(file)


jeu=data$f[["jeu"]]
print(dim(jeu))
TT=data$f[["TT"]]
ss=as.vector(data$f[["ss"]])
vv=as.vector(data$f[["vv"]])
md=data$f[["md"]]
cmd=data$f[["cmd"]]
labelset = data$f[['labelset']]

# match label to make it 1-indexed
cl1 = labelset
unique_cl1 = unique(cl1)
indices = match(cl1, unique_cl1)
cl1 = indices
print(unique(cl1))


if (grepl('COIL-100', key)){
  print('skip for coil-100')
  ccc=NA
}else{
  ccc = WBT(x=jeu, cl=cl1, P=TT,s=ss,vv=vv) #max
}

print('ccc')
dunn = Dunn(md, cl1, Data=jeu, method=NULL) #max Index.dunn(md, cl1); depend on md matrix
dunn2 = Dunn(cmd, cl1, Data=jeu, method=NULL) #max Index.dunn(md, cl1); depend on md matrix
print('dunn')
cind = - Cindex(d=md, cl=cl1) #min #depend on md matrix
cind2 = - Cindex(d=cmd, cl=cl1) #min #depend on md matrix
print('cind')
#db = - Indice.DB(x=jeu, cl=cl1, d = NULL, centrotypes = "centroids", p = 2, q = 2)$DB #min #need data
db = - DB(x=jeu, cl=cl1) #min #need data
print('db')
sdbw = - SDbw(jeu, cl1) #min # need data
print('sdbw')
ccdbw = CDbw(jeu, cl1) #max # need data
print('cdbw')
print('done')


np$savez(paste0(task,"/raw_tmp/rr_", key, ".npz"), ccc=ccc,
  dunn=dunn, cind=cind,
  dunn2=dunn2, cind2=cind2,
  db=db,sdbw=sdbw, ccdbw=ccdbw,
  models=args)
