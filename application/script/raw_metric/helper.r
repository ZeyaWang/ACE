# Description:
# This script is used as a helper file to calculate internal measures for the evaluation. 
# It is based on existing code from R package NbClust (https://github.com/cran/NbClust/) with minor modifications.
#
# Reference:
# Charrad, M., Ghazzali, N., Boiteau, V., & Niknafs, A. (2014). 
# NbClust: An R Package for Determining the Relevant Number of Clusters in a Data Set. 
# *Journal of Statistical Software, 61*(6), 1–36. 
# Available at: https://www.jstatsoft.org/v61/i06/
#
# License & Attribution:
# The original code remains under its respective license. Please refer to the original repository 
# for details. Any reuse should credit the original authors.
#
# Disclaimer:
# Provided "as is" without warranty. Use at your own risk.

Indice.Gap <- function (x1, x2, cl1, cl2, reference.distribution = "unif", B = 10, 
                        method = "ward.D2", d = NULL, centrotypes = "centroids") 
{
  GAP <- function(X, cl, referenceDistribution, B, method, d, centrotypes) 
  {
    set.seed(1)
    simgap <- function(Xvec) 
    {
      ma <- max(Xvec)
      mi <- min(Xvec)
      set.seed(1)
      Xout <- runif(length(Xvec), min = mi, max = ma)
      return(Xout)
    }
    pcsim <- function(X, d, centrotypes) 
    {
      if (centrotypes == "centroids") 
      {
        Xmm <- apply(X, 2, mean)
      }
      
      for (k in (1:dim(X)[2])) 
      {
        X[, k] <- X[, k] - Xmm[k]
      }
      ss <- svd(X)
      Xs <- X %*% ss$v
      Xnew <- apply(Xs, 2, simgap)
      Xt <- Xnew %*% t(ss$v)
      for (k in (1:dim(X)[2])) {
        Xt[, k] <- Xt[, k] + Xmm[k]
      }
      return(Xt)
    }
    if (is.null(dim(X))) 
    {
      dim(X) <- c(length(X), 1)
    }
    ClassNr <- max(cl)
    Wk0 <- 0
    WkB <- matrix(0, 1, B)
    for (bb in (1:B)) {
      if (reference.distribution == "unif") 
        Xnew <- apply(X, 2, simgap)
      else if (reference.distribution == "pc") 
        Xnew <- pcsim(X, d, centrotypes)
      else stop("Wrong reference distribution type")
      if (bb == 1) {
        pp <- cl
        if (ClassNr == length(cl)) 
          pp2 <- 1:ClassNr
        else if (method == "k-means") 
        { set.seed(1)
          pp2 <- kmeans(Xnew, ClassNr, 100)$cluster
        }
        else if (method == "single" || method == "complete" || 
                 method == "average" || method == "ward.D2" || 
                 method == "mcquitty" || method == "median" || 
                 method == "centroid"|| method=="ward.D") 
          pp2 <- cutree(hclust(dist(Xnew), method = method), 
                        ClassNr)
        else stop("Wrong clustering method")
        if (ClassNr > 1) {
          for (zz in (1:ClassNr)) {
            Xuse <- X[pp == zz, ]
            Wk0 <- Wk0 + sum(diag(var(Xuse))) * (length(pp[pp == 
                                                             zz]) - 1)/(dim(X)[1] - ClassNr)
            Xuse2 <- Xnew[pp2 == zz, ]
            WkB[1, bb] <- WkB[1, bb] + sum(diag(var(Xuse2))) * 
              (length(pp2[pp2 == zz]) - 1)/(dim(X)[1] - 
                                              ClassNr)
          }
        }
        if (ClassNr == 1) 
        {
          Wk0 <- sum(diag(var(X)))
          WkB[1, bb] <- sum(diag(var(Xnew)))
        }
      }
      if (bb > 1) {
        if (ClassNr == length(cl)) 
          pp2 <- 1:ClassNr
        else if (method == "k-means")
        {
          set.seed(1)
          pp2 <- kmeans(Xnew, ClassNr, 100)$cluster
        }
        else if (method == "single" || method == "complete" || 
                 method == "average" || method == "ward.D2" || 
                 method == "mcquitty" || method == "median" || 
                 method == "centroid"||method == "ward.D") 
          pp2 <- cutree(hclust(dist(Xnew), method = method), 
                        ClassNr)
        else stop("Wrong clustering method")
        if (ClassNr > 1) {
          for (zz in (1:ClassNr)) {
            Xuse2 <- Xnew[pp2 == zz, ]
            WkB[1, bb] <- WkB[1, bb] + sum(diag(var(Xuse2))) * 
              length(pp2[pp2 == zz])/(dim(X)[1] - ClassNr)
          }
        }
        if (ClassNr == 1) {
          WkB[1, bb] <- sum(diag(var(Xnew)))
        }
      }
    }
    Sgap <- mean(log(WkB[1, ])) - log(Wk0)
    Sdgap <- sqrt(1 + 1/B) * sqrt(var(log(WkB[1, ]))) * sqrt((B - 
                                                                1)/B)
    resul <- list(Sgap = Sgap, Sdgap = Sdgap)
    resul
  }
  if (sum(c("centroids", "medoids") == centrotypes) == 0) 
    stop("Wrong centrotypes argument")
  if ("medoids" == centrotypes && is.null(d)) 
    stop("For argument centrotypes = 'medoids' d can not be null")
  if (!is.null(d)) {
    if (!is.matrix(d)) {
      d <- as.matrix(d)
    }
    row.names(d) <- row.names(x1)
  }
  gap1 <- GAP(as.matrix(x1), cl1, reference.distribution, B, method, 
              d, centrotypes)
  gap <- gap1$Sgap
  gap2 <- GAP(as.matrix(x2), cl2, reference.distribution, B, method, 
              d, centrotypes)
  diffu <- gap - (gap2$Sgap - gap2$Sdgap)
  resul <- list(gap = gap, diffu = diffu)
  resul
}



Indices.WBT <- function(x,cl,P,s,vv) 
{
  n <- dim(x)[1]
  pp <- dim(x)[2]
  qq <- max(cl)
  z <- matrix(0,ncol=qq,nrow=n)
  clX <- as.matrix(cl)

  for (i in 1:n)
    for (j in 1:qq)
    {
      z[i,j]==0
      if (clX[i,1]==j) 
      {z[i,j]=1}
    }

  xbar <- solve(t(z)%*%z)%*%t(z)%*%x
  B <- t(xbar)%*%t(z)%*%z%*%xbar
  W <- P-B
  marriot <- (qq^2)*det(W)
  trcovw <- sum(diag(cov(W)))
  tracew <- sum(diag(W))
  if(det(W)!=0)
     scott <- n*log(det(P)/det(W))
  else {cat("Error: division by zero!")}
  friedman <- sum(diag(solve(W)*B))
  rubin <- sum(diag(P))/sum(diag(W))
  
  R2 <- 1-sum(diag(W))/sum(diag(P))
  v1 <- 1
  u <- rep(0,pp)
  c <- (vv/(qq))^(1/pp)
  u <- s/c
  k1 <- sum((u>=1)==TRUE)
  p1 <- min(k1,qq-1)
  if (all(p1>0,p1<pp))
  {
    for (i in 1:p1)
    v1 <- v1*s[i]
    c <- (v1/(qq))^(1/p1)
    u <- s/c
    b1 <- sum(1/(n+u[1:p1]))
    b2 <- sum(u[p1+1:pp]^2/(n+u[p1+1:pp]),na.rm=TRUE)
    E_R2 <- 1-((b1+b2)/sum(u^2))*((n-qq)^2/n)*(1+4/n)
    ccc <- log((1-E_R2)/(1-R2))*(sqrt(n*p1/2)/((0.001+E_R2)^1.2))
  }else 
  {
    b1 <- sum(1/(n+u))
    E_R2 <- 1-(b1/sum(u^2))*((n-qq)^2/n)*(1+4/n)
    ccc <- log((1-E_R2)/(1-R2))*(sqrt(n*pp/2)/((0.001+E_R2)^1.2))
  }
 results <- list(ccc=ccc,scott=scott,marriot=marriot,trcovw=trcovw,tracew=tracew,friedman=friedman,rubin=rubin)
 return(results)
}

Index.dunn <- function(md, clusters, Data=NULL, method="euclidean")
    {
      
      distance <- as.matrix(md)
      nc <- max(clusters)
      interClust <- matrix(NA, nc, nc)
      intraClust <- rep(NA, nc)
      
      for (i in 1:nc) 
      {
        c1 <- which(clusters==i)
        for (j in i:nc) {
          if (j==i) intraClust[i] <- max(distance[c1,c1])
          if (j>i) {
            c2 <- which(clusters==j)
            interClust[i,j] <- min(distance[c1,c2])
          }
        }
      }
      dunn <- min(interClust,na.rm=TRUE)/max(intraClust)
      return(dunn)
    }

Indice.cindex <- function (d, cl) 
{
    d <- data.matrix(d)
    DU <- 0
    r <- 0
    v_max <- array(1, max(cl))
    v_min <- array(1, max(cl))
    for (i in 1:max(cl)) {
        n <- sum(cl == i)
        if (n > 1) {
            t <- d[cl == i, cl == i]
            DU = DU + sum(t)/2
            v_max[i] = max(t)
            if (sum(t == 0) == n) 
                v_min[i] <- min(t[t != 0])
            else v_min[i] <- 0
            r <- r + n * (n - 1)/2
        }
    }
    Dmin = min(v_min)
    Dmax = max(v_max)
    if (Dmin == Dmax) 
        result <- NA
    else result <- (DU - r * Dmin)/(Dmax * r - Dmin * r)
    result
}


Indice.ptbiserial <- function (x,md,cl1)
{
  nn <- dim(x)[1]
  pp <- dim(x)[2]

  md2 <- as.matrix(md)
  m01 <- array(NA, c(nn,nn))
  nbr <- (nn*(nn-1))/2
  pb <- array(0,c(nbr,2))
  
  m3 <- 1
  for (m1 in 2:nn)
  {
       m12 <- m1-1
     for (m2 in 1:m12)
     {
    if (cl1[m1]==cl1[m2]) m01[m1,m2]<-0
    if (cl1[m1]!=cl1[m2]) m01[m1,m2]<-1
    pb[m3,1] <- m01[m1,m2]
    pb[m3,2] <- md2[m1,m2]
    m3 <- m3+1
     }
  }

  y <- pb[,1]
  x <- pb[,2] 

  biserial.cor <- function (x, y, use = c("all.obs", "complete.obs"), level = 1) 
  {
      if (!is.numeric(x)) 
          stop("'x' must be a numeric variable.\n")
      y <- as.factor(y)
      if (length(levs <- levels(y)) > 2) 
          stop("'y' must be a dichotomous variable.\n")
      if (length(x) != length(y)) 
          stop("'x' and 'y' do not have the same length")
      use <- match.arg(use)
      if (use == "complete.obs") {
          cc.ind <- complete.cases(x, y)
          x <- x[cc.ind]
          y <- y[cc.ind]
      }
      ind <- y == levs[level]
      diff.mu <- mean(x[ind]) - mean(x[!ind])
      prob <- mean(ind)
      diff.mu * sqrt(prob * (1 - prob))/sd(x)
  }

    ptbiserial <- biserial.cor(x=pb[,2], y=pb[,1], level = 2)
    return(ptbiserial)
}


Indice.DB <- function (x, cl, d = NULL, centrotypes = "centroids", p = 2, q = 2) 
{
    if (sum(c("centroids") == centrotypes) == 0) 
        stop("Wrong centrotypes argument")
    if (!is.null(d)) {
        if (!is.matrix(d)) {
            d <- as.matrix(d)
        }
        row.names(d) <- row.names(x)
    }
    if (is.null(dim(x))) {
        dim(x) <- c(length(x), 1)
    }
    x <- as.matrix(x)
    n <- length(cl)
    k <- max(cl)
    dAm <- d
    centers <- matrix(nrow = k, ncol = ncol(x))
    if (centrotypes == "centroids") {
        for (i in 1:k) {
            for (j in 1:ncol(x)) {
                centers[i, j] <- mean(x[cl == i, j])
            }
        }
    }
    else {
        stop("wrong centrotypes argument")
    }
    S <- rep(0, k)
    for (i in 1:k) {
        ind <- (cl == i)
        if (sum(ind) > 1) {
            centerI <- centers[i, ]
            centerI <- rep(centerI, sum(ind))
            centerI <- matrix(centerI, nrow = sum(ind), ncol = ncol(x), 
                byrow = TRUE)
            S[i] <- mean(sqrt(apply((x[ind, ] - centerI)^2, 1, 
                sum))^q)^(1/q)
        }
        else S[i] <- 0
    }
    M <- as.matrix(dist(centers, p = p))
    R <- array(Inf, c(k, k))
    r = rep(0, k)
    for (i in 1:k) {
        for (j in 1:k) {
            R[i, j] = (S[i] + S[j])/M[i, j]
        }
        r[i] = max(R[i, ][is.finite(R[i, ])])
    }
    DB = mean(r[is.finite(r)])
    resul <- list(DB = DB, r = r, R = R, d = M, S = S, centers = centers)
    resul
}


Index.sPlussMoins <- function (cl1,md)
{
    cn1 <- max(cl1)
    n1 <- length(cl1)
    dmat <- as.matrix(md)
    average.distance <- median.distance <- separation <- cluster.size <- within.dist1 <- between.dist1 <- numeric(0)
    separation.matrix <- matrix(0, ncol = cn1, nrow = cn1)
    di <- list()
    for (u in 1:cn1) {
        cluster.size[u] <- sum(cl1 == u)
        du <- as.dist(dmat[cl1 == u, cl1 == u])
        within.dist1 <- c(within.dist1, du)
        average.distance[u] <- mean(du)
        median.distance[u] <- median(du)
        bv <- numeric(0)
        for (v in 1:cn1) {
            if (v != u) {
                suv <- dmat[cl1 == u, cl1 == v]
                bv <- c(bv, suv)
                if (u < v) {
                  separation.matrix[u, v] <- separation.matrix[v,u] <- min(suv)
                  between.dist1 <- c(between.dist1, suv)
                }
            }
        }
    }

    nwithin1 <- length(within.dist1)
    nbetween1 <- length(between.dist1)
    meanwithin1 <- mean(within.dist1)
    meanbetween1 <- mean(between.dist1)
    
    s.plus <- s.moins <- 0 
    #s.moins<-sum(rank(c(within.dist1,between.dist1),ties="first")[1:nwithin1]-rank(within.dist1,ties="first"))
    #s.plus  <-sum(rank(c(-within.dist1,-between.dist1),ties="first")[1:nwithin1]-rank(-within.dist1,ties="first"))
    for (k in 1: nwithin1)
    {
      s.plus <- s.plus+(colSums(outer(between.dist1,within.dist1[k], ">")))
      s.moins <- s.moins+(colSums(outer(between.dist1,within.dist1[k], "<")))
    }    
    
    Index.Gamma <- (s.plus-s.moins)/(s.plus+s.moins)
    Index.Gplus <- (2*s.moins)/(n1*(n1-1))
    t.tau  <- (nwithin1*nbetween1)-(s.plus+s.moins)
    Index.Tau <- (s.plus-s.moins)/(((n1*(n1-1)/2-t.tau)*(n1*(n1-1)/2))^(1/2))

    results <- list(gamma=Index.Gamma, gplus=Index.Gplus, tau=Index.Tau)
    return(results)
}


centers<-function(cl,x)
{
    x <- as.matrix(x)
    n <- length(cl)
    k <- max(cl)
    centers <- matrix(nrow = k, ncol = ncol(x))
    {
        for (i in 1:k) 
        {
            for (j in 1:ncol(x)) 
            {
                centers[i, j] <- mean(x[cl == i, j])
            }
        }
    }
    return(centers)
}    

Average.scattering <- function (cl, x)
{
    x <- as.matrix(x)
    n <- length(cl)
    k <- max(cl)
    centers.matrix <- centers(cl,x)
    
    cluster.size <- numeric(0)  
    variance.clusters <- matrix(0, ncol = ncol(x), nrow = k)
    var <- matrix(0, ncol = ncol(x), nrow = k)
    
    for (u in 1:k) 
      cluster.size[u] <- sum(cl == u)

    for (u in 1:k) 
    {  
      for (j in 1:ncol(x)) 
      { 
         for(i in 1:n) 
         {               
           if(cl[i]==u)                   
              variance.clusters[u,j]<- variance.clusters[u,j]+(x[i, j]-centers.matrix[u,j])^2 
         }
      }            
    }

    for (u in 1:k) 
    {    
       for (j in 1:ncol(x)) 
          variance.clusters[u,j]= variance.clusters[u,j]/ cluster.size[u]   
    }
    
     
     variance.matrix <- numeric(0)
     for(j in 1:ncol(x)) 
        variance.matrix[j]=var(x[,j])*(n-1)/n

     
      Somme.variance.clusters<-0
      for (u in 1:k) 
         Somme.variance.clusters<-Somme.variance.clusters+sqrt((variance.clusters[u,]%*%(variance.clusters[u,])))
         

      # Standard deviation
      stdev<-(1/k)*sqrt(Somme.variance.clusters)
      
      #Average scattering for clusters  
      scat<- (1/k)* (Somme.variance.clusters /sqrt(variance.matrix %*% variance.matrix))
      
      scat <- list(stdev=stdev, centers=centers.matrix, variance.intraclusters= variance.clusters, scatt=scat)
      return(scat)
}

density.clusters<-function(cl, x)
{
   x <- as.matrix(x)
   k <- max(cl)
   n <- length(cl)
         
   distance <- matrix(0, ncol = 1, nrow = n)
   density <-  matrix(0, ncol = 1, nrow = k)
   centers.matrix<-centers(cl,x)
   stdev<-Average.scattering(cl,x)$stdev 
   for(i in 1:n) 
   {        
       u=1
       while(cl[i] != u )
          u<-u+1
       for (j in 1:ncol(x))   
       {               
           distance[i]<- distance[i]+(x[i,j]-centers.matrix[u,j])^2 
       }     
       distance[i]<-sqrt(distance[i])            
       if (distance[i] <= stdev)
          density[u]= density[u]+1                      
   }  
    dens<-list(distance=distance, density=density)    
    return(dens)          
 
}


density.bw<-function(cl, x)
{
   x <- as.matrix(x)
   k <- max(cl)
   n <- length(cl)   
   centers.matrix<-centers(cl,x)
   stdev<-Average.scattering(cl,x)$stdev 
   density.bw<- matrix(0, ncol = k, nrow = k)
   u<-1
   
   for(u in 1:k)
   {
     for(v in 1:k)
     {
       if(v!=u)
       {  
          distance<- matrix(0, ncol = 1, nrow = n)
          moy<-(centers.matrix[u,]+centers.matrix[v,])/2
          for(i in 1:n)
          {
            if((cl[i]==u)||(cl[i]==v))
            {
              for (j in 1:ncol(x))   
              {               
                 distance[i]<- distance[i]+(x[i,j]-moy[j])^2 
              }   
              distance[i]<- sqrt(distance[i])
              if(distance[i]<= stdev)
              {
                density.bw[u,v]<-density.bw[u,v]+1                  
              }  
            }           
          }
        }       
       }
      }
     density.clust<-density.clusters(cl,x)$density 
     S<-0
     for(u in 1:k)
       for(v in 1:k)
       {  
         if(max(density.clust[u], density.clust[v])!=0)
            S=S+ (density.bw[u,v]/max(density.clust[u], density.clust[v]))
       }   
     density.bw<-S/(k*(k-1))
     return(density.bw) 
  
 }      
     


Index.SDbw<-function(x, cl)
{
  x <- as.matrix(x)
  Scatt<-Average.scattering(cl,x)$scatt
  Dens.bw<-density.bw(cl,x)
  SDbw<-Scatt+Dens.bw
  return(SDbw)
}    

