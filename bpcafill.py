import pandas as pd, numpy as np
import sys

def table(x): 
    try: return pd.DataFrame(x)
    except: return pd.DataFrame(list(x.items()))

def bpca_fill(dX, max_iter = 200):
    y = dX.copy() # Just following the conventions used in the original paper by Oba et al. (2003)
    cols = y.columns.tolist()

    #--- Normalize `y` and convert to matrix format 
    means = y.mean(); sd = y.std(); y = (y-means)/sd
    y = np.matrix(y).astype(float) 

    #--- Initialization for Bayesian PCA imputation
    N,d = y.shape; q = d-1
    yest = np.copy(y); yest[np.isnan(yest)] = 0 

    missidx = {} # dictionary that contains NaN positions (row,col) of dX; {row : list of NaN col in row}
    bad = np.where(np.isnan(y))
    badrows = bad[0]; badcols = bad[1]
    for row in badrows: missidx[row] = []
    for link in range(len(badrows)): missidx[badrows[link]].append(badcols[link])

    nomissidx = {} # dictionary of indices complement to missidx
    good = np.where(~np.isnan(y))
    goodrows = good[0]; goodcols = good[1]
    for row in goodrows: nomissidx[row] = []
    for link in range(len(goodrows)): nomissidx[goodrows[link]].append(goodcols[link])

    ##--- Removing redundancies in row entries in `badrows` and `goodrows`
    gmiss = list(set(badrows)) # There are NaN(s) in these rows
    gnomiss = list(set(goodrows)) # None of these rows are completely NaN.

    covy = np.cov(yest.T) # covariance matrix of dX
    U, S, V = np.linalg.svd(np.matrix(covy)) # Singular Value Decomposition of the covariance matrix
    U = (U.T[0:q]).T;         S = S[0:q]*np.eye(q);           V = (V.T[0:q]).T

    mu = np.nanmean(np.copy(y), 0)
    W = U*np.sqrt(S)
    taumax = 1e20; taumin = 1e-20
    tau = 1/ (np.trace(covy)-np.trace(S));    tau = np.amax([np.amin([tau,taumax]),taumin])

    galpha0 = 1e-10;    balpha0 = 1;    alpha = (2*galpha0 + d)/(tau*np.diag(W.T*W)+2*galpha0/balpha0)
    gmu0  = 0.001;    btau0 = 1;    gtau0 = 1e-10;    SigW = np.eye(q)

    #--- End of bPCA initialization
    
    ### Outcome: N, d, q, yest, missidx, nomissidx, gmiss, gnomiss, 
    ###          mu, W, tau, galpha0, balpha0, alpha, gmu0, btau0, gtau0, SigW

    #--- Iteration

    ##--- Initialize `tau_old` and fill NaNs in `y` with `sys.maxsize`
    tau_old = 1000
    y = np.nan_to_num(y, nan = sys.maxsize)

    for epoch in range(max_iter):

        #--- Begin BPCA_dostep(): batch 1 step of Bayesian PCA EM algorithm
        Rx = np.eye(q)+tau*W.T*W+SigW;            Rxinv = np.linalg.inv(Rx)
        idx = gnomiss; n = len(idx)                  
        dy = y[idx,:] - np.tile(mu,(n,1));      x = tau * Rxinv * W.T * dy.T

        Td = dy.T*x.T;                            trS = np.sum(np.multiply(dy,dy))
        for n in range(len(gmiss)):
            i = gmiss[n]
            dyo = np.copy(y)[i,nomissidx[i]] - mu[nomissidx[i]]
            Wm = W[missidx[i],:];                                  Wo = W[nomissidx[i],:]
            Rxinv = np.linalg.inv( Rx - tau*Wm.T*Wm );             ex = tau * Wo.T * np.matrix(dyo).T;   x = Rxinv * ex
            dym = Wm * x;                                          dy = np.copy(y)[i,:]
            dy[nomissidx[i]] = dyo
            dy[missidx[i]] = dym.T
            yest[i,:] = dy + mu
            Td = Td + np.matrix(dy).T*x.T;                            Td[missidx[i],:] = Td[missidx[i],:] + Wm * Rxinv
            trS = trS + dy*np.matrix(dy).T +  len(missidx[i])/tau + np.trace( Wm * Rxinv * Wm.T )

        Td = Td/N;                trS = trS/N;

        ###--- Update outcome: (N, d, q,) yest(updated in n-loop above), (missidx, nomissidx, gmiss, gnomiss,)
        ###                 (mu,) W, tau, (galpha0, balpha0,) alpha, (gmu0, btau0, gtau0,) SigW

        Rxinv = np.linalg.inv(Rx); 
        Dw = Rxinv + tau*Td.T*W*Rxinv + np.diag(alpha)/N;    Dwinv = np.linalg.inv(Dw);

        W = Td * Dwinv;
        tau = (d+2*gtau0/N)/(trS-np.trace(Td.T*W)  + (mu*np.matrix(mu).T*gmu0+2*gtau0/btau0)/N)[0,0];
        SigW = Dwinv*(d/N);
        alpha = (2*galpha0 + d)/ (tau*np.diag(W.T*W)+np.diag(SigW)+2*galpha0/balpha0)

        dtau = np.log10(tau) - np.log10(tau_old)
        if np.abs(dtau) < 1e-4:  break;

        # Update `tau_old`
        tau_old = tau

    #--- Create imputed dataframe `dX_filled`
    
    dXb = pd.DataFrame(yest)
    dXb.columns = cols
    dXb = (dXb*sd)+means

    dXo = dX.copy()
    missing = dXo.isna()
    dXo[missing] = 0
    dXb[~missing] = 0

    dX_filled = dXo + dXb
    
    return dX_filled
