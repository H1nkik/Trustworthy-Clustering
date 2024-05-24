#!/data/users/zhouk/software/anaconda3/bin/python3
import numpy as np
#import pandas as pd
#import os
#os.chdir("./data") #your path
import torch
from metrics import cal_clustering_metric
import random
import data_loader as loader
import time
import warnings
warnings.filterwarnings('ignore')

def makeF(c, type=['simple', 'full', 'pairs'], pairs=None, Omega=True):
    if type == 'full':  # All the 2^c focal sets
        ii = np.arange(2**c)
        N = len(ii)
        F = np.zeros((N, c))
        CC = np.array([np.binary_repr(i, width=c) for i in range(N)])
        for i in range(N):
            F[i, :] = np.array([int(s) for s in CC[i]])
        F = F[:, ::-1]
    else:  # type = 'simple' or 'pairs'
        F = np.vstack((np.zeros(c), np.eye(c)))  # the empty set and the singletons
        if type == 'pairs':  # type = 'pairs'
            if pairs is None:  # pairs not specified: we take them all
                for i in range(c - 1):
                    for j in range(i + 1, c):
                        f = np.zeros(c)
                        f[[i, j]] = 1
                        F = np.vstack((F, f))
            else:  # pairs specified
                n = pairs.shape[0]
                for i in range(n):
                    f = np.zeros(c)
                    f[pairs[i, :]] = 1
                    F = np.vstack((F, f))
        if Omega and not ((type == "pairs") and (c == 2)) and not ((type == "simple") and (c == 1)):
            F = np.vstack((F, np.ones(c)))  # the whole frame
    return F


#---------------------- extractMass--------------------------------------------
def extractMass(mass, F, g=None, S=None, method=None, crit=None, Kmat=None, trace=None, D=None, W=None, J=None, param=None):
    n = mass.shape[0]
    c = F.shape[1]
    if any(F[0, :] == 1):
        F = np.vstack((np.zeros(c), F))  # add the empty set
        mass = np.hstack((np.zeros((n, 1)), mass))
    f = F.shape[0]
    card = np.sum(F, axis=1)
    conf = mass[:, 0]             # degree of conflict
    C = 1 / (1 - conf)
    mass_n = C[:, np.newaxis] * mass[:, 1:f]   # normalized mass function
    pl = np.matmul(mass, F)          # unnormalized plausibility
    pl_n = C[:, np.newaxis] * pl             # normalized plausibility
    p = pl / np.sum(pl, axis=1, keepdims=True)      # plausibility-derived probability
    bel = mass[:, card == 1]    # unnormalized belief
    bel_n = C[:, np.newaxis] * bel            # normalized belief
    y_pl = np.argmax(pl, axis=1)       # maximum plausibility cluster
    y_bel = np.argmax(bel, axis=1)     # maximum belief cluster
    Y = F[np.argmax(mass, axis=1), :]    # maximum mass set of clusters
    # non dominated elements
    Ynd = np.zeros((n, c))
    for i in range(n):
        ii = np.where(pl[i, :] >= bel[i, y_bel[i]])[0]
        Ynd[i, ii] = 1
    #P = F / card[:, np.newaxis]
    nonzero_card = np.where(card != 0)  
    P = np.zeros_like(F)
    P[nonzero_card] = F[nonzero_card] / card[nonzero_card, np.newaxis]
    P[0, :] = 0
    betp = np.matmul(mass, P)       # unnormalized pignistic probability
    betp_n = C[:, np.newaxis] * betp        # normalized pignistic probability
    lower_approx, upper_approx = [], []
    lower_approx_nd, upper_approx_nd = [], []
    nclus = np.sum(Y, axis=1)
    outlier = np.where(nclus == 0)[0]  # outliers
    nclus_nd = np.sum(Ynd, axis=1)
    for i in range(c):
        upper_approx.append(np.where(Y[:, i] == 1)[0])  # upper approximation
        lower_approx.append(np.where((Y[:, i] == 1) & (nclus == 1))[0])  # upper approximation
        upper_approx_nd.append(np.where(Ynd[:, i] == 1)[0])  # upper approximation
        lower_approx_nd.append(np.where((Ynd[:, i] == 1) & (nclus_nd == 1))[0])  # upper approximation
    # Nonspecificity
    card = np.concatenate(([c], card[1:f]))
    Card = np.tile(card, (n, 1))
    N = np.sum(np.log(Card) * mass) / np.log(c) / n
    clus = {'conf': conf, 'F': F, 'mass': mass, 'mass_n': mass_n, 'pl': pl, 'pl_n': pl_n, 'bel': bel, 'bel_n': bel_n,
            'y_pl': y_pl, 'y_bel': y_bel, 'Y': Y, 'betp': betp, 'betp_n': betp_n, 'p': p,
            'upper_approx': upper_approx, 'lower_approx': lower_approx, 'Ynd': Ynd,
            'upper_approx_nd': upper_approx_nd, 'lower_approx_nd': lower_approx_nd,
            'N': N, 'outlier': outlier , 'g': g, 'S': S,
            'crit': crit, 'Kmat': Kmat, 'trace': trace, 'D': D, 'method': method, 'W': W, 'J': J, 'param': param}
    return clus

def ecm(x, c, g0=None, type='full', pairs=None, Omega=True, ntrials=1, alpha=1, beta=2, delta=10,
        epsi=1e-2, init="kmeans", disp=True):
    #---------------------- initialisations --------------------------------------
    x = np.array(x)
    n = x.shape[0]
    d = x.shape[1]
    delta2 = delta ** 2
    if (ntrials > 1) and (g0 is not None):
        print('WARNING: ntrials>1 and g0 provided. Parameter ntrials set to 1.')
        ntrials = 1
    F = makeF(c, type, pairs, Omega) #R language: myF 
    f = F.shape[0]
    card = np.sum(F[1:f, :], axis=1)
    #------------------------ iterations--------------------------------
    Jbest = np.inf
    for itrial in range(ntrials):
        if g0 is None:
            if init == "kmeans":
                centroids, distortion = kmeans(x, c)
                g = centroids
            else:
                g = x[np.random.choice(n, c), :] + 0.1 * np.random.randn(c * d).reshape(c, d)
        else:
            g = g0
        pasfini = True
        Jold = np.inf
        gplus = np.zeros((f-1, d))
        iter = 0
        while pasfini:
            iter += 1
            for i in range(1, f):
                fi = F[i, :]
                truc = np.tile(fi, (d, 1)).T
                gplus[i-1, :] = np.sum(g * truc, axis=0) / np.sum(fi)
            # calculation of distances to centers
            D = np.zeros((n, f-1))
            for j in range(f-1):
                D[:, j] = np.nansum((x - np.tile(gplus[j, :], (n, 1))) ** 2, axis=1)
            # Calculation of masses
            m = np.zeros((n, f-1))
            for i in range(n):
                vect0 = D[i, :]
                for j in range(f-1):
                    vect1 = (np.tile(D[i, j], f-1) / vect0) ** (1 / (beta-1))
                    vect2 = np.tile(card[j] ** (alpha / (beta-1)), f-1) / (card ** (alpha / (beta-1)))
                    vect3 = vect1 * vect2
                    m[i, j] = 1 / (np.sum(vect3) + (card[j] ** alpha * D[i, j] / delta2) ** (1 / (beta-1)))
                    if np.isnan(m[i, j]):
                        m[i, j] = 1  # in case the initial prototypes are training vectors
            # Calculation of centers
            A = np.zeros((c, c))
            for k in range(c):
                for l in range(c):
                    truc = np.zeros(c)
                    truc[[k, l]] = 1
                    t = np.tile(truc, (f, 1))
                    indices = np.where(np.sum((F - t) - np.abs(F - t), axis=1) == 0)[0]   #indices of all Aj including wk and wl
                    indices = indices - 1
                    if len(indices) == 0:
                        A[l, k] = 0
                    else:
                        for jj in range(len(indices)):
                            j = indices[jj]
                            mj = m[:, j] ** beta
                            A[l, k] += np.sum(mj) * card[j] ** (alpha - 2)
            # Construction of the B matrix
            B = np.zeros((c, d))
            for l in range(c):
                truc = np.zeros(c)
                truc[l] = 1
                t = np.tile(truc, (f, 1))
                indices = np.where(np.sum((F - t) - np.abs(F - t), axis=1) == 0)[0]   # indices of all Aj including wl
                indices = indices - 1
                mi = np.tile(card[indices] ** (alpha - 1), (n, 1)) * m[:, indices] ** beta
                s = np.sum(mi, axis=1)
                mats = np.tile(s.reshape(n, 1), (1, d))
                xim = x * mats
                B[l, :] = np.sum(xim, axis=0)
            g = np.linalg.solve(A, B)
            mvide = 1 - np.sum(m, axis=1)
            J = np.nansum((m**beta)*D[:,:f-1]* np.tile(card[:f - 1] ** alpha, (n, 1))) + delta2 * np.nansum(mvide[:f-1]** beta)
            if disp:
                print("Iterations/Total loss: ",[iter, J])
            pasfini = (np.abs(J - Jold) > epsi)
            Jold = J
        if J < Jbest:
            Jbest = J
            mbest = m
            gbest = g
        res = np.array([itrial, J, Jbest])
        res = np.squeeze(res)
        if ntrials > 1:
            print(res)
    m = np.concatenate((1 - np.sum(mbest, axis=1).reshape(n, 1), mbest), axis=1)
    clus = extractMass(m, F, g=gbest, method="ecm", crit=Jbest, param={'alpha': alpha, 'beta': beta, 'delta': delta})
    return clus

class Dataset(torch.utils.data.Dataset): 
    def __init__(self, X):
        self.X = X

    def __getitem__(self, idx):
        return self.X[:, idx], idx

    def __len__(self):
        return self.X.shape[1]
class PretrainDoubleLayer(torch.nn.Module):    
    #pretrain initialization
    def __init__(self, X, dim, device, act, batch_size=128, lr=10**-3): #self指代父类torch.nn.Module
        super(PretrainDoubleLayer, self).__init__() 
        self.X = X
        self.dim = dim
        self.lr = lr
        self.device = device
        self.enc = torch.nn.Linear(X.shape[0], self.dim) #full-connect
        self.dec = torch.nn.Linear(self.dim, X.shape[0]) #full-connect
        self.batch_size = batch_size
        self.act = act #activation function  
        
    def forward(self, x):
        if self.act is not None: 
            z = self.act(self.enc(x))
            return z, self.act(self.dec(z))
        else:                    
            z = self.enc(x)
            return z, self.dec(z)

    def _build_loss(self, x, recons_x): #J_reconstruction 
        n = x.shape[0]
        return torch.norm(x-recons_x, p='fro')**2 / n #Frobenius norm

    def run(self):
        self.to(self.device) #to
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) 
        train_loader = torch.utils.data.DataLoader(Dataset(self.X), batch_size=self.batch_size, shuffle=True)
        loss = 0
        for epoch in range(10):
            for i, batch in enumerate(train_loader): #enumerate 
                x, _ = batch #x
                optimizer.zero_grad() 
                _, recons_x = self(x) #recons_x
                loss = self._build_loss(x, recons_x) 
                loss.backward() 
                optimizer.step() 
            print('epoch-{}: loss={}'.format(epoch, loss.item()))
        Z, _ = self(self.X.t()) 
        return Z.t() 
    
class DeepEvidentialCMeans(torch.nn.Module):
    def __init__(self,X,labels,layers=None,lam=0.99,alpha=1,beta=1.1,delta2=9,type='pairs',pairs=None,Omega=True,lr=10**-3,device=None, batch_size=128):
        super(DeepEvidentialCMeans, self).__init__() 
        if layers is None:
            layers = [X.shape[0], 500, 300]
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #cuda:0
        self.layers = layers
        self.device = device
        if not isinstance(X, torch.Tensor): #tensor
            X = torch.Tensor(X)
        self.X = X.to(device)
        self.labels = labels
        self.alpha = alpha
        self.lam = lam
        self.beta = beta
        self.delta2 = delta2
        self.c = len(np.unique(labels)) #number of prototype: c
        self.F = makeF(self.c,type, pairs, Omega) #power set framework: 2^c * c
        self.f = self.F.shape[0] #2^c
        self.card = np.sum(self.F[1:self.f,:],axis=1)
        self.batch_size = batch_size
        self.lr = lr
        self._build_up()

    def _build_up(self): 
        self.act = torch.tanh
        self.enc1 = torch.nn.Linear(self.layers[0], self.layers[1]) 
        self.enc2 = torch.nn.Linear(self.layers[1], self.layers[2]) 
        self.dec1 = torch.nn.Linear(self.layers[2], self.layers[1]) 
        self.dec2 = torch.nn.Linear(self.layers[1], self.layers[0]) 

    def forward(self, x): 
        z = self.act(self.enc1(x)) 
        z = self.act(self.enc2(z)) 
        recons_x = self.act(self.dec1(z)) 
        recons_x = self.act(self.dec2(recons_x))
        return z, recons_x

    def _build_loss(self, z, x, d, m, recons_x):
        n = x.shape[0]
        mvide = 1 - np.sum(m, axis=1)
        # J_reconstruction
        loss1 = 1/2 * torch.norm(x - recons_x, p='fro') ** 2 /n
        # J_partition
        loss2 = self.lam/2*(np.nansum((m ** self.beta) * d[:, :self.f - 1] * np.tile(self.card[:self.f - 1] ** self.alpha, (n, 1))) + self.delta2 * np.nansum(mvide[:self.f - 1] ** self.beta))/ n
        
        loss =loss1+loss2+ 0.00001 * (self.enc1.weight.norm()**2 + self.enc1.bias.norm()**2) / n
        loss =loss1+loss2+ 0.00001 * (self.enc2.weight.norm()**2 + self.enc2.bias.norm()**2) / n
        loss =loss1+loss2+ 0.00001 * (self.dec1.weight.norm()**2 + self.dec1.bias.norm()**2) / n
        loss =loss1+loss2+ 0.00001 * (self.dec2.weight.norm()**2 + self.dec2.bias.norm()**2) / n
        return loss
    
    
    def pretrain(self):
        string_template = '--------Start pretraining-{}--------'
        print(string_template.format(1))
        pre1 = PretrainDoubleLayer(self.X, self.layers[1], self.device, self.act, lr=self.lr)
        Z = pre1.run() #Pretrain
        self.enc1.weight = pre1.enc.weight
        self.enc1.bias = pre1.enc.bias
        self.dec2.weight = pre1.dec.weight
        self.dec2.bias = pre1.dec.bias
        print(string_template.format(2))
        pre2 = PretrainDoubleLayer(Z.detach(), self.layers[2], self.device, self.act, lr=self.lr)
        pre2.run()
        self.enc2.weight = pre2.enc.weight
        self.enc2.bias = pre2.enc.bias
        self.dec1.weight = pre2.dec.weight
        self.dec1.bias = pre2.dec.bias
        
        
    def run(self):
        print("device:", self.device)
        self.to(self.device)
        self.pretrain()
        Z, _ = self(self.X.t()) 
        Z = Z.detach()
        idx = random.sample(list(range(Z.shape[0])), self.c)
        self.g = Z[idx, :]
        self._update_gplus(Z)
        self._update_M(Z)
        print('--------Start training-------------')
        train_loader = torch.utils.data.DataLoader(Dataset(self.X), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
        loss = 0
        for epoch in range(20):
            D = self._update_D(Z, self.gplus) #Distance
            for i, batch in enumerate(train_loader):#enumerate
                x, idx = batch
                optimizer.zero_grad() 
                z, recons_x = self(x)
                d = D[idx, :]
                m = self.M[idx, :]
                loss = self._build_loss(z, x, d, m, recons_x)
                loss.backward()
                optimizer.step() #update network 
            #print("lr：", optimizer.param_groups[0]['lr'])
            scheduler.step()
            Z, _ = self(self.X.t())
            Z = Z.detach()
            self.clustering(Z) #update g,gplus,M (center, imprecise center and credal partition)
            M = np.concatenate((1 - np.sum(self.M, axis=1).reshape(Z.shape[0], 1), self.M), axis=1)
            clus = extractMass(M, self.F, g=self.g, param={'alpha': self.alpha, 'beta': self.beta, 'delta2': self.delta2})
            betp=clus['betp_n'] 
            pre_label=np.argmax(betp, axis=1) #pignistic transformation
            pre_label=pre_label+1
            nmi, ari,acc,ri= cal_clustering_metric(self.labels, pre_label)
            print('epoch-{}, loss={}, NMI={}, ARI={},ACC={},RI={}'.format(epoch, loss.item(), nmi, ari,acc,ri))
        

    def _update_D(self, Z, gplus): #update distance^2 
        n = Z.shape[0]
        D = np.zeros((n, self.f-1))
        ggplus=gplus.cpu() 
        ZZ=Z.cpu()
        for j in range(self.f-1):
            D[:, j] = np.nansum((ZZ - np.tile(ggplus[j, :], (n, 1))) ** 2, axis=1)
        return D
    
    def _update_gplus(self, Z):
        d = Z.shape[1]
        gplus = np.zeros((self.f-1,d))
        for i in range(1, self.f): 
            fi = self.F[i, :] #F 2^c*c, fi 1*c 
            truc = np.tile(fi, (d, 1)).T #c*d 
            g=self.g.cpu() 
            g=g.numpy()
            #g=self.g.numpy() #tensor 转numpy c*d
            gplus[i-1, :] = np.sum(g * truc, axis=0) / np.sum(fi) 
        gplus = torch.from_numpy(gplus)
        self.gplus = gplus # 2^c-1 * d
        
    def _update_M(self, Z):
        n = Z.shape[0]
        M = np.zeros((n, self.f-1))
        D = self._update_D(Z, self.gplus)
        for i in range(n):
            vect0 = D[i, :]
            for j in range(self.f-1):
                vect1 = (np.tile(D[i, j], self.f-1) / vect0) ** (1 / (self.beta-1))
                vect2 = np.tile(self.card[j] **(self.alpha/ (self.beta-1)), self.f-1)/(self.card **(self.alpha /(self.beta-1)))
                vect3 = vect1 * vect2
                M[i, j] = 1 / (np.sum(vect3) +(self.card[j] ** self.alpha * D[i, j]/ self.delta2) ** (1 / (self.beta-1)))
                if np.isnan(M[i, j]):
                    M[i, j] = 1  
        self.M = M
    
    def _updata_V(self,Z): # Calculation of centers
        A = np.zeros((self.c, self.c)) 
        for k in range(self.c):
            for l in range(self.c):
                truc = np.zeros(self.c)
                truc[[k, l]] = 1
                t = np.tile(truc, (self.f, 1))
                indices = np.where(np.sum((self.F - t) - np.abs(self.F - t), axis=1) == 0)[0]   
                indices = indices - 1
                if len(indices) == 0:
                    A[l, k] = 0
                else:
                    for jj in range(len(indices)):
                        j = indices[jj]
                        mj = self.M[:, j] ** self.beta
                        A[l, k] += np.sum(mj) * self.card[j] ** (self.alpha - 2)
        n = Z.shape[0]
        d = Z.shape[1]
        ZZ = Z.cpu()      
        ZZ = ZZ.numpy()
        B = np.zeros((self.c, d))
        for l in range(self.c):
            truc = np.zeros(self.c)
            truc[l] = 1
            t = np.tile(truc, (self.f, 1))
            indices = np.where(np.sum((self.F - t) - np.abs(self.F - t), axis=1) == 0)[0]   # indices of all Aj including wl
            indices = indices - 1
            mi = np.tile(self.card[indices] ** (self.alpha - 1), (n, 1)) * self.M[:, indices] ** self.beta
            s = np.sum(mi, axis=1)
            mats = np.tile(s.reshape(n, 1), (1, d))
            xim = ZZ * mats
            B[l, :] = np.sum(xim, axis=0)
        g = np.linalg.solve(A, B)
        self.g = torch.from_numpy(g)
    
        
    def clustering(self, Z):
        self._updata_V(Z)
        self._update_gplus(Z)
        self._update_M(Z)
              
            
# Data: d*n (dimension * number of instances)            

data=  np.load('yale_hog.npy') #165, 288
data = data.T
data = torch.from_numpy(data).float()  
labels=np.zeros(165)
for i in range(165):
    labels[i]= int(i/11)
labels +=1 
        

start_time = time.time()
dnecm = DeepEvidentialCMeans(data, labels, [data.shape[0], 128, 80], lam=0.01, type='pairs',alpha=1, beta=1.1, delta2=9, batch_size=128, lr=10**-4)
dnecm.run()
end_time = time.time()
print("Time: {:.2f} s".format(end_time - start_time))        