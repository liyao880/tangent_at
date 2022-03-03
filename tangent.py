import torch
import numpy as np
from sklearn.decomposition import PCA

def noise_along(encoded, m, i):
    encoded[:,i] = encoded[:,i] + torch.rand(m).cuda()
    return encoded

def compute_tangent_component(PC_dmn, x, x_adv):
    m = x.shape[0]
    x = x.reshape(m,-1)
    x_adv = x_adv.reshape(m,-1)
    comp = []
    for i in range(m):
        A = PC_dmn[:,i,:]
        b = np.matmul(x[i,:]-x_adv[i,:],np.matmul(A.T, np.linalg.pinv(np.matmul(A, A.T)))) # (x-x_adv)*A(A^TA)^-1
        comp.append(np.linalg.norm(b))
    return np.array(comp)


def check_tangent(autoencoder, x, x_adv, k=10):
    encoded, _ = autoencoder(x)
    encoded = encoded.detach()
    m, d = encoded.shape
    
    PC_dmn = []
    for i in range(d):
        X = []
        for j in range(k):
            encoded_new = noise_along(encoded, m, i)
            x_rec = autoencoder.decode(x.shape[0], encoded_new)
            x_rec = x_rec.view(x_rec.shape[0],-1).detach().cpu().numpy()
            X.append(x_rec)
        X = np.stack(X)
        PCs = []
        for q in range(m):
            X_kn = X[:,q,:]
            pc1 = PCA(n_components=1).fit(X_kn).components_
            PCs.append(pc1)
        PCs = np.concatenate(PCs,axis=0)
        PC_dmn.append(PCs)
    PC_dmn = np.stack(PC_dmn)
    components = compute_tangent_component(PC_dmn, x.detach().cpu().numpy(), x_adv.detach().cpu().numpy())
    return components

