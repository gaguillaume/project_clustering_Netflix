import numpy as np
import naive_em
import em
import common

X = np.loadtxt("datas/test_incomplete.txt")
X_gold = np.loadtxt("datas/test_complete.txt")

K = 4
n, d = X.shape
seed = 0

init_mixture, init_post = common.init(X, K, seed)
print(naive_em.fill_matrix(X, init_mixture))

EM_mixture, EM_post, EM_logvrais = naive_em.run(X, init_mixture, init_post)
print(EM_logvrais) # --> -1138.890899687267

