import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("datas/toy_data.txt")

K = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]

for k in K:
    KM_best_mixture, KM_best_post, KM_best_cost = None, None, np.inf
    EM_best_mixture, EM_best_post, EM_best_logvrais = None, None, -np.inf
    for seed in seeds:
        init_mixture, init_post = common.init(X, k, seed)
        # Modèle KMeans
        KM_mixture, KM_post, KM_cost = kmeans.run(X, init_mixture, init_post)
        if KM_cost < KM_best_cost:
            KM_best_mixture, KM_best_post, KM_best_cost = KM_mixture, KM_post, KM_cost
        # Modèle EM
        EM_mixture, EM_post, EM_logvrais = naive_em.run(X, init_mixture, init_post)
        if EM_logvrais > EM_best_logvrais:
            EM_best_mixture, EM_best_post, EM_best_logvrais = EM_mixture, EM_post, EM_logvrais
    common.plot(X, KM_best_mixture, KM_best_post, f"K-means K={k}")
    common.plot(X, EM_best_mixture, EM_best_post, f"EM K={k}")
