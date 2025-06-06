import numpy as np
def sample(names, samples, lam = 11):
    

    for i, n in enumerate(np.random.poisson(lam, samples)):
        for name in np.random.choice(names, n):
            print(name, i + 1)