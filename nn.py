import numpy as np
from sklearn.metrics import roc_auc_score


def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def roulette(fits):
    f = fits - fits.min()  # We make sure lowest fitness is 0
    mark = np.random.random() * f.sum()  # mark some spot
    total = 0
    # start adding the fitnesses until we cross the mark
    for index, i in enumerate(f):
        total += i
        if total >= mark:
            return index
    raise Exception('This is not supposed to happen')


def forward(i, wts, fn):
    "forward calculation of the net"
    for w in wts:
        i = fn(i @ w)
    return i


def fitness_score(exp, got):
    "Calculate the fitness score "
    return roc_auc_score(exp, got)


def getfitness(wts,  # weights of the layers
               i,  # inputs to test on
               oe,  # outputs expected
               fn):  # activation function
    "Get the fitness of a solution"
    w1, w2, w3 = wts
    o = forward(i, [w1, w2, w3], fn)
    e = fitness_score(oe, o)
    return e


def mutate(x, p_mutate):
    "Mutate a solution"
    mask = (np.random.random(x.shape) < p_mutate).astype(int)
    mutation = (np.random.random(x.shape) * 2 - 1)
    x = (mutation * mask) + x
    return x


def cross(c1, c2, p_cross, p_mutate):
    "Crossover two solutions to create a new one"
    n = []
    for a, b in zip(c1, c2):
        c = a
        if np.random.random() < p_cross:
            mask = (np.random.random(a.shape) < 0.5).astype(int)
            x = (a * mask) + (b * (np.abs(1-mask)))
            c = mutate(x, p_mutate)
        n.append(c)
    return n


def evolve(epochs, inp, out, p_mutate, p_cross, wt_shapes, fn, cutoff=1):
    "Evolve a population until an acceptable solution is found"
    pop = [[np.random.random(i) for i in wt_shapes]
           for _ in range(popsize)]
    log = []
    for _ in range(epochs):
        fitnesses = np.array([getfitness(i, inp, out, fn) for i in pop])
        log.append(fitnesses)
        m = fitnesses.max()
        print(m)
        if m == cutoff:
            break
        npop = [cross(pop[roulette(fitnesses)],
                      pop[roulette(fitnesses)],
                      p_cross, p_mutate) for _ in pop]
        pop = npop
    return pop, fitnesses, log


# Configuration
reps = 100  # how many repetitions of the dataset?
hd = 2  # hidden dim
popsize = 100  # population size
p_cross = 0.8  # probability of crossover
p_mutate = 0.01  # probability of mutation
# The shapes of the weights in the FF network
wts = [(2, hd), (hd, hd), (hd, 1)]


# Dataset
inp = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]]*reps)
out = np.array([0, 1, 1, 0]*reps)

evolve(1000, inp, out, p_mutate, p_cross, wts, sigmoid)
