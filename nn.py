import numpy as np  # For matrix calculations
from sklearn.metrics import roc_auc_score  # For measuring ROC


def sigmoid(x):  # Activation function used
    return 1 / (np.exp(-x) + 1)


def roulette(fits):
    """
    Given a list of fitnesses, selects an item from that list
    with probability proportional to the fitnesses given.
    """
    f = fits - fits.min()  # We make sure lowest fitness is 0
    mark = np.random.random() * f.sum()  # mark some spot
    tip_over = np.cumsum(f) >= mark
    index = np.argmax(tip_over)
    return index


def forward(i, layers, fn):
    "forward calculation of the net"
    # We feed the output of one layer as input
    # to the next layer
    for w in layers:
        i = fn(i @ w)  # Python 3 has @ as a matrix multiplication op
    return i


def fitness_score(exp, got):
    "Calculate the fitness score "
    # This is a simple calculation.
    # Needs to return a single number
    return roc_auc_score(exp, got)


def getfitness(wts,  # weights of the layers
               i,  # inputs to test on
               oe,  # outputs expected
               fn):  # activation function
    "Get the fitness of a solution"
    w1, w2, w3 = wts  # We know it's a 3 layer network
    o = forward(i, [w1, w2, w3], fn)
    e = fitness_score(oe, o)
    return e


def mutate(x, p_mutate):
    "Mutate a solution"
    # Only some parts of the network mutate
    mask = (np.random.random(x.shape) < p_mutate).astype(int)
    # The actual mutation
    mutation = (np.random.random(x.shape) * 2 - 1)
    # Perform the mutation
    x = (mutation * mask) + x
    return x


def cross(c1, c2, p_cross, p_mutate):
    "Crossover two solutions to create a new one"
    n = []
    for a, b in zip(c1, c2):
        c = a
        if np.random.random() < p_cross:
            # Randomly swap weights between the two
            mask = (np.random.random(a.shape) < 0.5).astype(int)
            x = (a * mask) + (b * (np.abs(1-mask)))
            # Mudate the resultant network
            c = mutate(x, p_mutate)
        n.append(c)
    return n


def evolve(epochs, inp, out, p_mutate, p_cross, wt_shapes, fn, cutoff=1):
    "Evolve a population until an acceptable solution is found"
    # From the shape of weights, we create random weights for the networks
    pop = [[np.random.random(i) for i in wt_shapes]
           for _ in range(popsize)]
    log = []
    for epoch in range(epochs):
        fitnesses = np.array([getfitness(i, inp, out, fn) for i in pop])
        log.append(fitnesses)
        m = fitnesses.max()
        print(epoch, m)
        if m >= cutoff:
            # Our network is perfect.
            break
        # A new population needs to be created
        # We select two parents by roulette selection
        # from the current population and breed them
        # by crossing them
        # We choose to maintain population size
        npop = [cross(pop[roulette(fitnesses)],
                      pop[roulette(fitnesses)],
                      p_cross, p_mutate) for _ in pop]
        pop = npop
    return pop, fitnesses, log


# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
reps = 100  # how many repetitions of the dataset?
hd = 2  # hidden dim
popsize = 100  # population size
p_cross = 0.8  # probability of crossover
p_mutate = 0.01  # probability of mutation
# The shapes of the weights in the FF network
wts = [
       (2, hd),  # input layer of network
       (hd, hd),  # hidden layer of network
       (hd, 1)  # output layer of network
       ]


# Dataset
inp = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]]*reps)
out = np.array([0, 1, 1, 0]*reps)

evolve(1000, inp, out, p_mutate, p_cross, wts, sigmoid)
