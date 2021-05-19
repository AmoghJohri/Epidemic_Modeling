import progressbar
import numpy as np 
import matplotlib.pyplot as plt 

class SEIR:
    """ 
    Equations:
        S = S + dt * (wedge - mu * S - beta * S * I/N)
        E = E + dt * (beta * S * I/N - (mu + eta) * E)
        I = I + dt * (eta * E - (gamma + mu + alpha) * I)
        R = R + dt * (gamma * I - mu * R)
        D = - (S + E + I + R)

    Parameters:
        N0      = 10000000 (10 million)
        alpha   = 0.006/day
        beta    = 0.75/day
        gamma   = 0.125/day
        eta     = 0.33/day
        wedge   = mu * N (to balance the birth and natural deaths)
        E0      = 20000
        I0      = 1
        R0      = 0
        S0      = N0 - (S0 + I0 + R0)

    """
    def __init__(self, wedge, mu, alpha, beta, eta, gamma, theta, N0, S0, E0, I0, R0):
        self.wedge = wedge 
        self.mu    = mu 
        self.alpha = alpha 
        self.beta  = beta 
        self.eta   = eta 
        self.theta = theta
        self.gamma = gamma
        self.N0    = N0 
        self.S0    = S0 
        self.E0    = E0 
        self.I0    = I0 
        self.R0    = R0 
        self.N     = N0 
        self.S     = S0 
        self.E     = E0 
        self.I     = I0 
        self.R     = R0
        self.T     = 0
        self.D     = 0

    def reset(self):
        self.N = self.N0 
        self.S = self.S0 
        self.E = self.E0 
        self.I = self.I0 
        self.R = self.R0
        self.T = 0
        self.D = 0
        
    def getReproductionRate(self):
        return (self.beta * self.eta) / ((self.eta + self.mu + self.theta) * (self.gamma + self.alpha + self.mu))

    def simulate(self, dt=0.01):
        self.S = self.S + dt * (self.wedge - (self.theta + self.mu) * self.S   - self.beta   * self.S      * self.I/self.N)
        self.E = self.E + dt * (self.beta  * self.S * self.I/self.N - (self.mu + self.eta    + self.theta) * self.E)
        self.I = self.I + dt * (self.eta   * self.E - (self.gamma   + self.mu  + self.alpha) * self.I)
        self.R = self.R + dt * (self.gamma * self.I + self.theta    * (self.S  + self.E)     - self.mu     * self.R)
        self.D = - (dt * (self.wedge - (self.theta + self.mu) * self.S   - self.beta   * self.S      * self.I/self.N) + \
                    dt * (self.beta  * self.S * self.I/self.N - (self.mu + self.eta    + self.theta) * self.E) + \
                    dt * (self.eta   * self.E - (self.gamma   + self.mu  + self.alpha) * self.I)     + \
                    dt * (self.gamma * self.I + self.theta    * (self.S  + self.E)     - self.mu     * self.R))
        self.T = self.T + dt

if __name__ == "__main__":
    N0       = 10000000
    alpha    = 0.006
    beta     = .375
    gamma    = 0.125
    eta      = 0.33
    I0       = 1
    E0       = 20000
    R0       = 0
    S0       = N0 - (I0 + E0 + R0)
    # ignoring births and deaths
    wedge    = 0
    mu       = 0
    theta    = 0.0001
    dt       = 0.01
    days     = 200
    model    = SEIR(wedge, mu, alpha, beta, eta, gamma, theta, N0, S0, E0, I0, R0)
    iterator = 0
    T        = []
    S        = []
    I        = []
    E        = []
    R        = []
    D        = []
    while iterator*dt < days:
        if iterator*dt - int(iterator*dt) < dt:
            T.append(iterator*dt)
            S.append(model.S)
            I.append(model.I)
            E.append(model.E)
            R.append(model.R)
            D.append(model.D)
        model.simulate(dt)
        iterator = iterator + 1
    plt.plot(T, S)
    plt.plot(T, E)
    plt.plot(T, I)
    plt.plot(T, R)
    plt.legend(['Susceptible', 'Exposed', 'Infected', 'Recovered'])
    plt.xlabel("Time (Days)")
    plt.ylabel("Population")
    plt.title("SEIR Model Analysis\nR0: " + str(model.getReproductionRate())[:4])
    plt.grid()
    plt.show()