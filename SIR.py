import numpy as np
import matplotlib.pyplot as plt

class SIR:
    def __init__(self, a, b, N, initial_infections = 1):
        self.a = a
        self.b = b 
        self.N = N 
        self.initial_infections = initial_infections
        self.S = np.asarray([N-self.initial_infections])
        self.I = np.asarray([self.initial_infections])
        self.R = np.asarray([0])

    def simulate(self, x, h=0.1, fetch=5):
        def f1(x, param):
            S = param[0]
            I = param[1]
            return -1*self.a*S*I 
        def f2(x, param):
            S = param[0]
            I = param[1]
            return self.a*S*I - self.b*I 
        def f3(x, param):
            I = param[0]
            return self.b*I
        f = [f1, f2, f3]
        x0 = 0
        y0 = [self.S, self.I, self.R]
        return SIR.rungeKutta4(x0, y0, f, x, fetch)
        
    @staticmethod
    def rungeKutta4(x0, y0, f, x, fetch, h=0.01):
        iterations = (int)((x - x0)/h)
        K1 = np.asarray([0. for i in range(len(y0))])
        K2 = np.asarray([0. for i in range(len(y0))])
        K3 = np.asarray([0. for i in range(len(y0))])
        K4 = np.asarray([0. for i in range(len(y0))])
        y = np.asarray(y0)
        x = x0
        out = []
        counter = 0
        for each in range(iterations):
            y = y.reshape(-1)
            for i in range(len(y)):
                K1[i] = h*f[i](x, y)
            for i in range(len(y)):
                K2[i] = h*f[i](x + h/2, y + K1/2)
            for i in range(len(y)):
                K3[i] = h*f[i](x + h/2, y + K2/2)
            for i in range(len(y)):
                K4[i] = h*f[i](x + h, y + K3)
            y = y + (1.0/6.0)*(K1 + 2*K2 + 2*K3 + K4)
            x = x + h
            counter += 1
            if counter == fetch:
                out.append(y)
                counter = 0
        return np.asarray(out)

if __name__ == "__main__":
    model1 = SIR(0.01, 0.1, 100)
    out = model1.simulate(30)
    S = out[:,0]
    I = out[:, 1]
    R = out[:, 2]
    plt.plot(S)
    plt.plot(I)
    plt.plot(R)
    plt.legend(["S", "I", "R"])
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title("SIR Model Dyanmics")
    plt.grid()
    plt.show()