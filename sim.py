import requests
import progressbar
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from SEIR import SEIR

class Value:
    def __init__(self, alpha, beta, eta, gamma, error):
        self.alpha = alpha 
        self.beta = beta 
        self.eta = eta 
        self.gamma = gamma 
        self.error = error
    
    def print(self):
        print("Alpha: " + str(self.alpha))
        print("Beta: " + str(self.beta))
        print("Eta: " + str(self.eta))
        print("Gamma: " + str(self.gamma))
        print("Error: " + str(self.error))

def getData(date1, date2):
    response = requests.get('https://api.rootnet.in/covid19-in/stats/history')
    try:
        if (response.status_code == 200):
            covid_history = response.json()['data']
        else:
            print("Connection Issue!")
    except:
        print("Connection Issue!")
    columns = ['day', 'total', 'confirmedCasesIndian', 'confirmedCasesForeign', 'confirmedButLocationUnidentified', 'discharged', 'deaths']
    data = pd.DataFrame([[d.get('day'), 
                                      d['summary'].get('total'), 
                                      d['summary'].get('confirmedCasesIndian'), 
                                      d['summary'].get('confirmedCasesForeign'),
                                      d['summary'].get('confirmedButLocationUnidentified'),
                                      d['summary'].get('discharged'), 
                                      d['summary'].get('deaths')] 
                                     for d in covid_history],
                        columns=columns)
    data = data.sort_values(by='day')
    data = data.loc[(data['day'] >= date1) & (data['day'] <= date2)]
    data['infected'] = data['total'] - data['discharged'] - data['deaths']
    return data 

def getSEIRModel(I0, R0, D0, eta=1/5.2, gamma=1/2.9, alpha=0.006, beta=.75, wedge = 0, mu = 0, theta = 0.01):
    N0 = 1380000000
    E0 = 1000
    I0 = I0/10
    R0 = R0 
    D0 = D0
    S0 = N0 - (E0 + I0 + R0 + D0)
    eta = eta
    gamma = gamma
    alpha = alpha
    beta = beta
    wedge = wedge
    mu = mu
    theta = theta
    return SEIR(wedge, mu, alpha, beta, eta, gamma, theta, N0, S0, E0, I0, R0)

def getError(I1, I2):
    return np.sum(np.sqrt(np.abs((np.square(np.asarray(I1)) - np.square(np.asarray(I2))))))

def parameterSpaceExploration(I_real, dt=.01, days=52, gran=5):
    all_values = []
    gran = gran
    alpha = [0.030, 0.04]
    beta = [0.36, 0.40]
    eta = [.043, .048]
    gamma = [.055, .075]
    alpha_explore   = np.linspace(alpha[0], alpha[1], num=gran)
    beta_explore    = np.linspace(beta[0], beta[1], num=gran)
    eta_explore     = np.linspace(eta[0], eta[1], num=gran)
    gamma_explore   = np.linspace(gamma[0], gamma[1], num=gran)
    bar             = progressbar.ProgressBar(maxval=(int(days/dt) + 1)*gran*gran*gran*gran, \
    widgets         = [progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    counter1 = 0
    print("Beginning Parameter Space Exploration...")
    bar.start()
    for alphaI in alpha_explore:
        counter2 = 0
        counter1 += 1
        for betaI in beta_explore:
            counter3 = 0
            counter2 += 1
            for etaI in eta_explore:
                counter4 = 0
                counter3 += 1
                for gammaI in gamma_explore:
                    counter4 += 1
                    model = getSEIRModel(I0, R0, D0, eta=etaI, gamma=gammaI, alpha=alphaI, beta=betaI)
                    iterator = 0
                    I = []
                    I_total = []
                    while iterator*dt < days:
                        if iterator*dt - int(iterator*dt) < dt/2:
                            T.append(iterator*dt)
                            I.append(model.I)
                            if iterator == 0:
                                I_total.append(I0 + I[-1])
                            else:
                                I_total.append(I_total[-1] + I[-1])
                        model.simulate(dt)
                        iterator = iterator + 1
                        bar.update(iterator*counter1*counter2*counter3*counter4 + 1)
                    error = getError(I_total, I_real)
                    all_values.append(Value(alphaI, betaI, etaI, gammaI, error))
    bar.finish()
    return sorted(all_values, key=lambda elem: elem.error)
    
def simAndPlot(I0, R0, D0, alpha, beta, eta, gamma, I_real, dt=0.01, days=52):
    model = getSEIRModel(I0, R0, D0, eta=eta, gamma=gamma, alpha=alpha, beta=beta)
    iterator = 0
    I = []
    I_total = []
    while iterator*dt < days:
        if iterator*dt - int(iterator*dt) < dt/2:
            T.append(iterator*dt)
            I.append(model.I)
            if iterator == 0:
                I_total.append(I0 + I[-1])
            else:
                I_total.append(I_total[-1] + I[-1])
        model.simulate(dt)
        iterator = iterator + 1
    plt.plot([d for d in range(len(I_real))], I_real)
    plt.plot([d for d in range(len(I_total))], I_total)
    plt.xlabel("Time (Days)")
    plt.ylabel("Population")
    plt.title("Infections")
    plt.legend(['Real Data', 'Simulation Data'])
    plt.grid()
    plt.show()

def getReproductionRate(beta, eta, gamma, alpha):
        return (beta * eta) / ((eta + 0.01) * (gamma + alpha))

if __name__ == "__main__":
    date1 = '2021-03-01'
    date2 = '2021-04-21'
    data = getData(date1, date2)
    data = data.drop(columns=['total', 'confirmedCasesIndian', 'confirmedCasesForeign', 'confirmedButLocationUnidentified'])
    I0 = min(data['infected'])
    D0 = min(data['deaths'])
    R0 = min(data['discharged'])
    model = getSEIRModel(I0, R0, D0)
    iterator = 0
    I = []
    I_total = []
    T = []
    I_real = data['infected']
    # all_values = parameterSpaceExploration(I_real)
    # best_value = all_values[0]
    # best_value.print()
    eta = .1
    gamma = .15
    beta = .435
    alpha = .06
    print(getReproductionRate(beta, eta, gamma, alpha))
    simAndPlot(I0, R0, D0, alpha, beta, eta, gamma, I_real)