import scipy.optimize as optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_ytms(bond_table, dates):
    # return a ytm dataframe
    ytm_data = pd.DataFrame(columns=dates, index=bond_table["Bond"]) 
    for i in range(len(dates)):
        for j in range(len(bond_table)):
            maturity_date = bond_table["Maturity Date"][j] 
            price = float(bond_table[dates[i]][j])
            coupon_rate = float(bond_table["Coupon"][j])
            payments = 1 + (pd.to_datetime(maturity_date) - pd.to_datetime(dates[i])).days//182
            ytm = calculate_bond_ytm(price, par, payments, coupon_rate)
            ytm_data.iloc[j, i] = ytm
    return ytm_data

def calculate_bond_ytm(price, par, payments, coup, freq=2, guess=0.05):
    # payments: The number of coupon payments remaining
    coupon = coup * par / float(freq)
    dt = [(i + 1) / freq for i in range(int(payments * freq))]
    ytm_func = lambda y: sum([coupon/((1 + y/freq)**(freq * t)) for t in dt]) + par/(1 + y/freq) ** (freq * payments) - price        
    return optimize.newton(ytm_func, guess) * 100

def get_spot_rates(bond_table, dates):
    # return a spot rate dataframe
    spot_data = pd.DataFrame(columns=dates, index=bond_table["Bond"])
    zero_coupon_bonds = []
    for i in range(len(dates)):
        spot_rate_zcb = get_zero_coupon_rate(bond_table[dates[i]][0], dates[i], bond_table["Maturity Date"][i])
        zero_coupon_bonds.append(spot_rate_zcb)
    spot_data.iloc[0,:] = zero_coupon_bonds
    for i in range(len(dates)):
        for j in range(len(bond_table)):
            if j != 0:
                spot_r = get_spot_rate(bond_table, spot_data, i, j, dates)
                spot_data[dates[i]][j] = abs(spot_r)
    return spot_data

def get_zero_coupon_rate(price, curr_date, maturity_date):
    time_to_maturity = (pd.to_datetime(maturity_date) - pd.to_datetime(curr_date)).days / 365
    rate = -np.log(price / par) / time_to_maturity
    return rate * 100

def get_spot_rate(bond_table, spot_data, date_index, bond_index, dates):
    curr_date = dates[date_index] 
    days_to_maturity = (pd.to_datetime(bond_table["Maturity Date"][bond_index]) - pd.to_datetime(curr_date)).days % 182
    clean_price = float(bond_table[curr_date][bond_index])
    coupon_rate = float(bond_table["Coupon"][bond_index])
    accrued_interest = days_to_maturity / 365 * coupon_rate * clean_price
    dirty_price =  accrued_interest + clean_price
    if bond_index == 1:
        t1 = bond_table["Times to Maturity"][0]
        t2 = bond_table["Times to Maturity"][1]
    else:
        maturity_date = bond_table["Maturity Date"]
        first_period = (pd.to_datetime(maturity_date[bond_index - 1]) - pd.to_datetime(maturity_date[bond_index - 2])).days
        second_period = (pd.to_datetime(maturity_date[bond_index]) - pd.to_datetime(maturity_date[bond_index - 2])).days
        t1 = first_period / 365
        t2 = second_period / 365
    c_t1 = float(bond_table["Coupon"][bond_index - 1] * 100) / 2
    r_t1 = np.exp((spot_data.iloc[bond_index - 1, date_index]) * t1)
    c_t2 = 100 + float(bond_table["Coupon"][bond_index] * 100) / 2
    r_t2 = np.log(c_t2 / (dirty_price - (c_t1 / r_t1))) / t2    
    return r_t2 * 100

def get_forward_rates(spot_data, dates):
    # return a forward rate dataframe
    forward_matrix = pd.DataFrame(columns=dates, index=["1yr-1yr","1yr-2yr", "1yr-3yr", "1yr-4yr"]) 
    for i in range(len(dates)):
        base_spot_rate = spot_data.iloc[0, i] / 100
        for j in range(0, 4):
            numerator = (1 + spot_data.iloc[(j + 1) * 2, i] / 100) ** (j + 2)
            denominator = (1 + base_spot_rate) 
            forward = (numerator / denominator) ** (1 / (j + 1)) - 1
            forward_matrix.iloc[j, i] = forward
    return forward_matrix

def plot_ytm(bonds, ytm_data, dates):     
    plt.figure(figsize=(12, 6), dpi= 100)
    fig = plt.subplot(1, 1, 1)
    for i in range(len(dates)):
        fig.plot(bonds["Times to Maturity"], ytm_data.iloc[:, i], label=dates[i])
    plt.xlim(0, 5)
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Yield to Maturity (%)")
    plt.title("Yield to Maturity curve")
    fig.legend(ytm_data.columns, loc='upper left', ncol=2)
    plt.grid(True, axis = 'both')
    plt.show()
    
def plot_spot(bonds, spot_data, dates):       
    plt.figure(figsize=(12, 6), dpi= 100)
    fig = plt.subplot(1, 1, 1)
    for i in range(len(dates)):
        fig.plot(bonds["Times to Maturity"], spot_data.iloc[:, i], label=dates[i])
    plt.xlim(0, 5)
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Spot rate (%)")
    plt.title("Spot rate curve")
    fig.legend(spot_data.columns, loc='upper left', ncol=2)
    plt.grid(True, axis = 'both')
    plt.show()

def plot_forward(bonds, forward_data, dates):
    #plot of forward rate       
    plt.figure(figsize=(12, 6), dpi= 100)
    fig = plt.subplot(1, 1, 1)
    for i in range(len(dates)):
        fig.plot(forward_data.iloc[:, i])
    plt.xlabel("Time to Maturity (Years)")
    plt.ylabel("Forward rate (%)")
    plt.title("Forward rate curve")
    fig.legend(forward_data.columns, loc='upper left', ncol=2)
    plt.grid(True, axis = 'both')
    plt.show()

def yield_cov(ytm_data):   
    cov_mat = np.zeros([9, 5])
    for i in range(0, 5):
        for j in range(1, 10):
            X_ij = np.log((ytm_data.iloc[i * 2, j]) / (ytm_data.iloc[i * 2, j - 1]))
            cov_mat[j - 1, i] = X_ij
    ytm_cov = np.cov(cov_mat.T)
    eig_val, eig_vec = np.linalg.eig(ytm_cov)
    print(ytm_cov)
    print((eig_val, eig_vec))
    print(eig_val[0] / sum(eig_val) * 100)

def forward_cov(forward_data):
    cov_mat = np.zeros([9, 4])
    for i in range(0, 4):
        for j in range(1, 10):
            X_ij = np.log((forward_data.iloc[i, j]) / (forward_data.iloc[i, j - 1]))
            cov_mat[j - 1, i] = X_ij
    forward_cov = np.cov(cov_mat.T)
    eig_val, eig_vec = np.linalg.eig(forward_cov)
    print(forward_cov)
    print((eig_val, eig_vec))
    print(eig_val[0] / sum(eig_val) * 100)

par = 100

if __name__ == '__main__':
    bond_table = pd.read_excel('data.xlsx') 
    dates = list(bond_table.columns.values)[5:]
    bond_table["Times to Maturity"] = [round((pd.to_datetime(bond_table["Maturity Date"][i]) - pd.to_datetime('2/1/2021')).days / 365, 3) for i in range(len(dates))]
    
    ytm_data = get_ytms(bond_table, dates)    
    plot_ytm(bond_table, ytm_data, dates)
    
    spot_data = get_spot_rates(bond_table, dates)
    plot_spot(bond_table, spot_data, dates)
    
    forward_data = get_forward_rates(spot_data, dates)
    plot_forward(bond_table, forward_data, dates)
    
    yield_cov(ytm_data)
    forward_cov(forward_data)