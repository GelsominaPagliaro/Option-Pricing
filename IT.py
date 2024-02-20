import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy import special
from scipy.optimize import minimize
from scipy.stats import norm
from datetime import datetime, timedelta
import math
from math import exp, pi
import streamlit as st

#function for time series plot
def plot_ts(data, labels):
      x_axis, y_axis, title = labels
      plt.plot(data, linestyle = "--", color = "blue")
      plt.xlabel(x_axis)
      plt.ylabel(y_axis)
      plt.title(title)
      plt.xticks(rotation=45)
      st.pyplot()

#function for the log likelihood of a geometric brownian motion process
def MoM_gBM(df):
   Delta = 1/252                                                       # Conversion factor for annualization
   alpha_hat = np.mean(df.values, axis=0) / Delta                      # Estimated drift
   sigma_hat = np.sqrt(np.var(df.values, axis=0) / Delta)              # Estimated volatility
   mu_hat = alpha_hat - 0.5 * sigma_hat**2                             # Estimated mean return
   initial_params_gBm = np.array([mu_hat, sigma_hat])

   return initial_params_gBm

def neg_log_likelihood_gBm(params, data):                               #function to compute the gBm negative likelihood 
        mu_hat, sigma_hat = params                                      #with the MoM paramteres
        log_likelihood = -np.sum(norm.logpdf(data, loc = mu_hat, scale = sigma_hat))
        return log_likelihood


#functions for: variance gamma PDF, variance gamma parameters with MoM, variance gamma log likelihood
def pdf_one_point(x=0.0, c=0.0, sigma=1.0, theta=0.0, nu=1.0):
    ''' VarGamma probability density function in a point x '''
    temp1 = 2.0 / (sigma * (2.0 * pi) ** 0.5 * nu ** (1 / nu) * special.gamma(1 / nu))
    temp2 = ((2 * sigma ** 2 / nu + theta ** 2) ** 0.5) ** (0.5 - 1 / nu)
    temp3 = exp(theta * (x - c) / sigma ** 2) * abs(x - c) ** (1 / nu - 0.5)
    temp4 = special.kv(1 / nu - 0.5, abs(x - c) * (2 * sigma ** 2 / nu + theta ** 2) ** 0.5 / sigma ** 2)
    return temp1 * temp2 * temp3 * temp4

def pdf(x=0.0, c=0.0, sigma=1.0, theta=0.0, nu=1.0):
    ''' VarGamma probability density function of an array or a point x '''
    if isinstance(x, (int, float, np.double)):  # works with lists, arrays, and numpy.float64
        return pdf_one_point(x, c, sigma, theta, nu)
    else:
        return [pdf_one_point(xi, c, sigma, theta, nu) for xi in x]

def fit_moments(data):
    ''' fits the parameters of VarGamma distribution to a given list of points
        via method of moments, assumes that theta is small (so theta^2 = 0)
        see: Seneta, E. (2004). Fitting the variance-gamma model to financial data. '''
    mu = np.mean(data)
    sigma_squared = np.mean((data - mu) ** 2)
    beta = np.mean((data - mu) ** 3) / np.mean((data - mu) ** 2) ** 1.5
    kapa = np.mean((data - mu) ** 4) / np.mean((data - mu) ** 2) ** 2

    # solve combined equations
    sigma = sigma_squared ** 0.5
    nu = kapa / 3.0 - 1.0
    theta = sigma * beta / (3.0 * nu)
    c = mu - theta
    return (c, sigma, theta, nu)

def neg_log_likelihood_VG(data, par):
    ''' negative log likelihood function for VarGamma distribution '''
    # par = array([c, sigma, theta, nu])
    if (par[1] > 0) and (par[3] > 0):
        return -sum(math.log(pdf_val) for pdf_val in pdf(data, c=par[0], sigma=par[1], theta=par[2], nu=par[3]) if pdf_val > 0)
    else:
        return float('inf')

#function for montecarlo simulation out of a gBm
def montecarlo_gBm(p, S0, sigma, r):
      M, t, T  = p
      dt = T / t
      matrix_gBm = pd.DataFrame(np.zeros((M, t + 1)))
      matrix_gBm.iloc[:, 0] = S0
      u = np.random.standard_normal(matrix_gBm.shape)
      for i in range(M):
        for j in range(1, t + 1):
          matrix_gBm.iloc[i,j] = matrix_gBm.iloc[i, j - 1]* math.exp((r - ((sigma ** 2)/2)) * (dt) + 
                                                                     sigma * (math.sqrt(dt)) * u[i, j])

      return matrix_gBm

#function for exponential VG
def VGexp(params, T, t, r, S0):
    sigma, theta, nu = params
    a = 1 / nu
    b = 1 / nu
    h = T / t
    t = np.linspace(0, T, t + 1)
    X = np.zeros(len(t))
    I = np.zeros(len(t) - 1)
    X[0] = S0

    for i in range(len(t) - 1):
        I[i] = np.random.gamma(a * h, scale=b)
        X[i + 1] = X[i] * np.exp(r * t[i] + theta * I[i] + 
                                 sigma * math.sqrt(I[i]) * np.random.standard_normal())

    return X

#function for montecarlo simulation out of a VG
def montecarlo_VG(M, t, params, T, r, S0):
    sigma, nu, theta = params
    matrix_VG = np.zeros((M, t+1))

    for i in range(M):
        matrix_VG[i, :] = VGexp(params, T, t, r, S0)

    return matrix_VG

#run and display the program by a streamlit app
st.set_option('deprecation.showPyplotGlobalUse', False) 
st.title("Option pricing")

stocks = st.selectbox("Choose a stock: ", ["AAPL", "GME", "AMZN"])                             #the user selects the stock that will be priced
default_date_string = "2005/01/01"
default_date_format = "%Y/%m/%d"

default_date = datetime.strptime(default_date_string, default_date_format).date()

start_date = st.date_input("Select a starting date:", key = "start", value = default_date, 
              min_value = default_date, max_value= datetime.now().date() - timedelta(days=1))

if start_date != default_date:
   st.success("Successfully updated!")

end_date = st.date_input("Select an ending date:", key = "end", value = start_date, 
              min_value = start_date, max_value= datetime.now().date() - timedelta(days=1))

if end_date != start_date:
   st.success("Successfully updated!")

M = st.slider("How many simulation do you want to perform?", 100, 1000, 100)                    #number of simulations: rows of the simulation matrix
exp_date = st.selectbox("When will the option expire? ", ["3 months", "6 months", "1 year"])    #number of grid points: columns of the simulation matrix
call_put = st.radio("Is a call or put option?", ["Call", "Put"])                                #according to the user choice the price will change
df = None                                                                                       #df is fullfiled when the program starts using all the user'a choices made so far

if st.button("Download and Plot"):                                                              #pressing the button the program starts
    data =  yf.download(stocks, start = start_date, end = end_date)
    data = data.iloc[:, 4]
    time_series_p  = f"{stocks} prices"                                                         #plot the untreated time series
    labels_un = ("Time", time_series_p, "Time series untreated")
    df = np.log(data).diff().dropna()
    time_series_l  = f"{stocks} log"                                                            #plot the treated time series
    labels_t = ("Time", time_series_l, "Time series treated")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Time series untreated")
        plot_un = plot_ts(data, labels_un)

    with col2:
        st.subheader("Time series treated")
        plot_t = plot_ts(df, labels_t)  
    print(df)

if df is not None:

    initial_params_gBm = MoM_gBM(df)                                                            #gBm initial parameters estimated with the method of moments
    result_gBm = minimize(neg_log_likelihood_gBm, initial_params_gBm, args=(df), method="BFGS") #minimizing the negative maximum likelihood
    ll_params_gBm =  result_gBm.x                                                               #MLE estimators

    if result_gBm.success == False:                                                             #try another non-linear optimization method if the previous does not converge
        result_gBm = minimize(neg_log_likelihood_gBm, initial_params_gBm, args=(df), method="Nelder-Mead") 

    gBm_AIC = -2 * result_gBm.fun + 2 * 2                                                       #Akaike criteria for gBm                                            

    initial_params_VG = fit_moments(df)                                                         #VG initial parameters estimated with the method of moments
    result_VG = minimize(neg_log_likelihood_VG, initial_params_VG, args= (np.array(df)), method="Nelder-Mead") #minimizing the negative maximum likelihood
    ll_params_VG = result_VG.x                                                                  #MLE estimators

    VG_AIC = -2 * result_VG.fun + 2 * 2                                                         #Akaike criteria for VG 

    st.write("The akaike criteria for the Geometric Brownian motion process is: ", gBm_AIC)
    st.write("The akaike criteria for the Variance Gamma process is: :", VG_AIC)
    
    distributional_assumptions = ("Variance Gamma", "Geometric Brownina Motion")                #the lower is the Akaike result the better the distribution approximates the data
    if VG_AIC < gBm_AIC:
      st.write("By comparing the Akaike criteria for both processes, the one that better approximates the data is:")
      st.markdown(f"**{distributional_assumptions[0]}**")
      st.write("So the simulations are going to be performed out of the VG process")
    else:
      st.write("By comparing the Akaike criteria for both processes, the one that better approximates the data is:")
      st.markdown(f"**{distributional_assumptions[1]}**")
      st.write("So the simulations are going to be performed out of the VGexp function")
    S0 = data[-1]                                                                               #stock starting value
    K = S0
    print(K)                                                                                    #strike price

    exp_T = (1/4, 1/2, 1)
    grid_points = (10, 100, 1000)
    T = None                                                                                    #expire date
    t = None                                                                                    #number of grid points
    for i in range(3):
      if exp_date[i] == exp_date[i]:
        T = exp_T[i]
        t = grid_points[i]
        break
      
    #risk-free rate from ^TNX
    x = None

    for i in range(4):
      start_date = datetime.now().date() - timedelta(days = i)
      x = yf.download("^TNX", start = start_date)
      if x is not None and not x.empty:
          break

    r = (x.iloc[0, 4])/100   
    print(r)                                                                                   #risk-free rate
    

    #simulation martrix by Montecarlo method for the best distributional assumption:
    if gBm_AIC < VG_AIC:
      sigma_gBm = ll_params_gBm[1]
      params_gBm = (M, t, T)
      matrix = montecarlo_gBm(params_gBm, S0, sigma_gBm, r)                             #montecarlo simualations for gBm if it is the best distributional assumption
                                                                                        #gBm already risk-neutralized, mu is plugged by r
      matrix = pd.DataFrame(matrix)
      S = np.mean(matrix.iloc[:, -1])                                                   #the last column corresponds to the expiry date, 
                                                                                        #assuming it is an european option is the only taken into account
    else:
      params_VG = initial_params_VG[1:4]
      VGresult = VGexp(params_VG, T, t, r, S0)
      matrix  = montecarlo_VG(M, t, params_VG, T, r, S0)                                #montecarlo simualations for VG if it is the best distributional assumption
                                                                                        
      matrix = pd.DataFrame(matrix)                                                     #risk-neutralization
      st.dataframe(matrix)
      final_prices = matrix.iloc[:, -1]
                                                                                        #mean correcting martingale
      matrix_rn = (S0 * final_prices * math.exp(r * T)) / (np.mean(final_prices))
      S = np.mean(matrix_rn)
      print(S)

    #payoff based on the kind of option selected by the user
    if call_put == "Call":
      price = math.exp(-r * T) * np.maximum(S - K, 0)
    else:
      price = math.exp(-r * T) * np.maximum(K - S, 0)

    st.write("The option price is:", price)

    if price == 0.0:
        st.write("The option should not be exercitate")
    else:
        st.write("The option is worth    to be exercitate")
    
    
    












