# functions.py>


# The inputs are two vectors of the same size, where y denotes the sales 
# of a store and y_ denotes the corresponding predictions

def MAPE(y, y_) :

    import numpy as np
    
    y = np.array(y)
    y_ = np.array(y_)
    n = len(y)
    return round((1/n)*np.sum((abs(y-y_)/y)), 5)

def MSE(y, y_) :

    import numpy as np
    
    y = np.array(y)
    y_ = np.array(y_)
    n = len(y)
    return round((1/n)*np.sum(((y-y_))**2), 5)

def MAE(y, y_) :

    import numpy as np
    
    y = np.array(y)
    y_ = np.array(y_)
    n = len(y)
    return round((1/n)*np.sum((abs(y-y_))), 5)

# Function to generate prices using the Heston method given de parameters. 

def Call_Price_Heston(S,K,T,r,kappa,theta,nu,rho,V0,alpha=1,L=1000):
    # P= Price of a call with Maturity T and Strike K using the characteristic function of the 
    #    price and the Carr-Madan formula - No FFT used
    # S= Initial price
    # r= risk free rate
    # kappa,theta,nu,rho = parameters Heston. 
    # (kappa: rate vt to theta;  theta: long average variance; 
    # nu: vol of vol;  rho: correlation)
    # V0= initial vol in Heston model
    # alpha = damping factor (alpha >0) typically take alpha=1
    # L = truncation bound for the integral
    import numpy as np
    import scipy.integrate as integrate
    
    i = complex(0,1)
    b=lambda x:(kappa-i*rho*nu*x)
    gamma=lambda x:(np.sqrt(nu**(2)*(x**2+i*x)+b(x)**2))
    a=lambda x:(b(x)/gamma(x))*np.sinh(T*0.5*gamma(x))
    c=lambda x:(gamma(x)*np.cosh(0.5*T*gamma(x))/np.sinh(0.5*T*gamma(x))+b(x))
    d=lambda x:(kappa*theta*T*b(x)/nu**2)

    f=lambda x:(i*(np.log(S)+r*T)*x+d(x))
    g=lambda x:(np.cosh(T* 0.5*gamma(x))+a(x))**(2*kappa*theta/nu**2)
    h=lambda x:(-(x**2+i*x)*V0/c(x))

    phi=lambda x:(np.exp(f(x))*np.exp(h(x))/g(x)) # Characteristic function
    integrand=lambda x:(np.real((phi(x-i*(alpha+1))/((alpha+i*x)*(alpha+1+i*x)))*np.exp(-i*np.log(K)*x)))
    integral = integrate.quad(integrand,0, L)
    P=(np.exp(-r*T-alpha*np.log(K))/np.pi) * integral[0]
    return P

# BS model Price calculation given the volatility

def C_BS(sigma, r, S, T, K):
    # sigma : volatility
    # r : risk-free interest
    # S : stock price
    # T : time to maturity
    # K : strike price
    
    import numpy as np
    from scipy.stats import norm
    
    t = 0
    d1 = (1/(sigma*np.sqrt(T-t)))*(np.log(S/K)+(r+(sigma**2)/2)*(T-t))
    d2 = d1-sigma*np.sqrt(T-t)
    
    
    return S*norm.cdf(d1) - K*np.exp(-r*(T-t))*norm.cdf(d2)

    # bi-section method searching implied volatility

def imp_vol(C, r, S, T, K, N = 20): 
    # N: iteration number
    # K : strike price
    # T : maturity, in "year"
    # t : present instant
    # r : active risk-free rate
    # S : stock price
    
    import numpy as np
    
    n = N+1
    x = np.zeros(n)
    y = np.ones(n) # index : 0 -> N
#     z = np.zeros(n)
    
    for i in range(1,n): # index : 0 -> N-1
        z = 1/2 * (x[i-1] + y[i-1])
        sig = z/ (1 - z)
        
        if (C > C_BS(sig, r, S, T, K)):
            x[i] = z
            y[i] = y[i-1]
        else:
            x[i] = x[i-1]
            y[i] = z
            
    z = (x[N] + y[N]) / 2
    sigma = z / (1-z)
    return sigma 