from flask import Flask, render_template, request, redirect
import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
from numpy import pi, linalg

import json

from sympy import solve, Eq, Function, symbols, sympify
import sympy as sp
import math

t = sp.symbols('t')

##### Se identifica el tipo de movimiento  #####
def ecuacionMovimiento(alpha,omega,Xen = 0,imprimeRaices=False):
    discr = alpha**2 - omega**2
    y = sympify("0")
    if discr == 0:
        if imprimeRaices:
            print("Movimiento criticamente amortiguado. alpha = %.4f ; omega = %.4f" % (alpha,omega))
            print("Raices s_1,2 = %0.4f"%(-alpha))
            
        y = sp.sympify("Xen + (C1*t+C2)*exp(-alpha*t)")
        y = y.subs("alpha",round(float(alpha) , 3 ))
        y = y.subs("Xen",round(float(Xen) , 3 ))
        
    elif discr > 0:
        
        s1 = -alpha + math.sqrt(discr)
        s2 = -alpha - math.sqrt(discr)
            
        if imprimeRaices:
            print("Movimiento sobreamortiguado. alpha = %.4f ; omega = %.4f" % (alpha,omega))
            print("Raices s_1 = %0.4f ; s2 = %0.4f"%(s1,s2))
        y = sp.sympify("Xen + C1*exp(s1*t)+C2*exp(s2*t)")
        y = y.subs("s1",round(float(s1) , 3 ) )
        y = y.subs("s2",round(float(s2) , 3 ))
        y = y.subs("Xen",round(float(Xen) , 3 ))
        
    else:
        
        s1 = math.sqrt((-1)*discr)
        omega_d = math.sqrt((-1)*discr)
        if imprimeRaices:
            print("Movimiento subamortiguado. alpha = %.4f ; omega = %.4f" % (alpha,omega))
            print("Raices s_1 = -%0.4f+j%0.4f ; s_2 = -%0.4f+j%0.4f"%(alpha,omega_d,alpha,omega_d))
            
        y = sp.sympify("Xen + exp(-alpha*t)*( C1* cos(omega_d*t) + C2* sin(omega_d*t) )")
        y = y.subs("alpha",round(float(alpha) , 3 ))
        y = y.subs("omega_d",round(float(omega_d) , 3 ) )
        y = y.subs("Xen",round(float(Xen) , 3 ))
        
    return y

#####  Se realiza la derivada de la solución  #####
def derivada(y_t):
    t = sp.symbols('t')
    yp_t = y_t.diff(t)
    return yp_t

#####  Se sustituyen las condiciones iniciales  #####
def solucionConstantes(_A, _b):
    mA = np.zeros(4,dtype=np.double)
    mb = np.zeros(2,dtype=np.double)
    for i in range(len(_A)):
        mA[i] = float(_A[i])

    for i in range(len(_b)):
        mb[i] = float(_b[i])
        
    mA = mA.reshape(2,2)
    mb = mb.reshape(2,1)
    return linalg.inv(mA).dot(mb)

def solucionSistema2x2(y,yp,xo,dxo):
    t, C1, C2 = sp.symbols('t, C1, C2')
    f1 = y.subs(t,0)
    f2 = yp.subs(t,0)
    
    Eqns = [sp.sympify(f1 - xo),sp.sympify(f2 - dxo)]
    #####  Se obtiene la solución final  #####
    
    _A,_b = sp.linear_eq_to_matrix(Eqns, [C1, C2])
    Cx = solucionConstantes(_A,_b)
    A1 = round(Cx[0][0],3)
    A2 = round(Cx[1][0],3)
    y = y.subs([(C1,A1),(C2,A2)])
    return y

##### Creación de constantes #####
RLC_PARALELO = 0
RLC_SERIE = 1

def variableGraficacion(R=float(0),L=float(0),C=float(0),Vo=float(0),Io=float(0),RLC_serie=int(1),Xen=float(0)):

    xo = float(0)
    dxo = float(0)
    alpha = float(0)
    omega = float(1/math.sqrt(L*C))
    vble_analisis = str('')

    if RLC_serie == 1 and Xen == 0:
        alpha = float(R/(2*L))
        xo = Io
        dxo = float(-(Vo+R*Io)/L)
        vble_analisis = 'i(t)'

    elif RLC_serie == 1 and Xen > 0:
        alpha = float(R/(2*L))
        xo = Vo
        dxo = float(Io/C)
        vble_analisis = 'v(t)'

    elif RLC_serie == 0 and Xen == 0:
        alpha = float(1/(2*R*C))
        xo = Vo
        dxo = float(-(Vo+R*Io)/(R*C))
        vble_analisis = 'v(t)'

    elif RLC_serie == 0 and Xen > 0:
        alpha = float(1/(2*R*C))
        xo = Io
        dxo = float(Vo/L)
        vble_analisis = 'i(t)'

    else:
        print("Error!!!!")
        
    y_t = ecuacionMovimiento(alpha,omega,Xen)
    
    yp_t = derivada(y_t)
    return solucionSistema2x2(y_t,yp_t,xo,dxo), vble_analisis

def f(time,f_t):
    t = sp.symbols('t')
    y = np.zeros(len(time))
    index = np.arange(len(time))
    for ind in index:
        y[ind] = float(f_t.subs(t,time[ind]))
    return y
#################################################################

def create_plot( xmax = float(10) , R = float(1) , L = float(1) , C = float(0.25) , Vo = float(12) , Io = float(12) , Xen = float(24) , TipoRLC = float(RLC_SERIE) ):
    N = 100
    t = np.linspace(0, xmax, N)
    ## Prueba de ejecución
    ##    variableGraficacion(R=float(0),L=float(0),C=float(0),Vo=float(0),Io=float(0),RLC_serie=int(1),Xen=float(0)):
    f_t, vble_analisis = variableGraficacion(float(R),float(L),float(C),float(Vo),float(Io),int(TipoRLC),float(Xen))
    print("f_t ->",f_t)
    v = f(t,f_t)
    df = pd.DataFrame({'x': t, 'y': v}) # creating a sample dataframe

    data = [
        go.Scatter(
            x=df['x'], # assign x as the dataframe column 'x'
            y=df['y'],
            mode= 'lines+markers',
            name= 'lines+markers',
            line = dict(color='#7b3e99',width=2)
        )
    ]

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON, f_t, vble_analisis

app = Flask(__name__)

@app.route('/', methods = ['POST', 'GET'])
def index():
    
    if request.method == 'POST':
        try:
            mXmax = float(request.form['xmax_text'])
        except:
            mXmax = 10
            
        try:
            mR = float(request.form['R_text'])
        except:
            mR = float(1)
        
        try:
            mL = float(request.form['L_text'])
        except:
            mL = float(1)
            
        try:
            mC = float(request.form['C_text'])
        except:
            mC = float(0.25)
        
        try:
            mVo = float(request.form['Vo_text'])            
        except:
            mVo = float(12)
        
        try:
            mIo = float(request.form['Io_text'])
        except:
            mIo = float(12)
        
        try:
            mXen = float(request.form['Xen_text'])
        except:
            mXen = float(24)
        
        try:
            mRLC_sel = float(request.form['RLC_sel'])
        except:
            mRLC_sel = float(0)
            
        bar, f_t, vbleAnalisis = create_plot( mXmax , mR , mL , mC , mVo , mIo , mXen, mRLC_sel )
        
    else:
        bar, f_t, vbleAnalisis = create_plot()
        
    return render_template('index.html', plot=bar, funcion=f_t, variable = vbleAnalisis)

if __name__ == '__main__':
    app.run()
