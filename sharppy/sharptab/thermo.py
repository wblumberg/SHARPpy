''' Thermodynamic Library '''
from __future__ import division
import numpy as np
import numpy.ma as ma
from sharppy.sharptab.utils import *
from sharppy.sharptab.constants import *

__all__ = ['drylift', 'thalvl', 'lcltemp', 'theta', 'wobf']
__all__ += ['satlift', 'wetlift', 'lifted', 'vappres', 'mixratio']
__all__ += ['temp_at_mixrat', 'wetbulb', 'thetaw', 'thetae']
__all__ += ['virtemp', 'relh']
__all__ += ['ftoc', 'ctof', 'ctok', 'ktoc', 'ftok', 'ktof']


# Constants Used
c1 = 0.0498646455 ; c2 = 2.4082965 ; c3 = 7.07475
c4 = 38.9114 ; c5 = 0.0915 ; c6 = 1.2035
eps = 0.62197

# Constants for RDJ2008/Bolton equations
k_d = ROCP
C = ZEROCNK
lambda_factor = 1./k_d ; A = 2675. # Kelvin
a = 17.67 ; b = 243.5 # Kelvin
p_0 = 1000.
k_0 = 3036 # K
k_1 = 1.78 # unitless
k_2 = 0.448 # unitless
k_3 = 0 # unitless
nu = k_d
epsilon = 0.622 # mb

def drylift(p, t, td, method='bolton'):
    '''
    Lifts a parcel to the LCL and returns its new level and temperature.

    Parameters
    ----------
    p : number, numpy array
        Pressure of initial parcel in hPa
    t : number, numpy array
        Temperature of inital parcel in C
    td : number, numpy array
        Dew Point of initial parcel in C
    method : string
        Method of which to perform the dry lift calculation
        wobus - use legacy NSHARP code
        bolton - use Bolton (1980) equations.

    Returns
    -------
    p2 : number, numpy array
        LCL pressure in hPa
    t2 : number, numpy array
        LCL Temperature in C

    '''
    t2 = lcltemp(t, td, method=method)
    p2 = thalvl(theta(p, t, 1000.), t2)
    return p2, t2


def lcltemp(t, td, method='bolton'):
    '''
    Returns the temperature (C) of a parcel when raised to its LCL.

    Parameters
    ----------
    t : number, numpy array
        Temperature of the parcel (C)
    td : number, numpy array
        Dewpoint temperature of the parcel (C)
    method : str
        Select the equation use to get the LCL temperature.
        wobus - use legacy NSHARP code
        bolton - use Bolton (1980) equation.

    Returns
    -------
    Temperature (C) of the parcel at it's LCL.

    '''
    if method == 'wobus':
        s = t - td
        dlt = s * (1.2185 + 0.001278 * t + s * (-0.00219 + 1.173e-5 * s -
            0.0000052 * t))
        return t - dlt
    else:
        num = 1.
        denom = (1./(ctok(td) - 56.)) + (np.log(ctok(t)/ctok(td))/ 800.)
        return ktoc((num/denom) + 56.)


def thalvl(theta, t):
    '''
    Returns the level (hPa) of a parcel.

    Parameters
    ----------
    theta : number, numpy array
        Potential temperature of the parcel (C)
    t : number, numpy array
        Temperature of the parcel (C)

    Returns
    -------
    Pressure Level (hPa [float]) of the parcel
    '''

    t = t + ZEROCNK
    theta = theta + ZEROCNK
    return 1000. / (np.power((theta / t),(1./ROCP)))


def theta(p, t, p2=1000.):
    '''
    Returns the potential temperature (C) of a parcel.

    Parameters
    ----------
    p : number, numpy array
        The pressure of the parcel (hPa)
    t : number, numpy array
        Temperature of the parcel (C)
    p2 : number, numpy array (default 1000.)
        Reference pressure level (hPa)

    Returns
    -------
    Potential temperature (C)

    '''
    return ((t + ZEROCNK) * np.power((p2 / p),ROCP)) - ZEROCNK


def thetaw(p, t, td, method='wobus'):
    '''
    Returns the wetbulb potential temperature (C) of a parcel.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)
    td : number
        Dew point of parcel (C)
    method : str

    Returns
    -------
    Wetbulb potential temperature (C)

    '''
    if method == 'wobus':
        p2, t2 = drylift(p, t, td, method='wobus')
        return wetlift(p2, t2, 1000.)
    else:
        # Computes Theta_W...see Eq. 3.8 from Davies-Jones (2008)
        a_0 = 7.101574
        a_1 = -20.68208
        a_2 = 16.11182
        a_3 = 2.574631
        a_4 = -5.205688
        b_1 = -3.552497
        b_2 = 3.781782
        b_3 = -0.6899655
        b_4 = -0.5929340

        te = ctok(np.atleast_1d(thetae(p, t, td)))  # atleast_1d might be useful for my calculations.
        thaw = np.empty(te.shape)
        idx = np.where(te < 173.15)[0]
        idx2 = np.where(te >= 173.15)[0]

        thaw[idx] = te[idx]
        X = te[idx2] / C
        corr = np.exp((a_0 + (a_1 * X) + (a_2 * np.square(X)) + (a_3 * (X**3)) + (a_4 * (X**4))) / \
                      (1 + (b_1 * X) + (b_2 * np.square(X)) + (b_3 * (X**3)) + (b_4 * (X**4))))
        thaw[idx2] = ktoc(te[idx2]) - corr
        if len(thaw) == 1:
            return thaw[0]
        else:
            return thaw

def thetae(p, t, td, method='bolton'):
    '''
    Returns the equivalent potential temperature (C) of a parcel.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)
    td : number
        Dew point of parcel (C)
    method: str
        The method of which to calculate the Theta-E value
        wobus - uses Wobus method and legacy NSHARP equations.
        bolton - uses the Bolton (1980) equations

    Returns
    -------
    Equivalent potential temperature (C)

    '''
    p2, t2 = drylift(p, t, td, method=method)
    if method == 'wobus':
        return theta(100., wetlift(p2, t2, 100.), 1000.)
    elif method == 'bolton':
        r = _e2r(vappres(td, method='bolton'), p)
        theta_dl = ctok(t) * (np.power(1000. / (p - vappres(t, method='bolton')), k_d) * np.power(ctok(t) / ctok(t2), r * (0.28 * 10 ** -3)))
        term1 = ((k_0 / ctok(t2)) - k_1)
        term2 = (r * (1. + (k_2 * r)))
        thetae = theta_dl * np.exp(term1 * term2)
        return ktoc(thetae)


def virtemp(p, t, td):
    '''
    Returns the virtual temperature (C) of a parcel.  If 
    td is masked, then it returns the temperature passed to the 
    function.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)
    td : number
        Dew point of parcel (C)

    Returns
    -------
    Virtual temperature (C)

    '''
    
    tk = t + ZEROCNK
    w = 0.001 * mixratio(p, td)
    vt = (tk * (1. + w / eps) / (1. + w)) - ZEROCNK
    if not QC(vt):
        return t
    else:
        return vt

def relh(p, t, td):
    '''
    Returns the virtual temperature (C) of a parcel.

    Parameters
    ----------
    p : number
        The pressure of the parcel (hPa)
    t : number
        Temperature of the parcel (C)
    td : number
        Dew point of parcel (C)

    Returns
    -------
    Relative humidity (%) of a parcel

    '''
    return 100. * mixratio(p, td) / mixratio(p, t)


def wobf(t):
    '''
    Implementation of the Wobus Function for computing the moist adiabats.

    Parameters
    ----------
    t : number, numpy array
        Temperature (C)

    Returns
    -------
    Correction to theta (C) for calculation of saturated potential temperature.

    '''
    t = t - 20
    try:
        # If t is a scalar
        if t is np.ma.masked:
            return t
        if t <= 0:
            npol = 1. + t * (-8.841660499999999e-3 + t * ( 1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))))
            npol = 15.13 / (np.power(npol,4))
            return npol
        else:
            ppol = t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))))
            ppol = 1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol))
            ppol = (29.93 / np.power(ppol,4)) + (0.96 * t) - 14.8
            return ppol
    except ValueError:
        # If t is an array
        npol = 1. + t * (-8.841660499999999e-3 + t * ( 1.4714143e-4 + t * (-9.671989000000001e-7 + t * (-3.2607217e-8 + t * (-3.8598073e-10)))))
        npol = 15.13 / (np.power(npol,4))
        ppol = t * (4.9618922e-07 + t * (-6.1059365e-09 + t * (3.9401551e-11 + t * (-1.2588129e-13 + t * (1.6688280e-16)))))
        ppol = 1 + t * (3.6182989e-03 + t * (-1.3603273e-05 + ppol))
        ppol = (29.93 / np.power(ppol,4)) + (0.96 * t) - 14.8
        correction = np.zeros(t.shape, dtype=np.float64)
        correction[t <= 0] = npol[t <= 0]
        correction[t > 0] = ppol[t > 0]
        return correction



def satlift(p, thetam):
    '''
    Returns the temperature (C) of a saturated parcel (thm) when lifted to a
    new pressure level (hPa)

    Parameters
    ----------
    p : number
        Pressure to which parcel is raised (hPa)
    thetam : number
        Saturated Potential Temperature of parcel (C)

    Returns
    -------
    Temperature (C) of saturated parcel at new level

    '''
    try:
        # If p and thetam are scalars
        if np.fabs(p - 1000.) - 0.001 <= 0: 
            return thetam
        eor = 999
        while np.fabs(eor) - 0.1 > 0:
            if eor == 999:                  # First Pass
                pwrp = np.power((p / 1000.),ROCP)
                t1 = (thetam + ZEROCNK) * pwrp - ZEROCNK
                e1 = wobf(t1) - wobf(thetam)
                rate = 1
            else:                           # Successive Passes
                rate = (t2 - t1) / (e2 - e1)
                t1 = t2
                e1 = e2
            t2 = t1 - (e1 * rate)
            e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK
            e2 += wobf(t2) - wobf(e2) - thetam
            eor = e2 * rate
        return t2 - eor
    except ValueError:
        # If p and thetam are arrays
        short = np.fabs(p - 1000.) - 0.001 <= 0
        lft = np.where(short, thetam, 0)
        if np.all(short):
            return lft

        eor = 999
        first_pass = True
        while np.fabs(np.min(eor)) - 0.1 > 0:
            if first_pass:                  # First Pass
                pwrp = np.power((p[~short] / 1000.),ROCP)
                t1 = (thetam[~short] + ZEROCNK) * pwrp - ZEROCNK
                e1 = wobf(t1) - wobf(thetam[~short])
                rate = 1
                first_pass = False
            else:                           # Successive Passes
                rate = (t2 - t1) / (e2 - e1)
                t1 = t2
                e1 = e2
            t2 = t1 - (e1 * rate)
            e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK
            e2 += wobf(t2) - wobf(e2) - thetam[~short]
            eor = e2 * rate
        lft[~short] = t2 - eor
        return lft


def wetlift(p, t, p2, theta_e=-9999, method='wobus'):
    '''
    Lifts a parcel moist adiabatically to its new level.

    Parameters
    -----------
    p : number
        Pressure of initial parcel (hPa)
    t : number
        Temperature of initial parcel (C)
    p2 : number
        Pressure of final level (hPa)
    theta_e : number (optional)
        Equivalent potential temperature of the parcel (C)
    method : method to calculate the pseudoadiabat
        wobus - use old NSHARP method
        bolton - use RDJ 2008 method

    Returns
    -------
    Temperature (C)

    '''
<<<<<<< HEAD
    #if p == p2:
    #    return t
    thta = theta(p, t, 1000.)
    if thta is np.ma.masked or p2 is np.ma.masked:
        return np.ma.masked
    thetam = thta - wobf(thta) + wobf(t)
    return satlift(p2, thetam)


=======
    if method == 'wobus':
        thta = theta(p, t, 1000.)
        if thta is np.ma.masked or p2 is np.ma.masked:
            return np.ma.masked
        thetam = thta - wobf(thta) + wobf(t)
        return satlift(p2, thetam)
    else:
        if np.atleast_1d(theta_e).any() == -9999:
            thetae = thetae(p, t, t, method=method)
        return wetlift_rdj(np.atleast_1d(theta_e), p2) - C 
>>>>>>> rdj_wetlift

def lifted(p, t, td, lev, method='wobus'):
    '''
    Calculate temperature (C) of parcel (defined by p, t, td) lifted
    to the specified pressure level.

    Parameters
    ----------
    p : number
        Pressure of initial parcel in hPa
    t : number
        Temperature of initial parcel in C
    td : number
        Dew Point of initial parcel in C
    lev : number
        Pressure to which parcel is lifted in hPa

    Returns
    -------
    Temperature (C) of lifted parcel

    '''
    p2, t2 = drylift(p, t, td, method=method)
    return wetlift(p2, t2, lev)


def vappres(t, method='bolton'):
    '''
    Returns the vapor pressure of dry air at given temperature

    Parameters
    ------
    t : number, numpy array
        Temperature of the parcel (C)

    Returns
    -------
    Vapor Pressure of dry air (hPa)

    '''
    if method == 'bolton':
        return 6.122 * np.exp((a * t) / (t + b))
    else:
        pol = t * (1.1112018e-17 + (t * -3.0994571e-20))
        pol = t * (2.1874425e-13 + (t * (-1.789232e-15 + pol)))
        pol = t * (4.3884180e-09 + (t * (-2.988388e-11 + pol)))
        pol = t * (7.8736169e-05 + (t * (-6.111796e-07 + pol)))
        pol = 0.99999683 + (t * (-9.082695e-03 + pol))
    return 6.1078 / pol**8

def mixratio(p, t):
    '''
    Returns the mixing ratio (g/kg) of a parcel

    Parameters
    ----------
    p : number, numpy array
        Pressure of the parcel (hPa)
    t : number, numpy array
        Temperature of the parcel (C)

    Returns
    -------
    Mixing Ratio (g/kg) of the given parcel

    '''
    x = 0.02 * (t - 12.5 + (7500. / p))
    wfw = 1. + (0.0000045 * p) + (0.0014 * x * x)
    fwesw = wfw * vappres(t)
    return 621.97 * (fwesw / (p - fwesw))


def temp_at_mixrat(w, p):
    '''
    Returns the temperature (C) of air at the given mixing ratio (g/kg) and
    pressure (hPa)

    Parameters
    ----------
    w : number, numpy array
        Mixing Ratio (g/kg)
    p : number, numpy array
        Pressure (hPa)

    Returns
    -------
    Temperature (C) of air at given mixing ratio and pressure
    '''
    x = np.log10(w * p / (622. + w))
    x = (np.power(10.,((c1 * x) + c2)) - c3 + (c4 * np.power((np.power(10,(c5 * x)) - c6),2))) - ZEROCNK
    return x


def wetbulb(p, t, td, method='wobus'):
    '''
    Calculates the wetbulb temperature (C) for the given parcel

    Parameters
    ----------
    p : number
        Pressure of parcel (hPa)
    t : number
        Temperature of parcel (C)
    td : number
        Dew Point of parcel (C)

    Returns
    -------
    Wetbulb temperature (C)
    '''
    p2, t2 = drylift(p, t, td, method=method)
    return wetlift(p2, t2, p)


def ctof(t):
    '''
    Convert temperature from Celsius to Fahrenheit

    Parameters
    ----------
    t : number, numpy array
        The temperature in Celsius

    Returns
    -------
    Temperature in Fahrenheit (number or numpy array)

    '''
    return (1.8 * t) + 32.


def ftoc(t):
    '''
    Convert temperature from Fahrenheit to Celsius

    Parameters
    ----------
    t : number, numpy array
        The temperature in Fahrenheit

    Returns
    -------
    Temperature in Celsius (number or numpy array)

    '''
    return (t - 32.) * (5. / 9.)


def ktoc(t):
    '''
    Convert temperature from Kelvin to Celsius

    Parameters
    ----------
    t : number, numpy array
        The temperature in Kelvin

    Returns
    -------
    Temperature in Celsius (number or numpy array)

    '''
    return t - ZEROCNK


def ctok(t):
    '''
    Convert temperature from Celsius to Kelvin

    Parameters
    ----------
    t : number, numpy array
        The temperature in Celsius

    Returns
    -------
    Temperature in Kelvin (number or numpy array)

    '''
    return t + ZEROCNK


def ktof(t):
    '''
    Convert temperature from Kelvin to Fahrenheit

    Parameters
    ----------
    t : number, numpy array
        The temperature in Kelvin

    Returns
    -------
    Temperature in Fahrenheit (number or numpy array)

    '''
    return ctof(ktoc(t))


def ftok(t):
    '''
    Convert temperature from Fahrenheit to Kelvin

    Parameters
    ----------
    t : number, numpy array
        The temperature in Fahrenheit

    Returns
    -------
    Temperature in Kelvin (number or numpy array)

    '''
    return ctok(ftoc(t))

# RDJ Wetlift equations
def _es(t):
    '''
    Private wrapper function to compute vapor pressure for Bolton (1980) and RDJ calculations.

    Parameters
    ----------
    t : number, numpy array
        Temperature of the parcel (C)
    Returns
    -------
    Vapor pressure of dry air (hPa)
    '''
    return vappres(t, method='bolton')

def _e2r(e, p):
    '''
    Private function to calculate water vapor mixing ratio for Bolton (1980) and RDJ calculations.

    Parameters
    ----------
    e : number, numpy array
        Vapor pressure of the parcel (hPa)
    p : number, numpy array
        Pressure of the parcel (hPa)

    Returns
    -------
    Water vapor mixing ratio (unitless; kg/kg)
    '''
    return (epsilon * e)/(p-e)

def _rs(tau, p):
    nondim_pres = np.power(p / p_0, 1./lambda_factor)
    es = _es(ktoc(tau))
    return _e2r(es, p) # convert to unitless

def _eqG(tau, rs):
    '''

    '''
    return ((k_0/tau) - k_1) * (rs + k_2 * np.power(rs,2))

def equiv_t(theta, p):
    '''
    Calculates the equivalent temperature by lifting a parcel defined by its potential temperature
    to a given pressure level (p)

    Parameters
    ----------
    theta : number, numpy array
        Potential temperature (K)
    p : number, numpy array
        Pressure (hPa)

    Returns
    -------
    t_e : number, numpy array
        Equivalent temperature of the parcel (K)
    '''
    nondim_pres = np.power(p / p_0, 1./lambda_factor)
    return theta * nondim_pres

def _f(tau, p):
    '''
    Calculates the result of RDJ 2008 Equation 2.3
    Used for updating the first guess of the wetbulb temperature.

    Parameters
    ----------
    tau : number, numpy array
        Temperature (K)
    p : number, numpy array
        Pressure (hPa)

    Returns
    -------
    f : number, numpy array
        Result to Equation 2.3
    '''
    nondim_pres = np.power(p / p_0, 1. / lambda_factor)
    es = _es(ktoc(tau))
    rs = _rs(tau, p)
    G = _eqG(tau, rs)
    f_tau_pi = np.power(C / tau, lambda_factor) * \
               np.power(1 - (es / (p_0 * np.power(nondim_pres, lambda_factor))), lambda_factor * nu) * \
               np.power(nondim_pres, -lambda_factor * k_3 * rs) * \
               np.exp(-lambda_factor * G)
    return f_tau_pi


def _dlnfdt(tau, p):
    '''
    Calculates the result of RDJ 2008 Equation A.1
    Used for updating the first guess of the wetbulb temperature.
    Returns several values for use in the accelerated pseudoadiabat method.

    Parameters
    ----------
    tau : number, numpy array
        Temperature (K)
    p : number, numpy array
        Pressure (hPa)

    Returns
    -------
    dflnf_dt : number, numpy array
        The first derivative of f(tau, p)
    des_dt : number, numpy array
        The first derivative of _es
    drs_dt : number, numpy array
        The first derivative of _rs
    dG_dt: number, numpy array
        The first derivative of the function G
    es: number, numpy array
        The saturation vapor pressure (hPa)
    r_s : number, numpy array
        Water vapor mixing ratio (unitless)
    '''
    es = _es(ktoc(tau))
    r_s = _e2r(es, p)
    des_dt = es * ((a * b) / ((tau - C + b)**2))
    drs_dt = ((epsilon * p) / ((p - es)**2)) * des_dt
    dG_dt = ((-k_0 / (tau**2)) * (r_s + k_2 * (r_s**2))) + \
            (((k_0 / tau) - k_1) * (1. + 2. * k_2 * r_s) * drs_dt)
    dlnf_dt = -lambda_factor * (((1. / tau) + ((nu / (p - es)) * des_dt)) + \
                                (k_3 * k_d * np.log(p / p_0) * drs_dt) + dG_dt)

    return dlnf_dt, des_dt, drs_dt, dG_dt, es, r_s

def _dlnf2dt2(tau, p, dlnf_dt, des_dt, drs_dt, dG_dt, f_tau_pi, es, rs):
    # tau - temp in C
    # p - pressure in p
    # dlnf_dt - dlnf_dt (eq. A.2)
    # des_dt - eq A.5
    # drs_dt - eq A.4
    # dG_dt - eq A.3
    # f_tau_pi = f[tau, pi]
    # es = saturation vapor pressure
    # rs = saturation mixing ratio
    # Returns df2_dt2 or f''[tau,pi] (eq. A.6)
    des2_dt2 = ((a*b)/((tau-C+b)**2)) * (des_dt - ((2*es)/(tau-C+b)))
    drs2_dt2 = ((epsilon * p)/((p-es)**2)) * \
               (des2_dt2 - ((2/(p-es)) * ((des_dt)**2)))
    dG2_dt2 = (((2 * k_0)/(tau**3)) * (rs + (k_2 * (rs**2)))) -\
              (((2 * k_0)/(tau**2)) * (1 + (2 * k_2 * rs)) * drs_dt) +\
              (((k_0/tau) - k_1) * 2 * k_2 * (drs_dt**2)) + \
              ((((k_0/tau) - k_1) * (1 + (2 * k_2 * rs))) * drs2_dt2)
    dlnf2_dt2 = lambda_factor * \
                ((1./(tau**2)) - ((nu/((p-es)**2)) * (des_dt**2)) -\
                ((nu/(p-es)) * des2_dt2) - \
                k_3 * k_d * np.log(p/p_0) * drs2_dt2 - dG2_dt2)

    return dlnf2_dt2

def _guess_Tw(thetae, p):
    '''
    Calculates the first guess temperature along a defined pseudoadiabat for a specific pressure.
    Each pseudoadiabat is defined by a unique equivalent potential temperature value.

    Parameters
    ----------
    thetae : number, numpy array
        Equivalent potential temperature (C)
    p : number, numpy array
        Pressure value the parcel is lifted to along the defined pseudoadiabat (hPa)

    Returns
    -------
    t_w : number, numpy array
        First guess of the wetbulb temperature on the defined pseudoadaibat (C)
    '''
    thetae, p = np.atleast_1d(thetae, p)
    t_e = equiv_t(thetae, p)  # returns in Kelvin
    nondim_pres = np.ma.power(p / p_0, 1. / lambda_factor)
    normalized_t_e = np.ma.power(C / t_e, lambda_factor)  # Unitless
    D = (0.1859 * (p / 1000.) + 0.6512)**-1

    r_s = _e2r(_es(ktoc(t_e)), p)
    deriv = ((a * b) / ((ktoc(t_e) + b)**2))

    t_w = np.empty(t_e.shape)
    k1_pi = -38.5 * (nondim_pres**2) + 137.81 * nondim_pres - 53.737  # RDJ 2007 Eq 4.3
    k2_pi = -4.392 * (nondim_pres**2) + 56.831 * nondim_pres - 0.384  # RDJ 2007 Eq 4.4

    # Criteria 1 (RDJ 2007 Eq 4.8)
    idx = np.where(normalized_t_e > D)  # [0]
    t_w[idx] = ktoc(t_e[idx]) - ((A * r_s[idx]) / (1 + A * r_s[idx] * deriv[idx]))
    # Criteria 2 (RDJ 2007 Eq 4.9)
    idx = np.where((1 <= normalized_t_e) & (normalized_t_e <= D))  # [0]
    t_w[idx] = k1_pi[idx] - k2_pi[idx] * normalized_t_e[idx]
    # Criteria 3 (RDJ 2007 Eq 4.10)
    idx = np.where((0.4 <= normalized_t_e) & (normalized_t_e < 1))  # [0]
    t_w[idx] = (k1_pi[idx] - 1.21) - ((k2_pi[idx] - 1.21) * normalized_t_e[idx])
    # Criteria 4 (RDJ 2007 Eq 4.11)
    idx = np.where(normalized_t_e < 0.4)  # [0]
    t_w[idx] = (k1_pi[idx] - 2.66) - ((k2_pi[idx] - 1.21) * normalized_t_e[idx]) + (0.58 * (normalized_t_e[idx]**-1))

    return t_w, t_e, nondim_pres

def wetlift_rdj(theta_e, p):
    '''
    Calculates the wetbulb temperature along a predefined pseudoadiabat.  This function uses
    the method created by Robert Davies-Jones (2008) to calculate the wetbulb temperature
    along a pseudoadidabat.  This method is faster and more accurate than the Wobus function
    previously used in SHARP.  This new function uses an improved first guess of the wetbulb temperature
    and then uses one iteration of an accelerated version of Newton's method to improve that guess.

    Parameters
    ----------
    theta_e : number, numpy array
        Equivalent potential temperature defining the pseudoadiabat (C)
    p : number, numpy array
        Pressure to calculate wetbulb temperature at (hPa)

    Returns
    -------
    t_w : number, numpy array
        Wetbulb temperature (C)
    '''
    # Get the improved first guess of T_w along the pseudoadiabat (Eq 4.8-4.11)
    tau_n, t_e, nondim_pres = _guess_Tw(ctok(theta_e), p)
    tau_n = ctok(tau_n)  # Convert the first guess to Celsius
    baseterm = np.ma.power(C / t_e, lambda_factor)
    # Get the derivative terms to perform Newton's Method
    f_tau_pi = _f(tau_n, p)
    dlnf_dt, des_dt, drs_dt, dG_dt, es, rs = _dlnfdt(tau_n, p)
    df_dt = (f_tau_pi) * (dlnf_dt)
    dlnf2_dt2 = _dlnf2dt2(tau_n, p, dlnf_dt, des_dt, drs_dt, dG_dt, f_tau_pi, es, rs)
    df2_dt2 = df_dt * dlnf_dt + f_tau_pi * dlnf2_dt2
    # Equation 2.8 in RDJ 2008 is flawed, but reference to Henrici 1964 is valid.
    # Using that equation instead.  See Eq 10-4 on pg 199 of Henrici:
    # https://ia800700.us.archive.org/23/items/ElementsOfNumericalAnalysis/Henrici-ElementsOfNumericalAnalysis.pdf
    c = f_tau_pi - baseterm
    b = df_dt
    a = df2_dt2 / 2.
    # Compute the roots for the accelerated method
    quad_tau_n1 = tau_n - ((2. * c) / (b + np.sqrt((b**2) - 4. * a * c)))
    quad_tau_n2 = tau_n - ((2. * c) / (b - np.sqrt((b**2) - 4. * a * c)))
    # Compute the increment for the non-accelerated method
    lin_tau_n2 = tau_n - (c / df_dt)
    # Pick the accelerated method root that is closest to the result from the non-accelerated method.
    #print quad_tau_n1, quad_tau_n2, lin_tau_n2, p
    idx = np.ma.argmin(np.ma.abs(np.ma.asarray([quad_tau_n1, quad_tau_n2]) - lin_tau_n2), axis=0)
    idx_true = np.where(np.ma.asarray(theta_e).mask == False)[0][0]
    #print idx, idx_true
    pseudoadiabat = np.ma.asarray([quad_tau_n1, quad_tau_n2])[idx[idx_true], :]
    #print pseudoadiabat
    return pseudoadiabat

"""
print "Bolton ThetaE:", thetae(1000, -40, -40, method='bolton')
print "Wobus ThetaE:", thetae(1000, -40, -40, method='wobus')

print "Bolton ThetaW:", thetaw(1000, 30, 20, method='bolton')
print "Wobus ThetaW:", thetaw(1000, 30, 20, method='wobus')

print "Bolton e:",vappres(25, method='bolton')
print "Wobus e:", vappres(25, method='wobus')

the = thetae(1000, 0, 0, method='bolton')

print _guess_Tw(ctok(the), 500)
pres = np.arange(1000,5,-5)
the = the * np.ones(len(pres))
from datetime import datetime
t = datetime.now()
tw = wetlift(1000,30, pres, theta_e=the, method='bolton')
print "Time for RDJ2008:", datetime.now() - t

t = datetime.now()
for p in pres:
    tw = wetlift(1000,30, p, theta_e=the, method='wobus')
print tw 
print "Time for Wobus:", datetime.now() - t
print len(pres)
"""
