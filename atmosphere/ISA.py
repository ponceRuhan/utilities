'''

			 █████╗ ███████╗██████╗  ██████╗ ███████╗██╗      ██████╗ ██╗    ██╗
			██╔══██╗██╔════╝██╔══██╗██╔═══██╗██╔════╝██║     ██╔═══██╗██║    ██║
			███████║█████╗  ██████╔╝██║   ██║█████╗  ██║     ██║   ██║██║ █╗ ██║
			██╔══██║██╔══╝  ██╔══██╗██║   ██║██╔══╝  ██║     ██║   ██║██║███╗██║
			██║  ██║███████╗██║  ██║╚██████╔╝██║     ███████╗╚██████╔╝╚███╔███╔╝
			╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚══╝ 
                                                                    
_______________________________________________________________________________________________

    AEROFLOW is an aerodynamics analysis project for aviation using multi-fidelity models.
    ___________________
    Author(s):
        Ruhan Ponce @ponceRuhan (gitHUB)/ ruhan.ponce@gmail.com
    
    __________________
    CODE: ISA.py
    
    Standart atmosphere model.
    
    _________________
    INTRODUCTION:
    
    The U.S Standard Atmosphere, 1976 is an idealized, steady-state
    representation of the earth's atmosphere from surface to 1000 km (geometric
    altitude)
    
    _______________
    REFERENCES:
    1 - U.S. STANDARD ATMOSPHERE 1976
        https://ntrs.nasa.gov/api/citations/19770009539/downloads/19770009539.pdf
'''
from sys import exit
from numpy import (array, array_equal, ones_like, unique, zeros_like, exp,
                   argwhere)
from scipy.interpolate import interp1d
from sty import bg

class constant:
    '''
    U.S. STANDAT ATMOSPHERE - CONSTANTS
    '''
    def __init__(self,hp):
        try:
            if len(hp) > 1:
                self.hp = array(hp, dtype = float)
        except:
            self.hp = array([hp], dtype = float)
    
    @property
    def mean_molecular_weight_sealevel(self):
        '''
        MEAN MOLECULAR WEIGHT AT SEA-LEVEL [kg/kmol]

        Standart air composition to sea-level dry air
        Gas   M (kg/kmol)   Fraction Vol. (%)
        N2  -> 28.01340    : 0.78084
        O2  -> 31.99880    : 0.209476
        Ar  -> 39.94800    : 0.00934
        CO2 -> 44.00995    : 0.000314
        Ne  -> 20.18300    : 1.818e-5
        He  -> 4.002600    : 5.24e-6
        Kr  -> 83.80000    : 1.14e-6
        Xe  -> 131.3000    : 8.7e-8
        CH4 -> 16.04303    : 2.0e-6
        H2  -> 2.015940    : 5.0e-7
        '''
        return (28.01340*0.78084  + 31.99880*0.209476 + 
                39.94800*0.00934  + 44.00995*0.000314 + 
                20.18300*1.818e-5 + 4.002600*5.24e-6  +
                83.80000*1.14e-6  + 131.3000*8.7e-8   +
                16.04303*2.0e-6   + 2.015940*5.0e-7)
    
    @property
    def mean_molecular_weight_ratio(self):
        M_per_M0 = ones_like(self.hp)
        pos = argwhere(self.hp >= 79000.0)[:,0] 
         
        if len(pos) != 0:
            H = array((79000, 79500, 80000, 80500, 81000, 81500,
                       82000, 82500, 83000, 83500, 84000, 84500))
            M_per_M0_table = array((1.0, 0.999996, 0.999988, 0.999969, 0.999938,
                              0.999904, 0.999864, 0.999822, 0.999778, 0.999731,
                              0.999681, 0.999679))
            M_per_M0[pos] = [interp1d(H, M_per_M0_table)(self.hp[p]) for p in pos]

        return M_per_M0

    @property
    def gas_constant(self):
        #Universal gas constant [(N.m)/(kmol.K)]
        return 8.31432e3
    
    @property
    def gravity_sealevel(self):
        #Sea-level value of the acceleration of gravity [m/s²]
        return 9.80665
    
    @property
    def pressure_sealevel(self):
        #Standart sea-level atmospheric pressure [Pa]
        return 101325.0
   
    @property
    def temperature_sealevel(self):
        #Temperature sea-level [K]
        return 288.15

    @property
    def density_sealevel(self):
        '''Density sea-level'''
        Rair0 = self.gas_constant/self.mean_molecular_weight_sealevel
        return self.pressure_sealevel/(Rair0*self.temperature_sealevel)
    
    @property
    def speed_sound_sealevel(self):
        T0 = self.temperature_sealevel
        M0 = self.mean_molecular_weight_sealevel
        Rair0 = self.gas_constant/M0
        gamma = self.ratio_specific_heat
        return (gamma*Rair0*T0)**0.5

    @property
    def air_constant(self):
        '''Gas constant of air [(N.m)/(kmol.K)]'''
        M0 = self.mean_molecular_weight_sealevel
        return self.gas_constant/(self.mean_molecular_weight_ratio*M0)

    @property
    def gradients(self):
        #Constants to determinate pressure  to any reference-level
        #Pb, where b in referent level
        M0 = self.mean_molecular_weight_sealevel
        g0 = self.gravity_sealevel
        Rair = self.air_constant
        
        #Molecular-scale temperatura gradient [K/m']
        LMb = zeros_like(self.hp)
        Hb   = zeros_like(self.hp)
        TMb  = zeros_like(self.hp)
        pb   = zeros_like(self.hp)
        
        lmb = array([-6.5e-3, 0.0, 1.0e-3, 2.8e-3, 0.0, -2.8e-3, -2.0e-3])
        hb  = array((0.0, 11000.0, 20000.0, 32000.0, 47000, 51000.0, 71000.0,
                     84852.0)) 
        for i,hp in enumerate(self.hp):
            pos = argwhere(hb <= hp)[:,0][-1]
            LMb[i] = lmb[pos]
            Hb[i]   = hb[pos]
            
            sumT    = sum([lmb[j]*(hb[j+1] - hb[j]) for j in range(pos)])
            TMb[i]  = self.temperature_sealevel + sumT
            
            #Condition to pressure calculation p-1 
            factor = 1
            for j,p in enumerate(range(pos)):
                sumT_pn1 = sum([lmb[k]*(hb[k+1] - hb[k]) for k in range(p)])
                sumT = sum([lmb[k]*(hb[k+1] - hb[k]) for k in range(p + 1)])
                
                tmb = self.temperature_sealevel + sumT
                tmb_pn1 = self.temperature_sealevel + sumT_pn1

                if lmb[p] != 0:
                    factor *= (tmb_pn1/tmb)**(g0/(Rair[p]*lmb[p]))
                else:
                    factor *= exp(-g0*(hb[p+1] - hb[p])/(Rair[p]*tmb_pn1))
            pb[i] = factor*self.pressure_sealevel
        return LMb, Hb, TMb, pb


    @property
    def effective_earth_radius(self):
        #Effective earth radius obtained from equation givem by Harrison (1968)
        return 6356766.0 #[m]
    
    @property
    def ratio_specific_heat(self):
        '''Ratio of specific heat gamma = cp/cv of air'''
        return 1.40
    
class ISA(constant):
    '''
    U.S. STANDAT ATMOSPHERE - MODELS

    INPUTS:
        - hp    : Altitude pressure or geopotential height              [m]
        - disa  : ISA variation                                         [K]
    
    OUTPUTS:
        - geometric_height      : geometric height                      [m]
        - gravity               : gravity acceleration                  [m/s²]
        - mean_molecular_weight : meam molecular weight of standart air [kg/kmol]
        - temperature           : static temperature                    [K]
        - pressure              : absolute static pressure              [Pa]
        - density               : mass density                          [kg/m³]
        - dynamic_viscosity     : dynamic viscosity                     [Pa.s]
        - kinematic_viscosity   : kinematic viscosity                   [m²/s]
        - thermal_conductivity  : thermal conductivity coefficient      [W/(m.K)] 
        - speed_sound           : speed of sound                        [m/s]
        - cp_mass               : specific heat at constant pressure    [J/(kg.K)]
        - cv_mass               : Specific heat at constant volume      [J/(kg.K)]
        - prandtl               : prandtl number                        [-]
    '''
    def __init__(self, hp, disa):
        constant.__init__(self, hp)
        try:
            if len(hp) > 1:
                self.hp = array(hp, dtype = float)
        except:
            self.hp = array([hp], dtype = float)

        try:
            if len(disa) > 1:
                self.disa = array(disa, dtype = float)
        except:
            self.disa = array([disa], dtype = float)
 
        if len(self.disa) == 1:
            self.disa = ones_like(self.hp)*self.disa
        else:
            print(bg.red + 'ERROR: ' + bg.rs + 'disa different lenght of hp')
            exit()

    @property
    def geometric_height(self):
        'Geometric height [m]'
        r0 = self.effective_earth_radius
        Gamma = 1 #g0/g0' = 1 m'/m
        return (r0*self.hp)/(Gamma*r0 - self.hp)

    @property
    def gravity(self):
        '''Gravity acceleration [m/s²]

        this equation is the inver-square law of gravitation
        '''
        g0 = self.gravity_sealevel
        r0 = self.effective_earth_radius
        Z  = self.geometric_height
        return g0*(r0/(r0 + Z))**2
    
    @property
    def mean_molecular_weight(self):
        '''Mean molecular weight [kg/kmol]'''
        return self.mean_molecular_weight_ratio*self.mean_molecular_weight_sealevel

    @property
    def temperature(self):
        LMb, Hb, TMb, _ = self.gradients
        return TMb + LMb*(self.hp - Hb) + self.disa

    @property
    def pressure(self):
        M0   = self.mean_molecular_weight_sealevel
        g0   = self.gravity_sealevel
        p0   = self.pressure_sealevel
        Rair = self.air_constant
        LMb, Hb, TMb, pb = self.gradients
         
        p = zeros_like(LMb)        
        for i, lmb in enumerate(LMb):
            if lmb != 0:
                p[i] = pb[i]*(TMb[i]/(TMb[i] +
                        LMb[i]*(self.hp[i] - Hb[i])))**(g0/(Rair[i]*LMb[i]))
            else:
                p[i] = pb[i]*exp(-g0*(self.hp[i] - Hb[i])/(Rair[i]*TMb[i]))
        return p

    @property
    def density(self):
        'Mass density [kg/m³]'
        return self.pressure/(self.air_constant*self.temperature) 

    @property
    def dynamic_viscosity(self):
        '''Dynamic viscosity - Sutherland's law [(N.s)/m²] = [Pa.s]'''
        #Constants
        beta = 1.458e-6 #[kg/(s.m.K^0.5)]
        S = 110.4 #[K]
        return beta*self.temperature**(3/2)/(self.temperature + S)

    @property
    def kinematic_viscosity(self):
        '''Kinematic viscosity [m²/s]'''
        return self.dynamic_viscosity/self.density

    @property
    def thermal_conductivity(self):
        ''' Coefficient of thermal coductivity [W/(m.K)]'''
        return (2.64638e-3*self.temperature**(3/2)/(self.temperature +
                245.4*10**(-12/self.temperature)))

    @property
    def speed_sound(self):
        ''' Speed of sound [m/s] '''
        return (self.ratio_specific_heat*self.air_constant*self.temperature)**0.5
    
    @property
    def cp_mass(self):
        '''Specific heat at constant pressure [J/(kg.K)]
        Rair = cp - cv and gamma = cp/cv
        '''
        return self.air_constant*self.ratio_specific_heat/(self.ratio_specific_heat - 1) 

    @property
    def cv_mass(self):
        '''Specific heat at constant volume [J/(kg.K)]
        gamma = cp/cv
        '''
        return self.cp_mass/self.ratio_specific_heat

    @property
    def prandtl(self):
        '''Prandtl number [-]'''
        return self.cp_mass*self.dynamic_viscosity/self.thermal_conductivity

    @property
    def ratio_temperature(self):
        '''Ratio of temperature in relation of sea-level'''
        return self.temperature/self.temperature_sealevel
    
    @property
    def ratio_pressure(self):
        '''Ratio of pressure in relation of sea-level'''
        return self.pressure/self.pressure_sealevel

    @property
    def ratio_density(self):
        '''Ratio of density in relation if sea-level'''
        return self.density/self.density_sealevel

