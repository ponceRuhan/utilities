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
    CODE: speed.py
    
    This function aims to convert airspeed (calibrated, equivalent, true and mach number).
    And, calculate the dynamic pressure and impact pressure.
    
    ___________________
    REFERENCE:
    1 - Aircraft Performance Flight Test - Olson, 2000
'''
from sys import path, exit
path.insert(1, '..')
from numpy import zeros_like
from scipy.optimize import fsolve
from sty import bg
from atmosphere.ISA import ISA

class speed:
    def __init__(self, hp, disa = None, 
                       mach = None, eas = None, cas = None, tas = None):
        '''Convert airspeed and mach correction over dynamic pressure

        INPUTs:
            hp      : pressure-altitude             [m]
            disa    : ISA variation                 [K]
            mach    : mach number                   [-]
            eas     : equivalent airspeed           [m/s]
            cas     : calibrated airspeed           [m/s]
            tas     : true airspeed                 [m/s]

        OUTPUTs:
            mach    : mach number                               [-]
            eas     : equivalent airspeed                       [m/s]
            cas     : calibrated airspeed                       [m/s]
            tas     : true airspeed                             [m/s]
            dynamic_pressure    : dynamic pressure              [Pa]
            impact_pressure     : dynamic pressure correction with mach   [Pa]'''
        
        self.hp     = hp
        self.disa   = disa
        self.mach   = mach
        self.eas    = eas
        self.cas    = cas
        self.tas    = tas        
        self.calculate()

    def calculate(self):
        #Check de input of disa
        if self.disa is not None:
            atm = ISA(self.hp, self.disa) 
        else:
            self.disa = zeros_like(self.hp)
            atm = ISA(self.hp, self.disa)
        
        a = atm.speed_sound
        sigma = atm.ratio_density
        theta = atm.ratio_temperature
        delta = atm.ratio_pressure
        p = atm.pressure
        density = atm.density

        #constants
        gamma = atm.ratio_specific_heat
        p0 = atm.pressure_sealevel 
        a0 = atm.speed_sound_sealevel
        
        def cas_correction(cas, qc):
            '''Correction to Vc >= speed_sound_sealevel'''
            return cas - a0*0.881285*((qc/p0 + 1)*(1 - (1/(7*(cas/a0)**2)))**2.5)**0.5
        
        def impact_pressure(mach):
            if mach < 1 :
                '''Impact pressure
                Olson - Aircraft Performance Flight Testing Equation 4.9
                '''
                qc = p*((1.0 + 0.2*mach**2)**3.5 - 1.0)
            else:
                '''Impact pressure
                Olson - Aircraft Performance Flight Testing Equation 4.15
                '''
                qc = p*((1.2*mach**2)**3.5*((2.4/(-0.4 +
                    2.8*mach**2))**2.5) - 1.0)
            return qc

        def mach_correction(mach, qc):
            return mach - 0.881285*((qc/p + 1.0)*(1.0 - 1.0/(7.0*mach**2))**2.5)**0.5

        if self.mach is not None:            
            #impact pressure
            self.qc = impact_pressure(self.mach)

            #conversion from mach number to true airspeed
            self.tas = self.mach*a
            
            #dynamic pressure
            self.q = 0.5*density*self.tas**2

            '''equivalent airspeed
               Olson - Equation 4.26'''
            self.eas = self.tas*sigma**0.5

            '''calibrated airspeed
               Olson - Equation 4.20'''
            self.cas = a0*((5.0*((self.qc/p0 + 1.0)**(1.0/3.5) - 1)))**0.5
            
            if self.cas >= a0:
                self.cas = fsolve(cas_correction, self.cas, args = (self.qc))
        
        elif self.eas is not None:
            '''mach number
               Olson - Equation 4.28'''
            self.mach = self.eas/(a0*delta**0.5)

            '''true airspeed
               Olson - Equation 4.26'''
            self.tas = self.eas/sigma**0.5

            #impact pressure
            self.qc = impact_pressure(self.mach)

            #dynamic pressure
            self.q = 0.5*density*self.tas**2
            
            #calibrated airspeed
            '''Olson - Equation 4.20'''
            self.cas = a0*((5.0*((self.qc/p0 + 1.0)**(1.0/3.5) - 1)))**0.5
            
            if self.cas >= a0:
                self.cas = fsolve(cas_correction, self.cas, args = (self.qc))
        
        elif self.tas is not None:
            #mach number
            self.mach = self.tas/a

            #equivalent airspeed
            self.eas = self.tas*sigma**0.5

            #impact pressure
            self.qc = impact_pressure(self.mach)

            #dynamic pressure
            self.q = 0.5*density*self.tas**2
            
            #calibrated airspeed
            '''Olson - Equation 4.20'''
            self.cas = a0*((5.0*((self.qc/p0 + 1.0)**(1.0/3.5) - 1)))**0.5
            
            if self.cas >= a0:
                self.cas = fsolve(cas_correction, self.cas, args = (self.qc))

        elif self.cas is not None:
            
            if self.cas < a0:
                '''Impact pressure
                    Olson - Equation 4.19'''
                self.qc = p*((1 + 0.2*(self.cas/a0)**2)**3.5 - 1.0)
            else:
                '''Impact pressure
                   Olson - Equation 4.21'''
                self.qc = p*(166.9216*(self.cas/a0)**7/((7*(self.cas/a0)**2 
                                - 1)**2.5) - 1.0)
            
            #mach number
            self.mach = (5*((self.qc/p + 1)**(2.0/7.0) - 1.0))**0.5
            
            if self.mach >= 1.0:
                self.mach = fsolve(mach_correction, self.mach, args = (qc))
            
            #true airspeed
            self.tas = self.mach*a

            #equivalent airspeed
            self.eas = self.tas*sigma**0.5

            #dynamic pressure
            self.q = 0.5*density*self.tas**2
        else:
            print(bg.red + 'ERROR: ' + bg.rs + 'input error')
            exit()
    
    @property
    def get_mach(self):
        '''Mach number [-]'''
        return self.mach

    @property
    def get_eas(self):
        '''Equivalent airspeed [m/s]'''
        return self.eas

    @property
    def get_cas(self):
        '''Calibrated airspeed [m/s]'''
        return self.cas

    @property
    def get_tas(self):
        '''True airspeed [m/s]'''
        return self.tas

    @property
    def get_impact_pressure(self):
        '''Impact pressure [Pa]'''
        return self.qc

    @property
    def get_dynamic_pressure(self):
        '''Dynamic pressure [Pa]'''
        return self.q

if __name__ == '__main__':
    s = speed(10000, disa = 47.5, mach = 0.735)
    print(s.eas,s.cas,s.tas)
