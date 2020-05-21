import pandas

#using variable and parameter names that are consistent with the GAMS formulation available at the end of
#http://www.econ.yale.edu/~nordhaus/homepage/documents/Dicemanualfull.pdf


#A Param object can contain any value: string, floating point, etc.
#The default domain for parameters is Any, so by default no domain validation is
#performed. However, it is often valuable to specify the space of valid
#parameter values to pro- vide checking of the input data.
#the domain of feasible parameter values can be specified using the within option


# TO execute in Python simply run
# execfile('DICE16.py')
Run_as_DICE16 = True
if Run_as_DICE16 == True:
	import os
	import csv
	import random
	import scipy
	import sqlite3
	import numpy as np
	import matplotlib.pyplot as plt
	from pyomo.environ import *
	from pyomo.dae import *
	m = ConcreteModel()
	# dicsr_points = range(duration)
	duration = 99 # 300
	tttstep = 1 #5.0
	dicsr_points = range(100) #np.arange(0.0, duration, tttstep)
	m.t = ContinuousSet(bounds=(0,duration), initialize = dicsr_points)

	### Declare all suffixes
	# Ipopt bound multipliers (obtained from solution)
	m.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
	m.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
	# Ipopt bound multipliers (sent to solver)
	m.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
	m.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
	# Obtain dual solutions from first solve and send to warm start
	m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
	###


#m.miuchange = Param(initialize = 0.1, mutable=True)
####PARAMETERS
paramDF = pandas.read_csv('diceParams16.csv')
params = dict(zip(paramDF.key,paramDF.value))
#### Availability of fossil fuels
m.fosslim = Param(initialize = params['fosslim']) #Maximum cumulative extraction fossil fuels (GtC)  /6000/
#### Time Step
m.tstep = Param(initialize = params['tstep']) #Years per Period                                  /5/
#### If optimal control
# m.ifopt = Param(initialize = params['ifopt']) #Indicator where optimized is 1 and base is 0/0/
#### Preferences
m.elasmu = Param(initialize = params['elasmu']) #Elasticity of marginal utility of consumption/1.45 /
m.prstp = Param(initialize = params['prstp']) #Initial rate of social time preference per year/.015  /
#### Population and technology
m.gama = Param(initialize = params['gama']) #Capital elasticity in production function        /.300    /
m.pop0 = Param(initialize = params['pop0']) #Initial world population 2015 (millions)         /7403    /
m.popadj = Param(initialize = params['popadj']) #Growth rate to calibrate to 2050 pop projection/0.134   /
m.popasym = Param(initialize = params['popasym']) #Asymptotic population (millions)                 /11500   /
m.dk = Param(initialize = params['dk']) #Depreciation rate on capital (per year)/.100    /
m.q0 = Param(initialize = params['q0']) #Initial world gross output 2015 (trill 2010 USD/105.5   /
m.k0 = Param(initialize = params['k0']) #Initial capital value 2015 (trill 2010 USD)/223     /
m.a0 = Param(initialize = params['a0']) #Initial level of total factor productivity/5.115    /
m.ga0 = Param(initialize = params['ga0']) #Initial growth rate for TFP per 5 years/0.076   /
m.dela = Param(initialize = params['dela']) #Decline rate of TFP per 5 years/0.005   /
#### Emissions parameters
m.gsigma1 = Param(initialize = params['gsigma1']) #Initial growth of sigma (per year)                   /-0.0152 /
m.dsig = Param(initialize = params['dsig']) #Decline rate of decarbonization (per period)         /-0.001  /
m.eland0 = Param(initialize = params['eland0']) #Carbon emissions from land 2015 (GtCO2 per year)     / 2.6    /
m.deland = Param(initialize = params['deland']) #Decline rate of land emissions (per period)          / .115   /
m.e0 = Param(initialize = params['e0']) #Industrial emissions 2015 (GtCO2 per year)           /35.85    /
m.miu0 = Param(initialize = params['miu0']) #Initial emissions control rate for base case 2015    /.03     /
#### Carbon cycle
## Initial Conditions
m.mat0 = Param(initialize = params['mat0']) #Initial Concentration in atmosphere 2015 (GtC)/851    /
m.mu0 = Param(initialize = params['mu0']) #Initial Concentration in upper strata 2015 (GtC)/460    /
m.ml0 = Param(initialize = params['ml0']) #Initial Concentration in lower strata 2015 (GtC)/1740   /
m.mateq = Param(initialize = params['mateq']) #Equilibrium concentration atmosphere  (GtC)/588    /
m.mueq = Param(initialize = params['mueq']) #Equilibrium concentration in upper strata (GtC)/360    /
m.mleq = Param(initialize = params['mleq']) #Equilibrium concentration in lower strata (GtC)/1720   /
## Flow paramaters
m.b12 = Param(initialize = params['b12']) #Carbon cycle transition matrix/.12   /
m.b23 = Param(initialize = params['b23']) #Carbon cycle transition matrix/0.007 /
## Parameters for long-run consistency of carbon cycle
        # b11 = 1 - b12;
        # b21 = b12*MATEQ/MUEQ;
        # b22 = 1 - b21 - b23;
        # b32 = b23*mueq/mleq;
        # b33 = 1 - b32 ;
m.b11 = Param(initialize = 1 - m.b12)
m.b21 = Param(initialize = m.b12 * m.mateq / m.mueq)
m.b22 = Param(initialize = 1- m.b21 - m.b23)
m.b32 = Param(initialize = m.b23 * m.mueq / m.mleq)
m.b33 = Param(initialize = 1 - m.b32)
#converts trillion $ -> GtC02
m.sig0 = Param(initialize = m.e0/(m.q0*(1-m.miu0))) #Carbon intensity 2010 (kgCO2 per output 2005 USD 2010)

####  Climate model parameters
m.t2xco2 = Param(initialize = params['t2xco2']) #Equilibrium temp impact (oC per doubling CO2)/ 3.1  /
m.fex0 = Param(initialize = params['fex0']) #2015 forcings of non-CO2 GHG (Wm-2)/ 0.5  /
m.fex1 = Param(initialize = params['fex1']) #2100 forcings of non-CO2 GHG (Wm-2)/ 1.0  /
m.tocean0 = Param(initialize = params['tocean0']) #Initial lower stratum temp change (C from 1900)/.0068 /
m.tatm0 = Param(initialize = params['tatm0']) #Initial atmospheric temp change (C from 1900)/0.85  /
m.c1 = Param(initialize = params['c1']) #Climate equation coefficient for upper level/0.1005  /
m.c3 = Param(initialize = params['c3']) #Transfer coefficient upper to lower stratum     /0.088   /
m.c4 = Param(initialize = params['c4']) #Transfer coefficient for lower level             /0.025   /
m.fco22x = Param(initialize = params['fco22x']) #Forcings of equilibrium CO2 doubling (Wm-2)      /3.6813  /
####  Climate damage parameters
m.a1 = Param(initialize = params['a1']) #Damage intercept/0       /
m.a2 = Param(initialize = params['a2']) #Damage quadratic term/0.00236 /
m.a3 = Param(initialize = params['a3']) #Damage exponent/2.00    /
m.a10 = Param(initialize = params['a10']) #Initial damage intercept/0/
m.a20 = Param(initialize = m.a2) #Initial damage quadratic term
####  Abatement cost
m.expcost2 = Param(initialize = params['expcost2']) #Exponent of control cost function/ 2.6  /
m.pback = Param(initialize = params['pback']) #Cost of backstop 2010$ per tCO2 2015/ 550  /
m.gback = Param(initialize = params['gback']) #Initial cost decline backstop cost per period/ .025 /
m.limmiu = Param(initialize = params['limmiu']) #Upper limit on control rate after 2150/ 1.2 /
m.tnopol = Param(initialize = params['tnopol']) #Period before which no emissions controls base / 45   /
m.cprice0 = Param(initialize = params['cprice0']) #Initial base carbon price (2010$ per tCO2)/ 2    /
m.gcprice = Param(initialize = params['gcprice']) #Growth rate of base carbon price per year/.02   /

####  Scaling and inessential parameters
## Note that these are unnecessary for the calculations
## They ensure that MU of first period's consumption =1 and PV cons = PV utilty
m.scale1 = Param(initialize = params['scale1']) #Multiplicative scaling coefficient/0.0302455265681763 /
m.scale2 = Param(initialize = params['scale2']) #Additive scaling coefficient/-10993.704/

#Rules to derive PARAMETERS
def _Population(model, i): #l(t+1)=l(t)*(popasym/L(t))**popadj
	if i == model.t.first():
		return model.pop0
	else:
		prevpop = model.l[model.t.prev(i)]
		return prevpop * (m.popasym/prevpop)**m.popadj

def _GrowthRateOfProductivity(model, i): #ga(t)=ga0*exp(-dela*5*((t.val-1)));
	per = model.t.ord(i) - 1
	return model.ga0 * exp(-model.dela * model.tstep * per)

def _LevelOfProductivity(model, i): #al(t+1)=al(t)/((1-ga(t))
	if i == model.t.first():
		return model.a0
	else:
		return model.al[model.t.prev(i)]/(1 - model.ga[model.t.prev(i)])

def _CumulativeEfficiencyImprovement(model, i): #gsig(t+1)=gsig(t)*((1+dsig)**tstep)
	if i == model.t.first():
		return model.gsigma1
	else:
		return model.gsig[model.t.prev(i)] * (1 + model.dsig)**model.tstep

def _GrowthRate(model, i): #sigma(t+1)=(sigma(t)*exp(gsig(t)*tstep)
	if i == model.t.first():
		return model.sig0
	else:
		prevgsig = model.gsig[model.t.prev(i)]
		return model.sigma[model.t.prev(i)] * exp(model.tstep * prevgsig)

def _BackstopPrice(model, i):#pback*(1-gback)**(t.val-1)
	per = model.t.ord(i) - 1
	return model.pback * (1-model.gback) ** per
#
def _AdjustedCostForBackstop(model, i): #pbacktime(t)*sigma(t)/expcost2/1000
	return model.pbacktime[i] * model.sigma[i]/model.expcost2/1000
	#sigma converts YGROSS in 10^12 USD to ind emissions (10^9 tonnes of CO2), OK

def _EmmissionsFromDeforestation(model, i): #eland0*(1-deland)**(t.val-1)
	per = model.t.ord(i) - 1
	return model.eland0*(1-model.deland) ** per

def _CumEmmissionsFromLand(model, i): #cumetree(t)+etree(t)*(5/3.666)
	if i == model.t.first():
		return 100
	else:
		per = model.t.ord(i) - 1
		return model.cumetree[model.t.prev(i)]+model.etree[model.t.prev(i)]*(model.tstep/3.666)

def _AverageUtilitySocialDiscountRate(model, i): #1/((1+prstp)**(tstep*(t.val-1)))
	per = model.t.ord(i) - 1
	return 1./((1+model.prstp)**(model.tstep*per))

def _ExogenousForcingOfOtherGreenhouseGases(model, i): #fex0+ (1/17)*(fex1-fex0)*(t.val-1)$(t.val lt 18)+ (fex1-fex0)$(t.val ge 18)
	per = model.t.ord(i) - 1
	if model.t.ord(i) <= 17:
		return model.fex0 + (1./17.)*(model.fex1-model.fex0)*per
	else:
		return model.fex1

def _CarbonPriceBase(model, i): #cpricebase(t)= cprice0*(1+gcprice)**(5*(t.val-1))
	per = model.t.ord(i) - 1
	return model.cprice0*(1+model.gcprice)**(model.tstep*per)

### DERIVED PARAMETERS
m.l = Param(m.t, initialize = _Population) #Level of population and labor
m.ga = Param(m.t, initialize = _GrowthRateOfProductivity) #Growth rate of productivity from
m.al = Param(m.t, initialize = _LevelOfProductivity) #Level of total factor productivity
m.gsig = Param(m.t, initialize = _CumulativeEfficiencyImprovement) #Change in sigma (cumulative improvement of energy efficiency)
m.sigma = Param(m.t, initialize = _GrowthRate) #CO2-equivalent-emissions output ratio
m.pbacktime = Param(m.t, initialize = _BackstopPrice) #Backstop price
m.cost1 = Param(m.t, initialize = _AdjustedCostForBackstop) #Adjusted cost for backstop
m.etree = Param(m.t, initialize = _EmmissionsFromDeforestation) #Emissions from deforestation
m.cumetree = Param(m.t, initialize = _CumEmmissionsFromLand) #Cumulative from land
m.rr = Param(m.t, initialize = _AverageUtilitySocialDiscountRate) #Average utility social discount rate (factor!?)
m.forcoth = Param(m.t, initialize = _ExogenousForcingOfOtherGreenhouseGases) #Exogenous forcing for other greenhouse gases
m.cpricebase = Param(m.t, initialize = _CarbonPriceBase) #Carbon price in base case
#Climate model parameter
m.lam = Param(initialize = m.fco22x/ m.t2xco2)
#Optimal long-run savings rate used for transversality
m.optlrsav = Param(initialize = (m.dk + .004)/(m.dk + .004*m.elasmu + m.prstp)*m.gama) #(dk + .004)/(dk + .004*elasmu + prstp)*gama


#         gl(t)         Growth rate of labor
#         gcost1        Growth of cost factor#
#         gfacpop(t)    Growth factor population
#         scc(t)        Social cost of carbon
#         photel(t)     Carbon Price under no damages (Hotelling rent condition)
#         ppm(t)        Atmospheric concentrations parts per million
#         atfrac(t)     Atmospheric share since 1850
#         atfrac2010(t)     Atmospheric share since 2010 ;
#
# ###SHARED VARIABLES###
#Using the all-caps standard as set in the GAMS model



def miuBounds(model,i):
	if i == model.t.first():
		return (model.miu0, model.miu0)
	elif model.t.ord(i) < 30:
		return (0.0001, 1.)
	else:
		return (0.0001, 1.2)# model.limmiu)
#I need to check initial concentrations

def cBounds(model,i):
	return (2.,scipy.inf)

def kBounds(model,i):
	if i == model.t.first():
		return (model.k0, model.k0)
	else:
		return (1., scipy.inf)

def cpcBounds(model,i):
	return (.01, scipy.inf)

def sBounds(model,i):
	per = model.t.ord(model.t.last()) - 10
	if model.t.ord(i) <= per:
		return (0, 1.0) # a reasonable assumption HELPS
	else:
		return (model.optlrsav, model.optlrsav)

def ccaBounds(model,i):
	if i == m.t.first():
		return (400.,400.)
	else:
		return (-model.fosslim, model.fosslim) #lower bound is set to speed up convergence

m.MIU = Var(m.t,domain=NonNegativeReals,bounds=miuBounds) #Emission control rate GHGs
m.C = Var(m.t,domain=NonNegativeReals,bounds=cBounds) #Consumption (trillions 2005 US dollars per year, 1e+12)
m.K = Var(m.t,domain=NonNegativeReals,bounds=kBounds) #Capital stock (trillions 2005 US dollars)
m.CPC = Var(m.t,domain=NonNegativeReals,bounds=cpcBounds) #Per capita consumption (thousands 2005 USD per year)
m.I = Var(m.t,domain=NonNegativeReals) #Investment (trillions 2005 USD per year)
m.S = Var(m.t,domain=Reals,bounds=sBounds) #Gross savings rate as fraction of gross world product
m.RI = Var(m.t,domain=Reals) #Real interest rate (per annum)
m.Y = Var(m.t,domain=NonNegativeReals) #Gross world product net of abatement and damages (trillions 2005 USD per year)
m.YGROSS = Var(m.t,domain=NonNegativeReals) #Gross world product GROSS of abatement and damages (trillions 2005 USD per year)
m.YNET = Var(m.t,domain=Reals) #Output net of damages equation (trillions 2005 USD per year)
m.DAMAGES = Var(m.t,domain=Reals) #Damages (trillions 2005 USD per year)
m.DAMFRAC = Var(m.t,domain=Reals)#, bounds = (-0.05, 0.34)) #Damages as fraction of gross output
m.ABATECOST = Var(m.t,domain=Reals) #Cost of emissions reductions  (trillions 2005 USD per year)
m.MCABATE = Var(m.t,domain=Reals) #Marginal cost of abatement (2005$ per ton CO2)
m.CCA = Var(m.t,domain=Reals,bounds=ccaBounds) #Cumulative industrial carbon emissions (GTC)
m.PERIODU = Var(m.t,domain=Reals) #One period utility function
m.CPRICE = Var(m.t,domain=Reals) #Carbon price (2005$ per ton of CO2)
m.CEMUTOTPER = Var(m.t,domain=Reals) #Period utility
m.UTILITY = Var(domain=Reals, initialize = 4500, bounds = (4000, 5000)) #Welfare function


###Additional vars ONLY if we RUN Dice2016R
if Run_as_DICE16 == True:
	def matBounds(model,i):
		if i == m.t.first():
			return (model.mat0, model.mat0) #Initial conditions
		else:
			return (10.,scipy.inf)
	def tatmBounds(model,i):
		if i == m.t.first():
			return (model.tatm0, model.tatm0) #Initial conditions
		else:
			return (0.,10.) #AB: to improve convergence (0., 12.35) --> (0., 10.)
	def toceanBounds(model,i):
		if  i == m.t.first():
			return (model.tocean0, model.tocean0) #Initial conditions
		else:
			return (0., 10.) #AB: to improve convergence (-1., 20.0) --> (0., 10.)
	def muBounds(model,i):
		if i == m.t.first():
			return (model.mu0, model.mu0) #Initial conditions
		else:
			return (100.,scipy.inf)
	def mlBounds(model,i):
		if i == m.t.first():
			return (model.ml0, model.ml0) #Initial conditions
		else:
			return (1000.,scipy.inf)
	m.FORC = Var(m.t,domain=Reals) #Increase in radiative forcing (watts per m2 from 1900)
	m.TATM = Var(m.t,domain=NonNegativeReals,bounds=tatmBounds) #Increase temperature of atmosphere (degrees C from 1900)
	m.TOCEAN = Var(m.t,domain=Reals,bounds=toceanBounds) #Increase temperatureof lower oceans (degrees C from 1900)
	m.MAT = Var(m.t,domain=NonNegativeReals,bounds=matBounds) #Carbon concentration increase in atmosphere (GtC from 1750)
	m.MU = Var(m.t,domain=NonNegativeReals,bounds=muBounds) #Carbon concentration increase in shallow oceans (GtC from 1750)
	m.ML = Var(m.t,domain=NonNegativeReals,bounds=mlBounds) #Carbon concentration increase in lower oceans (GtC from 1750)
	m.EIND = Var(m.t,domain=Reals) #Industrial emissions (GtCO2 per year)
	m.E = Var(m.t,domain=Reals) #Total CO2 emissions (GtCO2 per year, 1e+9 tonnes)

	#Emissions equation;  E(t) =E= EIND(t) + etree(t)
	def _emmissions_eq(m,t):
		return m.E[t] == m.EIND[t] + m.etree[t]
	m.emmissions_eq = Constraint(m.t, rule = _emmissions_eq)

	def _DICE_industrialEmmissions_eq(m,t): #EIND(t)=E= sigma(t) * YGROSS(t) * (1-(MIU(t)));
		return m.EIND[t] == m.sigma[t] * m.YGROSS[t] * (1 - m.MIU[t])
	#we declare the actual constraint below

	#Radiative forcing equation;FORC(t)=E= fco22x * ((log((MAT(t)/588.000))/log(2))) + forcoth(t);
	def _DICE_radiativeForcing(m,t):
		return m.FORC[t] == m.fco22x * ( log(m.MAT[t] / 588.0)/log(2) ) + m.forcoth[t]
	m.radiativeForcing = Constraint(m.t, rule=_DICE_radiativeForcing)

	def _DICE_damageFraction(m,t):
		return m.DAMFRAC[t] == m.a1 * m.TATM[t] + m.a2 * m.TATM[t]**m.a3
	#we declare the actual constraint below

	############################ Climate and carbon cycle is all here
	# Lower ocean concentration;  ML(t+1) =E= ML(t)*b33  + MU(t)*b23
	def _DICE_lowerOceanConcentration(m,t):
		if t == m.t.first():
			return Constraint.Skip
		else:
			prt = m.t.prev(t)
			return m.ML[t] == m.ML[prt] * m.b33 + m.MU[prt] * m.b23
	m.lowerOceanConcentration = Constraint(m.t,rule=_DICE_lowerOceanConcentration)

	# Shallow ocean concentration; MU(t+1)=E= MAT(t)*b12 + MU(t)*b22 + ML(t)*b32;
	def _DICE_upperOceanConcentration(m,t):
		if t == m.t.first():
			return Constraint.Skip
		else:
			prt = m.t.prev(t)
			return m.MU[t] == m.ML[prt]*m.b32 + m.MU[prt] * m.b22 + m.MAT[prt] * m.b12
	m.upperOceanConcentration = Constraint(m.t,rule=_DICE_upperOceanConcentration)

	# Temperature-climate equation for atmosphere; TATM(t) + c1 * ((FORC(t+1)-(lam*TATM(t))-(c3*(TATM(t)-TOCEAN(t))));
	def _DICE_atmosphericTemperature(m,t):
		if t == m.t.first():
			return Constraint.Skip
		else:
			prt = m.t.prev(t)
			return m.TATM[t] == m.TATM[prt] \
			+ m.c1 * ((m.FORC[t] - m.lam * m.TATM[prt]) \
			- m.c3 * (m.TATM[prt] - m.TOCEAN[prt]))
	m.atmosphericTemperature = Constraint(m.t,rule=_DICE_atmosphericTemperature)

	# Temperature-climate equation for lower oceans; TOCEAN(t+1) =E= TOCEAN(t) + c4*(TATM(t)-TOCEAN(t));
	def _DICE_oceanTemperature(m,t):
		if t == m.t.first():
			return Constraint.Skip
		else:
			prt = m.t.prev(t)
			return m.TOCEAN[t] == m.TOCEAN[prt] + m.c4 * (m.TATM[prt] - m.TOCEAN[prt])
	m.oceanTemperature = Constraint(m.t,rule=_DICE_oceanTemperature)

	# Atmospheric concentration equation; MAT(t+1)=E= MAT(t)*b11 + MU(t)*b21 + (E(t)*(tstep/3.666));
	def _DICE_atmosphericConcentration(m,t):
		if t == m.t.first():
			return Constraint.Skip
		else:
			prt = m.t.prev(t)
			return m.MAT[t] == m.MAT[prt]*m.b11 + m.MU[prt] * m.b21 + m.E[prt] * m.tstep / 3.666
	m.atmosphericConcentration = Constraint(m.t,rule = _DICE_atmosphericConcentration)

	#Cumulative industrial carbon emissions; CCA(t+1) =E= CCA(t)+ EIND(t)*5/3.666;
	def _DICE_cumCarbonEmmissions(m,t):
		if t == m.t.first():
			return Constraint.Skip
		else:
			prt = m.t.prev(t)
			return m.CCA[t] == m.CCA[prt] + m.EIND[prt] * m.tstep / 3.666


### EQUATIONS
############################ Emissions and Damages

#Cumulative industrial carbon emissions; CCA(t+1) =E= CCA(t)+ EIND(t)*5/3.666;
def _cumCarbonEmmissions(m,t):
	if t == m.t.first():
		return Constraint.Skip
	else:
		prt = m.t.prev(t)
		return m.CCA[t] == m.CCA[prt] + m.Eco2[prt] * m.tstep

# Damage equation; DAMAGES(t)     =E= YGROSS(t) * DAMFRAC(t)
def _damageEq(m,t):
	return m.DAMAGES[t] == m.YGROSS[t] * m.DAMFRAC[t]
m.damageEq = Constraint(m.t, rule = _damageEq)

#Cumulative total carbon emissions; CCATOT(t)      =E= CCA(t)+cumetree(t);
##?


#Cost of emissions reductions equation; ABATECOST(t)   =E= YGROSS(t) * cost1(t) * (MIU(t)**expcost2);
def _abatementCost(m,t):
	return m.ABATECOST[t] == m.YGROSS[t] * m.cost1[t] * m.MIU[t]**m.expcost2
m.abatementCost = Constraint(m.t, rule = _abatementCost)

#Equation for MC abatement; MCABATE(t)     =E= pbacktime(t) * MIU(t)**(expcost2-1);
def _mcAbatement(m,t):
 	ex = m.expcost2 - 1
	return m.MCABATE[t] == m.pbacktime[t] * m.MIU[t]**ex
m.mcAbatement = Constraint(m.t,rule=_mcAbatement)


# #NOTE PROBLEM: these are to similar equations!
# #Carbon price equation from abatement; CPRICE(t)      =E= pbacktime(t) * (MIU(t))**(expcost2-1);
# def _carbonPrice(m,t): #there was a PARTFRAC in DICE13. This fraction was used to define CPRICE
# #Now we do not have it anymore, so CPRICE doesnot make sense. UNcomment if needed
# 	ex = m.expcost2 - 1
# 	return m.CPRICE[t] == m.pbacktime[t] * m.MIU[t]**ex
# m.carbonPriceEq = Constraint(m.t,rule=_carbonPrice)

######==================================================
######===============LINKAGE is here ===================
######==================================================
### We want to link m.EIND[t]
### to Eco2 functio (We assume that E[t] and EIND[t] are the same)
#DICE ORIGINAL: EIND(t)=E= sigma(t) * YGROSS(t) * (1-(MIU(t)));
def _industrialEmmissions_eq(m,t): #EIND(t)        =E= sigma(t) * YGROSS(t) * (1-(MIU(t)));
	return m.Eco2[t] == m.sigma[t] * m.YGROSS[t] * (1 - m.MIU[t])/3.666 #we convert to carbon equivalent
#DICE ORIGINAL: DAMFRAC(t)=E= (a1*TATM(t))+(a2*TATM(t)**a3) ;
def _damageFraction(m,t):
	return m.DAMFRAC[t] == m.a1 * m.T[t] + m.a2 * m.T[t]**m.a3

if Run_as_DICE16 == True:
	m.industrialEmmissions = Constraint(m.t,rule= _DICE_industrialEmmissions_eq)
	m.damageFraction = Constraint(m.t,rule=_DICE_damageFraction)
	m.cumCarbonEmmissions = Constraint(m.t, rule = _DICE_cumCarbonEmmissions)
else:
	m.industrialEmmissions = Constraint(m.t,rule=_industrialEmmissions_eq)
	m.damageFraction = Constraint(m.t,rule=_damageFraction)
	m.cumCarbonEmmissions = Constraint(m.t, rule = _cumCarbonEmmissions)
######==================================================
######==================================================
######==================================================

### Economic variables

# Output gross equation; YGROSS(t)=E= (al(t)*(L(t)/1000)**(1-GAMA))*(K(t)**GAMA);
def _grossOutput(m,t):
	coeff = m.al[t]* (m.l[t]/1000.)**(1-m.gama)
	return m.YGROSS[t] == coeff * m.K[t]**m.gama
m.grossOutput = Constraint(m.t,rule=_grossOutput)
# Output net of damages equation; YNET(t)=E= YGROSS(t)*(1-damfrac(t));
def _netDamages(m,t):
	return m.YNET[t] == m.YGROSS[t] * (1-m.DAMFRAC[t])
m.netDamages = Constraint(m.t,rule=_netDamages)
# Output net equation; Y(t)=E= YNET(t) - ABATECOST(t);
def _netOutput(m,t):
	return m.Y[t] == m.YNET[t] - m.ABATECOST[t]
m.netOutput = Constraint(m.t,rule=_netOutput)
# Consumption equation; C(t)=E= Y(t) - I(t);
def _consumption(m,t):
	return m.C[t] == m.Y[t] - m.I[t]
m.consumption = Constraint(m.t,rule=_consumption)
# Per capita consumption definition; CPC(t)=E= 1000 * C(t) / L(t);
def _perCapitaConsumption(m,t):
	return m.CPC[t] == 1000. * m.C[t] / m.l[t] ## 10^12/10^6 * 1000 == thousands per capita
m.perCapitaConsumption = Constraint(m.t,rule=_perCapitaConsumption)
# Savings rate equation; I(t)=E= S(t) * Y(t);
def _savingsRate(m,t):
	return m.I[t] ==  m.S[t] * m.Y[t]
m.savingsRate = Constraint(m.t,rule=_savingsRate)


# NOTE that the DICE in GAMS uses <= instead of ==
# Capital balance equation; K(t+1)=L= (1-dk)**tstep * K(t) + tstep * I(t);
def _capitalBalance(m,t):
	if t == m.t.first():
		return Constraint.Skip
	else:
		prt = m.t.prev(t)
		return m.K[t] == (1 - m.dk)**m.tstep * m.K[prt] + m.tstep * m.I[prt]
m.capitalBalance = Constraint(m.t,rule=_capitalBalance)
# Interest rate equation; RI(t)=E= (1+prstp) * (CPC(t+1)/CPC(t))**(elasmu/tstep) - 1;
def _interestRate(m,t):
	if t == m.t.last():
		return Constraint.Skip
	else:
		nxt = m.t.next(t) #See Dasgupta 2008, p. 149, Eq. 4
		expr = (m.CPC[nxt]/m.CPC[t])**(m.elasmu/m.tstep)
		return m.RI[t] == (1 + m.prstp) * expr - 1
m.interestRate = Constraint(m.t,rule=_interestRate)

### Utility
# Period utility; CEMUTOTPER(t)=E= PERIODU(t) * L(t) * rr(t);
def _periodUtility(m,t):
	return m.CEMUTOTPER[t] ==  m.PERIODU[t] * m.l[t] * m.rr[t]
m.periodUtility = Constraint(m.t,rule=_periodUtility)
# Instantaneous utility function equation; PERIODU(t)=E= ( (C(T)*1000/L(T))**(1-elasmu)-1 )/ (1-elasmu) - 1;
def _instUtility(m,t):
	expr = (m.C[t] * 1000 / m.l[t])**(1-m.elasmu) #the same as (m.CPC)**(1-m.elasmu)
	return m.PERIODU[t] ==  (expr-1) / (1-m.elasmu) - 1
m.instUtility = Constraint(m.t,rule=_instUtility)
# Objective function; UTILITY=E= tstep * scale1 * sum(t,  CEMUTOTPER(t)) + scale2 ;
def _utilityCalc(m):
	return m.UTILITY == m.tstep * m.scale1 * summation(m.CEMUTOTPER) + m.scale2
m.utilityCalc = Constraint(rule=_utilityCalc)


if Run_as_DICE16 == True:
	def obj_rule(m):
		return  m.UTILITY
	m.OBJ = Objective(rule=obj_rule, sense = maximize)
	# solver_manager = SolverManagerFactory('neos')
	# results = solver_manager.solve(m, opt="conopt")
	# results.write()

	solver=SolverFactory('ipopt') # bonmin
	solver.options['halt_on_ampl_error'] = 'yes'
	solver.options['acceptable_tol'] = 1e-8
	solver.options['constr_viol_tol'] = 1e-12
	solver.options['max_iter'] = 1000
	#solver.options['warm_start_init_point'] = 'yes'
	# solver.options['hessian_approximation'] = 'limited-memory'
	# solver.options['symbolic_solver_labels'] = 'yes' #True
	results = solver.solve(m,tee=True)

	print m.OBJ.expr.value
	#estimate SCC as in DICE16: scc(t) = -1000*eeq.m(t)/(.00001+cc.m(t));
	scc_RUN = True
	if scc_RUN == True:
		cond16 = sqlite3.connect(r'/Users/ab/OneDrive - IIASA/TG_model/model_for_artem/model+DICE_v1/DAE/DICE2016R-091916ap_results.db')
		# pandas.read_sql('select * from loc', con)
		cursor = cond16.cursor()
		cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
		print(cursor.fetchall())
		scc = []
		for t in m.t: ##1e-5 or 1e-8 makes sagnifficant difference! In general, this should be 0.0
		#I guess for numerical purposes dice16 uses 1e-5
		#C is measured in 1e+12 USD; emissions in 1e+9 tonnes C
		#thus, 1000* is needed to make usd per tonnes
			# scc.append(-1000*m.dual.get(m.emmissions_eq[t])/(1e-5+m.dual.get(m.consumption[t])))
			#the following formula is equivalent to the above formula (from DICE16) and similar to the one used in the linked model
			scc.append(-1000*m.dual.get(m.industrialEmmissions[t])/(1e-10+m.dual.get(m.consumption[t])))
		plt.subplot(2, 1, 1)
		plt.plot(scc,  '.-', label = 'DICE16 from PYOMO')
		plt.title('GAMS vs PYOMO: formula for SCC has strange epsilon')
		plt.ylabel('SCC')
		scc_gams = pandas.read_sql_query('SELECT * from %s' % 'scc', cond16)
		plt.plot(scc_gams['value'], '.-', label = 'DICE16 from GAMS')
		plt.legend()
		plt.subplot(2, 1, 2)
		plt.ylabel('Atm Temperature')
		val = []
		for i in m.t:
			val.append(m.TATM[i].value)
		plt.plot(val, label = 'DICE16 from PYOMO')
		tatm_gams = pandas.read_sql_query('SELECT * from %s' % 'TATM', cond16)
		plt.plot(tatm_gams['level'], label = 'DICE16 from GAMS')
		plt.legend()
		plt.show()
		plt.clf()
		#
		#
		# plt.show()


	# p_p(m.TATM)
	# p_p(m.MIU)
