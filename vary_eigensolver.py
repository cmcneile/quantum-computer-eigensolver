#
#  Variational  eigensolver
#


import numpy as np
import math
import cmath
from scipy.optimize import minimize
import sys
from scipy.optimize import basinhopping

from numpy import linalg as LA

infile = "Hstore_4dim.npy"
Hclass = np.load(infile)
print("Matrix read from " , infile )
print("m = " , Hclass )


w, v = LA.eig(Hclass)

print("Classical eigenvalues:")
for xx in w:
   print(xx)


print("")

##sys.exit(0)


#
# Variational 
#


print("Starting the variational analysis")

# https://qiskit.org/textbook/ch-applications/vqe-molecules.html#varforms
# Check whether corect
#
#  U3 *  (1 1)^t
#
def create_2vec(theta_1, phi_1, lamb_1,theta_2, phi_2, lamb_2) :

   vec = np.zeros((4,1), dtype=complex)

   vec[0] = math.cos(theta_1/2.0) - math.sin(theta_1/2.0) *  cmath.exp( lamb_1*1j)
   vec[1] = math.sin(theta_1/2.0) *  cmath.exp( phi_1*1j) + math.cos(theta_1/2.0) * cmath.exp( (lamb_1 +phi_1)*1j) 
            
   vec[2] = math.cos(theta_2/2.0) - math.sin(theta_2/2.0) *  cmath.exp( lamb_2*1j)
   vec[3] = math.sin(theta_2/2.0) *  cmath.exp( phi_2*1j) +  math.cos(theta_2/2.0) * cmath.exp( (lamb_2 +phi_2)*1j) 

   return vec


if 0 :
  print("Test parameterization")
  ans = create_2vec(1.0, 1.0, 1.0,1.0, 1.0, 1.0 ) 
  print(ans)
  nn = np.vdot(ans,ans)
  print("norm = " , nn)


print("Start the optimization") 

#
#  Eigenvalue estimate  (v,H * v) / (v,v)
#  where v is a parameterized vector
#

def  vary_lowest_eigenval(x) :
    theta_1 = x[0]
    phi_1   = x[1]
    lamb_1  = x[2]
    theta_2 = x[3]
    phi_2   = x[4]
    lamb_2  = x[5]

    vv = create_2vec(theta_1, phi_1, lamb_1,theta_2, phi_2, lamb_2 ) 
    Mvv = np.matmul(Hclass, vv)

    ans = np.vdot(vv,Mvv) / np.vdot(vv,vv)
##    ans = np.vdot(vv,Mvv) 

    return np.real(ans)


x0 = np.array([1.3, 0.7, 0.2, 0.5, 0.3 , 0.2])
print("First guess of the parameters = " , x0)

res = minimize(vary_lowest_eigenval, x0, method='nelder-mead',
              options={'xatol': 1e-8, 'disp': True})

##  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html#scipy.optimize.basinhopping
##res = basinhopping(vary_lowest_eigenval, x0)

print("final parameters = " , res.x)

lowest_eig =  vary_lowest_eigenval( res.x )

print("Estimate of lowest eigenvalue  = " , lowest_eig )
