
# coding: utf-8

# # Programming Project

# In this programming project, I will use Python to code a solver for a first-order linear PDE:

# $$
# \frac{\partial u}{\partial t} + v \frac{\partial u}{\partial x} = 0 \quad \mbox{ on }(x,t) \in[0,L]\times[0,T]
# $$
# 
# for a scalar and positive $v\in\mathbb R$ with initial conditions:
# 
# $$
#     u(x,t=0) = h(x),
# $$
# and $v=1.0$.

# I am going to define a function "upwind" for upwind method. This function takes as INPUT:
# - u_old   (real array of length N, discrete function evaluations of u at grid points)
# - T_start (real number, start time of simulation)
# - T_end   (real number, final time of simulation)
# - v       (real number, velocity, positive) 
# - dx   (real number, grid spacing)
# - dt  (real number time step)
# 
# As OUTPUT it gives u_new (solution at new time)

# In[22]:


import numpy as np
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[23]:


def upwind(u_old,T_start,T_end,v,dx,dt):
    #This program works for positive velocity:
    if (v<0):
        print("Velocity is negative. Method will give wrong results")
    
    # Determine number of grid points (degrees of freedom):
    num_dof = len(u_old)      # len() gives the length of the array 'u_old'
    
    # Determine number of time steps from T_start to T_end:
    num_tsteps = int(np.ceil((T_end - T_start)/dt))  # ceil() gives the smallest integer
    
    # Definition on mu (Courant-Friedrich-Lewy number):
    mu = v * (dt / dx)
    
    # Initialize two arrays of length num_dof. copy() is a function used to return a copy of u_old.
    # I don't want to overwrite u_old vector.
    u_tmp = u_old.copy()
    u_new = u_old.copy()
    
    # Loop through all time steps
    for k in range(num_tsteps):
        # for every dt do this
        # Loop through all grid points
        for l in range(num_dof):
            # If you are at the right end of your interval, use this to take into account the PBC
            #if (l==num_dof):
            #    u_new[l] = u_tmp[0] - mu * (u_tmp[0] - u_tmp[l-1])
            #else:
                u_new[l] = u_tmp[l] - mu * (u_tmp[l] - u_tmp[l-1])
                
        # Update the temporary array
        u_tmp = u_new.copy()
    
    # array of u at all grid points at time T_end will be returned by this function
    return u_new


# In[24]:


def lax_wendroff(u_old,T_start,T_end,v,dx,dt):
    #This program works for positive velocity:
    if (v<0):
        print("Velocity is negative. Method will give wrong results")
    
    # Determine number of grid points (degrees of freedom):
    num_dof = len(u_old)      # len() gives the length of the array 'u_old'
    
    # Determine number of time steps from T_start to T_end:
    num_tsteps = int(np.ceil((T_end - T_start)/dt))  # ceil() gives the smallest integer
    
    # Definition on mu (Courant-Friedrich-Lewy number):
    mu = v * (dt / dx)
    
    # Initialize two arrays of length num_dof. copy() is a function used to return a copy of u_old.
    # I don't want to overwrite u_old vector.
    u_tmp = u_old.copy()
    u_new = u_old.copy()
    
    # Loop through all time steps
    for k in range(num_tsteps):
        # for every dt do this
        # Loop through all grid points
        for l in range(num_dof-1):
            # If you are at the right end of your interval, use this to take into account the PBC
            if (l==num_dof-1): 
                u_new[l] =  mu/2.0 * ( 1 + mu ) * u_tmp[l-1] + (1 - mu**2) * u_tmp[l] - mu/2.0 * (1 - mu) * u_tmp[0]
            else:
                u_new[l] =  mu/2.0 * ( 1 + mu ) * u_tmp[l-1] + (1 - mu**2) * u_tmp[l] - mu/2.0 * (1 - mu) * u_tmp[l+1]
                
        # Update the temporary array
        u_tmp = u_new.copy()
    
    # array of u at all grid points at time T_end will be returned by this function
    return u_new


# # EXERCISE 1

# Use the solver to solve the problem with data
# T = 1, L = 1.5, v(x,t) =1, g(x,t)=0, h(x)=exp(-10(4x-1)^2)

# In[25]:


T_start = 0.0
T_2 = 0.25
T_3 = 0.50
T_end   = 1.0

L = 1.5
v = 1.0

# Define the initial condition
def Init_Cond(x,t):
    return np.exp(-10 * ( 4*(x-t) - 1)**2 )

# Number of grid points
N = 200

# Define array of all grid points
x = np.linspace(0,L,N)

# I use the parameter N to determine the grid spacing dx
dx = L/float(N)

# I set the time step
dt = 0.005

# Initialize two arrays: one for the analytical solution and one for the numerical solution
# The first dimension gives the number of grid points and the second is 2: 
# one for u (analytical and numerical) at initial time t = 0.0 and one for u at time t=1.0

u_anal   = np.zeros((N,4))
u_upwind = np.zeros((N,4))
diff = np.zeros((N,4))   # function of the difference between numerical and analitical solutions

# Evaluate the initial condition at time t=0.0 and save the result in the first column of the arrays:
u_anal[:,0]   = Init_Cond(x,T_start)
u_upwind[:,0] = Init_Cond(x,T_start)
diff[:,0] = u_upwind[:,0] - u_anal[:,0]

# The analytical solution at T_end is saved to the second column
u_anal[:,1] = np.exp(-10*(4*(x-0.25)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_upwind[:,1] = upwind(u_upwind[:,0],T_start,T_2,v,dx,dt)

# Difference at 0.25
diff[:,1] = u_upwind[:,1] - u_anal[:,1] 

# The analytical solution at T_end is saved to the second column
u_anal[:,2] = np.exp(-10*(4*(x-0.50)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_upwind[:,2] = upwind(u_upwind[:,0],T_start,T_3,v,dx,dt)

# Difference at 0.50
diff[:,2] = u_upwind[:,2] - u_anal[:,2] 

# The analytical solution at T_end is saved to the second column
u_anal[:,3] = np.exp(-10*(4*(x-1)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_upwind[:,3] = upwind(u_upwind[:,0],T_start,T_end,v,dx,dt)

# Difference at 1.0
diff[:,3] = u_upwind[:,3] - u_anal[:,3] 


# We use Simpson's rule to calculate the integral with scipy.integrate.simps command

# In[26]:


from scipy.integrate import simps

# Let us define the function to integrate:
integrand = []
ans = []
for i in range(4):
    integrand.append([x*x for x in diff[:,i]])
    ans.append(simps(integrand[i],x))    
print(ans[1:4])


# We notice that the error is increasing. So we are expecting that the difference between 
# analytical and numerical solutions calculated with upwind method is stronger for increasing time. 

# In[27]:


# Plot
fig = plt.figure(1)
fig.subplots_adjust(hspace=0.8, wspace=0.8)
fig.set_figwidth(2*fig.get_figwidth())
fig.set_figheight(1*fig.get_figheight())

plt.subplot(221)
plt.plot(x,u_anal[:,0],'r',label="analytical")
plt.plot(x,u_upwind[:,0],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,0.0)")
plt.title("u at time $t=0.0$")

plt.subplot(222)
plt.plot(x,u_anal[:,1],'r',label="analytical")
plt.plot(x,u_upwind[:,1],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.25$")

plt.subplot(223)
plt.plot(x,u_anal[:,2],'r',label="analytical")
plt.plot(x,u_upwind[:,2],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.50$")

plt.subplot(224)
plt.plot(x,u_anal[:,3],'r',label="analytical")
plt.plot(x,u_upwind[:,3],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=1.0$")

plt.legend()
plt.show()


# As we calculated in the error, we can see that when time t=1 the difference between analytical
# and numerical is more relevant. 

# # LAX WENDROFF

# In[28]:


T_start = 0.0
T_2 = 0.25
T_3 = 0.50
T_end   = 1.0

L = 1.5
v = 1.0

# Define the initial condition
def Init_Cond(x,t):
    return np.exp(-10*(4*x-1)**2)

# Number of grid points
N = 200

# Define array of all grid points
x = np.linspace(0,L,N)

# I use the parameter N to determine the grid spacing dx
dx = L/float(N)

# I set the time step
dt = 0.005

# Initialize two arrays: one for the analytical solution and one for the numerical solution
# The first dimension gives the number of grid points and the second is 2: 
# one for u (analytical and numerical) at initial time t = 0.0 and one for u at time t=1.0

u_anal   = np.zeros((N,4))
u_lax_wendroff = np.zeros((N,4))
diff2 = np.zeros((N,4))

# Evaluate the initial condition at time t=0.0 and save the result in the first column of the arrays:
u_anal[:,0]   = Init_Cond(x,T_start)
u_lax_wendroff[:,0] = Init_Cond(x,T_start)
diff2[:,0] = u_lax_wendroff[:,0] - u_anal[:,0]

# The analytical solution at T_end is saved to the second column
u_anal[:,1] = np.exp(-10*(4*(x-0.25)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_lax_wendroff[:,1] = lax_wendroff(u_lax_wendroff[:,0],T_start,T_2,v,dx,dt)

# difference at 0.25
diff2[:,1] = u_lax_wendroff[:,1] - u_anal[:,1]

# The analytical solution at T_end is saved to the second column
u_anal[:,2] = np.exp(-10*(4*(x-0.50)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_lax_wendroff[:,2] = lax_wendroff(u_lax_wendroff[:,0],T_start,T_3,v,dx,dt)

# Difference at 0.50
diff2[:,2] = u_lax_wendroff[:,2] - u_anal[:,2] 

# The analytical solution at T_end is saved to the second column
u_anal[:,3] = np.exp(-10*(4*(x-1)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_lax_wendroff[:,3] = lax_wendroff(u_lax_wendroff[:,0],T_start,T_end,v,dx,dt)

# Difference at 1.00
diff2[:,3] = u_lax_wendroff[:,3] - u_anal[:,3] 


# In[29]:


from scipy.integrate import simps

# Let us define the function to integrate:
integrand2 = []
ans2 = []
for i in range(4):
    integrand2.append([x*x for x in diff2[:,i]])
    ans2.append(simps(integrand2[i],x))    
print(ans2[1:4])


# We notice that the errors with Lax Wendroff method is much smaller than Upwind method.
# Therefore we expect that the analytical and numerical solutions become closer in the graph.
# The error is increasing with time but less than in the upwind case.

# In[30]:


# Plot
fig = plt.figure(1)
fig.subplots_adjust(hspace=0.8, wspace=0.8)
fig.set_figwidth(2*fig.get_figwidth())
fig.set_figheight(1*fig.get_figheight())

plt.subplot(221)
plt.plot(x,u_anal[:,0],'r',label="analytical")
plt.plot(x,u_lax_wendroff[:,0],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,0.0)")
plt.title("u at time $t=0.0$")

plt.subplot(222)
plt.plot(x,u_anal[:,1],'r',label="analytical")
plt.plot(x,u_lax_wendroff[:,1],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.25$")

plt.subplot(223)
plt.plot(x,u_anal[:,2],'r',label="analytical")
plt.plot(x,u_lax_wendroff[:,2],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.50$")

plt.subplot(224)
plt.plot(x,u_anal[:,3],'r',label="analytical")
plt.plot(x,u_lax_wendroff[:,3],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=1.0$")

plt.legend()
plt.show()



# We observe that analytical and numerical solutions are on the top of each other.

# # EXERCISE 2

# In[31]:


# This is for the case B of exercise 2.

T_start = 0.0
T_2 = 0.25
T_3 = 0.50
T_end   = 1.0

L = 1.5
v = 1.0

# Define the initial condition

# Number of grid points
N = 200

# Define array of all grid points
x = np.linspace(0,L,N)

def Init_Cond_2b(x,t):
    u = []
    for elem in x:
        # print(x)
        if ( elem >= t + 0.1 and elem <= t + 0.3):
            u.append(1)
        else:
            u.append(0)
    # print(u)
    return u


# I use the parameter N to determine the grid spacing dx
dx = L/float(N)

# I set the time step
dt = 0.005

# Initialize two arrays: one for the analytical solution and one for the numerical solution
# The first dimension gives the number of grid points and the second is 4: 
# one for u (analytical and numerical) at initial time t = 0.0. t=0.25, t=0.5 and one for u at time t=1.0

u_anal   = np.zeros((N,4))
u_upwind = np.zeros((N,4))
diff3 = np.zeros((N,4))

# Evaluate the initial condition at time t=0.0 and save the result in the first column of the arrays:
u_anal[:,0]   = Init_Cond_2b(x,T_start)
u_upwind[:,0] = Init_Cond_2b(x,T_start)
diff3[:,0] = u_upwind[:,0] - u_anal[:,0]

# The analytical solution at T_2 is saved to the second column
u_anal[:,1] = Init_Cond_2b(x,T_2)

# Use the function to solve the equation and save the result in the second column of u_upwind
u_upwind[:,1] = upwind(u_upwind[:,0],T_start,T_2,v,dx,dt)

# Difference at t=0.25
diff3[:,1] = u_upwind[:,1] - u_anal[:,1]

# The analytical solution at T_2 is saved to the second column
u_anal[:,2] = Init_Cond_2b(x,T_3)

# Use the function to solve the equation and save the result in the third column of u_upwind
u_upwind[:,2] = upwind(u_upwind[:,0],T_start,T_3,v,dx,dt)

# Difference at t=0.50
diff3[:,2] = u_upwind[:,2] - u_anal[:,2]

# The analytical solution at T_2 is saved to the second column
u_anal[:,3] = Init_Cond_2b(x,T_end)

# Use the function to solve the equation and save the result in the fourth column of u_upwind
u_upwind[:,3] = upwind(u_upwind[:,0],T_start,T_end,v,dx,dt)

# Difference at t=1.0
diff3[:,3] = u_upwind[:,3] - u_anal[:,3]


# In[32]:


from scipy.integrate import simps

# Let us define the function to integrate:
integrand3 = []
ans3 = []
for i in range(4):
    integrand3.append([x*x for x in diff3[:,i]])
    ans3.append(simps(integrand3[i],x))    
print(ans3[0:4])


# In[33]:


# Plot
fig = plt.figure(1)
fig.subplots_adjust(hspace=0.8, wspace=0.8)
fig.set_figwidth(2*fig.get_figwidth())
fig.set_figheight(1*fig.get_figheight())

plt.subplot(221)
plt.plot(x,u_anal[:,0],'r',label="analytical")
plt.plot(x,u_upwind[:,0],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,0.0)")
plt.title("u at time $t=0.0$")

plt.subplot(222)
plt.plot(x,u_anal[:,1],'r',label="analytical")
plt.plot(x,u_upwind[:,1],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.25$")

plt.subplot(223)
plt.plot(x,u_anal[:,2],'r',label="analytical")
plt.plot(x,u_upwind[:,2],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.50$")

plt.subplot(224)
plt.plot(x,u_anal[:,3],'r',label="analytical")
plt.plot(x,u_upwind[:,3],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=1.0$")

plt.legend()
plt.show()


# # LAX WENDROFF

# In[34]:


# This is for the case B of exercise 2.

T_start = 0.0
T_2 = 0.25
T_3 = 0.5
T_end   = 1.0

L = 1.5
v = 1.0

# Define the initial condition

# Number of grid points
N = 200

# Define array of all grid points
x = np.linspace(0,L,N)

def Init_Cond_2b(x,t):
    u = []
    for elem in x:
        # print(x)
        if ( elem >= t + 0.1 and elem <= t + 0.3):
            u.append(1)
        else:
            u.append(0)
    # print(u)
    return u


# I use the parameter N to determine the grid spacing dx
dx = L/float(N)

# I set the time step
dt = 0.005

# Initialize two arrays: one for the analytical solution and one for the numerical solution
# The first dimension gives the number of grid points and the second is 2: 
# one for u (analytical and numerical) at initial time t = 0.0 and one for u at time t=1.0

u_anal   = np.zeros((N,4))
u_upwind = np.zeros((N,4))
diff4 = np.zeros((N,4))

# Evaluate the initial condition at time t=0.0 and save the result in the first column of the arrays:
u_anal[:,0]   = Init_Cond_2b(x,T_start)
u_lax_wendroff[:,0] = Init_Cond_2b(x,T_start)
diff4[:,0] = u_lax_wendroff[:,0] - u_anal[:,0]

# The analytical solution at T_2 is saved to the second column
u_anal[:,1] = Init_Cond_2b(x,T_2)

# Use the function to solve the equation and save the result in the second column of u_upwind
u_lax_wendroff[:,1] = lax_wendroff(u_lax_wendroff[:,0],T_start,T_2,v,dx,dt)

# difference at 0.25
diff4[:,1] = u_lax_wendroff[:,1] - u_anal[:,1]

# The analytical solution at T_end is saved to the second column
u_anal[:,2] = Init_Cond_2b(x,T_3)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_lax_wendroff[:,2] = lax_wendroff(u_lax_wendroff[:,0],T_start,T_3,v,dx,dt)

# Difference at 0.50
diff4[:,2] = u_lax_wendroff[:,2] - u_anal[:,2] 

# The analytical solution at T_end is saved to the second column
u_anal[:,3] = Init_Cond_2b(x,T_end)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_lax_wendroff[:,3] = lax_wendroff(u_lax_wendroff[:,0],T_start,T_end,v,dx,dt)

# Difference at 1.00
diff4[:,3] = u_lax_wendroff[:,3] - u_anal[:,3] 


# In[35]:


from scipy.integrate import simps

# Let us define the function to integrate:
integrand4 = []
ans4 = []
for i in range(4):
    integrand4.append([x*x for x in diff2[:,i]])
    ans4.append(simps(integrand2[i],x))    
print(ans4[0:4])


# We can see that also in the second exercise the error is smaller using Lax Wendroff method.

# In[36]:


# Plot
fig = plt.figure(1)
fig.subplots_adjust(hspace=0.8, wspace=0.8)
fig.set_figwidth(2*fig.get_figwidth())
fig.set_figheight(1*fig.get_figheight())

plt.subplot(221)
plt.plot(x,u_anal[:,0],'r',label="analytical")
plt.plot(x,u_lax_wendroff[:,0],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,0.0)")
plt.title("u at time $t=0.0$")

plt.subplot(222)
plt.plot(x,u_anal[:,1],'r',label="analytical")
plt.plot(x,u_lax_wendroff[:,1],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.25$")

plt.subplot(223)
plt.plot(x,u_anal[:,2],'r',label="analytical")
plt.plot(x,u_lax_wendroff[:,2],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.50$")

plt.subplot(224)
plt.plot(x,u_anal[:,3],'r',label="analytical")
plt.plot(x,u_lax_wendroff[:,3],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=1.0$")

plt.legend()
plt.show()


# We notice that the numerical solution is closer to the analytic solution than in the previous case.

# # TEST THE CONVERGENCE

# Now we want to study the time convergence for the Upwind method. We are going to use 2 different time steps.

# In[37]:


T_start = 0.0
T_2 = 0.25
T_3 = 0.50
T_end   = 1.0

L = 1.5
v = 1.0

# Define the initial condition
def Init_Cond(x,t):
    return np.exp(-10 * ( 4*(x-t) - 1)**2 )

# Number of grid points
N = 200

# Define array of all grid points
x = np.linspace(0,L,N)

# I use the parameter N to determine the grid spacing dx
dx = L/float(N)

# I set a smaller time step
dt = 0.0005

# Initialize two arrays: one for the analytical solution and one for the numerical solution
# The first dimension gives the number of grid points and the second is 2: 
# one for u (analytical and numerical) at initial time t = 0.0 and one for u at time t=1.0

u_anal   = np.zeros((N,4))
u_upwind = np.zeros((N,4))
diff5 = np.zeros((N,4))   # function of the difference between numerical and analitical solutions

# Evaluate the initial condition at time t=0.0 and save the result in the first column of the arrays:
u_anal[:,0]   = Init_Cond(x,T_start)
u_upwind[:,0] = Init_Cond(x,T_start)
diff5[:,0] = u_upwind[:,0] - u_anal[:,0]

# The analytical solution at T_end is saved to the second column
u_anal[:,1] = np.exp(-10*(4*(x-0.25)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_upwind[:,1] = upwind(u_upwind[:,0],T_start,T_2,v,dx,dt)

# Difference at 0.25
diff5[:,1] = u_upwind[:,1] - u_anal[:,1] 

# The analytical solution at T_end is saved to the second column
u_anal[:,2] = np.exp(-10*(4*(x-0.50)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_upwind[:,2] = upwind(u_upwind[:,0],T_start,T_3,v,dx,dt)

# Difference at 0.50
diff5[:,2] = u_upwind[:,2] - u_anal[:,2] 

# The analytical solution at T_end is saved to the second column
u_anal[:,3] = np.exp(-10*(4*(x-1)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_upwind[:,3] = upwind(u_upwind[:,0],T_start,T_end,v,dx,dt)

# Difference at 1.0
diff5[:,3] = u_upwind[:,3] - u_anal[:,3] 




# In[38]:


integrand5 =[]
ans5 = []
for i in range(4):
    integrand5.append([x*x for x in diff5[:,i]])
    ans5.append(simps(integrand5[i],x))    
print(ans5[0:4])

# Plot
fig = plt.figure(1)
fig.subplots_adjust(hspace=0.8, wspace=0.8)
fig.set_figwidth(2*fig.get_figwidth())
fig.set_figheight(1*fig.get_figheight())

plt.subplot(221)
plt.plot(x,u_anal[:,0],'r',label="analytical")
plt.plot(x,u_upwind[:,0],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,0.0)")
plt.title("u at time $t=0.0$")

plt.subplot(222)
plt.plot(x,u_anal[:,1],'r',label="analytical")
plt.plot(x,u_upwind[:,1],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.25$")

plt.subplot(223)
plt.plot(x,u_anal[:,2],'r',label="analytical")
plt.plot(x,u_upwind[:,2],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.50$")

plt.subplot(224)
plt.plot(x,u_anal[:,3],'r',label="analytical")
plt.plot(x,u_upwind[:,3],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=1.0$")

plt.legend()
plt.show()

# With a smaller time step the error increases. Let's choose a finer grid:
# In[39]:


T_start = 0.0
T_2 = 0.25
T_3 = 0.50
T_end   = 1.0

L = 1.5
v = 1.0

# Define the initial condition
def Init_Cond(x,t):
    return np.exp(-10 * ( 4*(x-t) - 1)**2 )

# Number of grid points
N = 400

# Define array of all grid points
x = np.linspace(0,L,N)

# I use the parameter N to determine the grid spacing dx
dx = L/float(N)

# I set the time step
dt = 0.005

# Initialize two arrays: one for the analytical solution and one for the numerical solution
# The first dimension gives the number of grid points and the second is 2: 
# one for u (analytical and numerical) at initial time t = 0.0 and one for u at time t=1.0

u_anal   = np.zeros((N,4))
u_upwind = np.zeros((N,4))
diff6 = np.zeros((N,4))   # function of the difference between numerical and analitical solutions

# Evaluate the initial condition at time t=0.0 and save the result in the first column of the arrays:
u_anal[:,0]   = Init_Cond(x,T_start)
u_upwind[:,0] = Init_Cond(x,T_start)
diff6[:,0] = u_upwind[:,0] - u_anal[:,0]

# The analytical solution at T_end is saved to the second column
u_anal[:,1] = np.exp(-10*(4*(x-0.25)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_upwind[:,1] = upwind(u_upwind[:,0],T_start,T_2,v,dx,dt)

# Difference at 0.25
diff6[:,1] = u_upwind[:,1] - u_anal[:,1] 

# The analytical solution at T_end is saved to the second column
u_anal[:,2] = np.exp(-10*(4*(x-0.50)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_upwind[:,2] = upwind(u_upwind[:,0],T_start,T_3,v,dx,dt)

# Difference at 0.50
diff6[:,2] = u_upwind[:,2] - u_anal[:,2] 

# The analytical solution at T_end is saved to the second column
u_anal[:,3] = np.exp(-10*(4*(x-1)-1)**2)  # This is from pen and paper!

# Use the function to solve the equation and save the result in the second column of u_upwind
u_upwind[:,3] = upwind(u_upwind[:,0],T_start,T_end,v,dx,dt)

# Difference at 1.0
diff6[:,3] = u_upwind[:,3] - u_anal[:,3] 


# In[40]:



integrand6 =[]
ans6 = []
for i in range(4):
    integrand6.append([x*x for x in diff6[:,i]])
    ans6.append(simps(integrand6[i],x))    
print(ans6[0:4])

# Plot
fig = plt.figure(1)
fig.subplots_adjust(hspace=0.8, wspace=0.8)
fig.set_figwidth(2*fig.get_figwidth())
fig.set_figheight(1*fig.get_figheight())

plt.subplot(221)
plt.plot(x,u_anal[:,0],'r',label="analytical")
plt.loglog(x,u_upwind[:,0],basex = np.e, basey = np.e)
#plt.plot(x,u_upwind[:,0],'k--',label="numerical")
#plt.xlabel("x")
#plt.ylabel("u(x,0.0)")
plt.title("u at time $t=0.0$")

plt.subplot(222)
plt.plot(x,u_anal[:,1],'r',label="analytical")
plt.loglog(x,u_upwind[:,1],basex = np.e, basey = np.e)
#plt.plot(x,u_upwind[:,1],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.25$")

plt.subplot(223)
plt.plot(x,u_anal[:,2],'r',label="analytical")
plt.loglog(x,u_upwind[:,2],basex = np.e, basey = np.e)
#plt.plot(x,u_upwind[:,2],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=0.50$")

plt.subplot(224)
plt.plot(x,u_anal[:,3],'r',label="analytical")
plt.loglog(x,u_upwind[:,3],basex = np.e, basey = np.e)
#plt.plot(x,u_upwind[:,3],'k--',label="numerical")
plt.xlabel("x")
plt.ylabel("u(x,1.0)")
plt.title("u at time $t=1.0$")

plt.legend()
plt.show()


# With a smaller grid the error increases. My results don't match the theoretical observation.
