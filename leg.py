import odrive.core
import time
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# For symbolic processing
import sympy
from sympy import symbols
from sympy import sin, cos, asin, acos,  atan
from sympy.utilities.lambdify import lambdify
from sympy import Matrix
from sympy.solvers import solve
from scipy import linalg

#%matplotlib qt5 #opens new window for plotting things on the same figure

class Leg:
    """
    This is our first class in class :)

    We will define a leg class to interface with the leg and standardize 
    the kind of operations we want to perform

    """
    global l1, l2, l_base, theta0_sym, theta1_sym, alpha0_sym, alpha1_sym, encoder2angle
    #### Variables outside the init function are constants of the class
    # leg geometry
    l1 = 7  # NEED TO UPDATE units of cm
    l2 = 14  # NEED TO UPDATE units of cm
    l_base = 7.7  # NEED TO UPDATE units of cm
    theta0_sym, theta1_sym, alpha0_sym, alpha1_sym, = symbols('theta0_sym theta1_sym alpha0_sym alpha1_sym' , real=True)

    # motor controller parameters
    encoder2angle = 2048 * 4

    ### Methods
    # Classes are initiated with a constructor that can take in initial parameters. At
    # a minimum it takes in a copy of itself (python... weird). The constructor
    # is one place we can define and initialize class variables

    def __init__(self, simulate = True):
        """
        This is the constructor for the leg class. Whenever you make a new leg
        this code will be called. We can optionally make a leg that will simulate
        the computations without needing to be connected to the ODrive
        """

        self.simulate = simulate #simulate

            

        # make the option to code without having the odrive connected
        if self.simulate == False:
            self.drv = self.connect_to_controller()
            self.m0 = self.drv.motor0  # easier handles to the motor commands
            self.m1 = self.drv.motor1

            # current positions
            m0_pos, m1_pos = self.get_joint_pos()
            self.joint_0_pos = m0_pos
            self.joint_1_pos = m1_pos

        else:
            self.drv = None
            self.joint_0_pos = 2
            self.joint_1_pos = 1.4

        # home angles
        self.joint_0_home = 0
        self.joint_1_home = 0
        
        self.home = (0,0)

        

        # We will compute the jacobian and inverse just once in the class initialization.
        # This will be done symbolically so that we can use the inverse without having
        # to recompute it every time
        self.J = self.compute_jacobian()
        #print('jacobian done')
        #self.J_inv = self.J.pinv()
        #print('inverse done')

    def connect_to_controller(self):
        """
        Connects to the motor controller
        """
        drv = odrive.core.find_any(consider_usb=True, consider_serial=False)

        if drv is None:
            print('No controller found')
        else:
            print('Connected!')
        return drv

    ###
    ### Motion functions
    ###
    def get_joint_pos(self):
        """
        Get the current joint positions and store them in self.joint_0_pos and self.joint_1_pos in degrees.
        Also, return these positions using the return statement to terminate the function
        """
        # if simulating exit function
        if self.simulate == True:
            return (self.joint_0_pos, self.joint_1_pos)
        else: # your code here
            self.joint_0_pos = self.m0.encoder.pll_pos/encoder2angle*(2*math.pi) +math.pi/2 #degree conversion
            self.joint_1_pos = self.m1.encoder.pll_pos/encoder2angle*(2*math.pi) +math.pi/2

        return (self.joint_0_pos, self.joint_1_pos)

    def set_home(self):
        """
        This function updates the home locations of the motors so that 
        all move commands we execute are relative to this location. 
        """
        # if simulating exit function
        if self.simulate == True:
            return
        else: # Your code here
            self.home = (self.m0.encoder.pll_pos/encoder2angle*(2*math.pi)+math.pi/2,
                         self.m1.encoder.pll_pos/encoder2angle*(2*math.pi) +math.pi/2) #angle
            


    def set_joint_pos(self, theta0, theta1, vel0=0, vel1=0, curr0=0, curr1=0):
        """
        Set the joint positions in units of deg, and with respect to the joint homes.
        We have the option of passing the velocity and current feedforward terms.
        """
        # if simulating exit function
        if self.simulate == True:
            self.joint_0_pos = theta0
            self.joint_1_pos = theta1
        else: # Your code here
            self.get_joint_pos()
            self.m0.set_pos_setpoint((theta0-self.home[0])*encoder2angle/(2*math.pi)-math.pi/2 ,vel0,curr0) #encoder value = angle*(-4000)/math.pi determiend in first homework
            self.m1.set_pos_setpoint((theta1-self.home[1])*encoder2angle/(2*math.pi)-math.pi/2 ,vel1,curr1)
            
            

    def move_home(self):
        """
        Move the motors to the home position
        """
        # if simulating exit function
        if self.simulate == True:
            return
        else: # Your code here
            self.m0.set_pos_setpoint(self.home[0])
            self.m1.set_pos_setpoint(self.home[1])
        

    def set_foot_pos(self, x, y):
        """
        Move the foot to position x, y. This function will call the inverse kinematics 
        solver and then call set_joint_pos with the appropriate angles
        """
        # if simulating exit function
        if self.simulate == True:
            (theta_0, theta_1) = self.inverse_kinematics(x,y)
            self.set_joint_pos(theta_0,theta_1) 
            return (theta_0,theta_1)
        else:
            (theta_0, theta_1) = self.inverse_kinematics(x,y)
            self.set_joint_pos(theta_0,theta_1) 
        #
        # Your code here
        #

    def move_trajectory(self, tt, xx, yy):
        """
        Move the foot over a cyclic trajectory to positions xx, yy in time tt. 
        This will repeatedly call the set_foot_pos function to the new foot 
        location specified by the trajectory.
        """
        # if simulating exit function
        
        
        if self.simulate == True:
            tra_theta0, tra_theta1 = [], []
            tra_alpha0, tra_alpha1 = [], []
            for i in range(tt):
				# Computer theta's for trajectory and store them
                (theta_0, theta_1) = self.set_foot_pos(xx[i],yy[i])
                tra_theta0.append(theta_0)
                tra_theta1.append(theta_1)
                (alpha_0, alpha_1) = self.compute_internal_angles(theta_0, theta_1)
                tra_alpha0.append(alpha_0)
                tra_alpha1.append(alpha_1)     
            np.savetxt('thetas', (tra_theta0, tra_theta1))
            return (tra_theta0, tra_theta1, tra_alpha0, tra_alpha1)    
        else:
            for i in range(tt):
                self.set_foot_pos(xx[i],yy[i])
                          
        #
        # Your code here
        #


    ###
    ### Leg geometry functions
    ###
    def compute_internal_angles(self, theta_0, theta_1):
        """
        Return the internal angles of the robot leg 
        from the current motor angles
        """
        l1 = 7  # NEED TO UPDATE units of cm
        l2 = 14  # NEED TO UPDATE units of cm
        l_base = 7.7  # NEED TO UPDATE units of cm
        
        BE=(l_base**2+l1**2-2*l_base*l1*sympy.cos(math.pi-theta_0))**0.5
        ABE_angle=asin(l1*sympy.sin(math.pi-theta_0)/BE)
        AEB_angle=theta_0-ABE_angle
        CBE_angle=theta_1-ABE_angle
        CE=(2*l1**2+l_base**2-2*l_base*l1*sympy.cos(math.pi-theta_0) - 2*l1*BE*sympy.cos(CBE_angle))**0.5
        BCE_angle=asin(BE*sympy.sin(CBE_angle)/CE)
        BEC_angle=math.pi-BCE_angle-CBE_angle
        ECD_angle=acos((CE/2)/l2)
        alpha_1=BCE_angle+ECD_angle-(math.pi-theta_1)
        alpha_0=2*math.pi-(math.pi-theta_0)-AEB_angle-BEC_angle-ECD_angle
        
 
        return (alpha_0, alpha_1)
        
        
    
       
     
       
        return (alpha_0, alpha_1)

    def compute_jacobian(self):
        """
        This function implements the symbolic solution to the Jacobian.
        """

        # initiate the symbolic variables
        #theta0_sym, theta1_sym, alpha0_sym, alpha1_sym, = symbols(
         #   'theta0_sym theta1_sym alpha0_sym alpha1_sym' , real=True)

        # Your code here that solves J as a matrix
        [alpha0_sym,alpha1_sym] = self.compute_internal_angles(theta0_sym, theta1_sym)
        
        x = l_base/2 + l1*cos(theta0_sym) + l2*cos(alpha0_sym)
        y=l1*sin(theta0_sym)+l2*sin(alpha0_sym)
        
        FK=Matrix([[x],[y]])
        J=FK.jacobian([theta0_sym,theta1_sym])
        
        #
        # Your code here that solves J as a matrix
        #

        return J
     

    def inverse_kinematics(self, x, y):
        """
        This function will compute the required theta_0 and theta_1 angles to position the 
        foot to the point x, y. We will use an iterative solver to determine the angles.
        """
        error = Matrix([1e5, 1e5])
        (theta_0,theta_1) = self.get_joint_pos()  
        
        #(theta_0,theta_1) = (3.14/2, 3.14/3)  
        


        while error.norm() > 1e-2: 
            (alpha_0,alpha_1) = self.compute_internal_angles(theta_0,theta_1)

            

            current_x = l_base/2 + l1*cos(theta_0) + l2*cos(alpha_0)
            
            current_y = l1*sin(theta_0) + l2*sin(alpha_0)

            error = sympy.N(Matrix([x-current_x, y-current_y]))
            
            #error = Matrix([x-current_x, y-current_y])
            print (error.norm())
    
            J1 = self.J.subs([(theta0_sym, theta_0), (theta1_sym, theta_1), (alpha0_sym,alpha_0), (alpha1_sym, alpha_1)])
            
            J1 = sympy.N(J1)
            J1_inv = J1.pinv()
            
            #increment = 0.001 * J1_inv * error
            #theta_0 = theta_0 + increment[0]
            #theta_1 = theta_1 + increment[1]
        
            
            
            d_theta = J1_inv@error*0.05 #how much to move theta
            theta_0 = theta_0 + d_theta[0] #update theta 
            theta_1 = theta_1 + d_theta[1]
           
            
         
            theta_0=theta_0%(2*math.pi)
            if theta_0%(2*math.pi)>2/3*math.pi:
                theta_0=(theta_0+3*math.pi)%(2*math.pi)
            if theta_0%(2*math.pi)>1/2*math.pi:
                theta_0=(theta_0+3*math.pi)%(2*math.pi)
            
            

            

        return (theta_0, theta_1) 

    ###
    ### Visualization functions
    ###
    def sim_draw_leg(l0, l1, l2, theta1, theta2, alpha1, alpha2, ax=False):
        """
        This function takes in the four angles of the leg and draws
        the configuration
        """

        link1, link2, width = l1, l2, l0
        cor_range = l1 + l2
    
        if ax == False:
            ax = plt.gca()
            ax.cla()

        ax.plot(-width / 2, 0, 'ok')
        ax.plot(width / 2, 0, 'ok')

        ax.plot([-width / 2, 0], [0, 0], 'k')
        ax.plot([width / 2, 0], [0, 0], 'k')

        ax.plot(-width / 2 + np.array([0, link1 * cos(theta1)]), [0, link1 * sin(theta1)], 'k')
        ax.plot(width / 2 + np.array([0, link1 * cos(theta2)]), [0, link1 * sin(theta2)], 'k')

        ax.plot(-width / 2 + link1 * cos(theta1) + np.array([0, link2 * cos(alpha1)]), \
                link1 * sin(theta1) + np.array([0, link2 * sin(alpha1)]), 'k');
        ax.plot(width / 2 + link1 * cos(theta2) + np.array([0, link2 * cos(alpha2)]), \
                np.array(link1 * sin(theta2) + np.array([0, link2 * sin(alpha2)])), 'k');
            
        curr_x = float(width / 2 + link1 * cos(theta2) + link2 * cos(alpha2))
        curr_y = float(np.array(link1 * sin(theta2) + link2 * sin(alpha2)))

        ax.plot(curr_x, curr_y, 'ro');

        ax.axis([-cor_range, cor_range, -l1, cor_range])
        ax.invert_yaxis()

 #   def draw_leg(self,ax=False):
 #       """
  #      This function takes in the four angles of the leg and draws
   #     the configuration
    #    """
     #   

#        theta1, theta2 = self.joint_0_pos, self.joint_1_pos
 #       link1, link2, width = l1, l2, l_base
#
 #       (alpha1, alpha2) = self.compute_internal_angles(theta1,theta2)
#
 #       def pol2cart(rho, phi):
  #          x = rho * np.cos(phi)
   #         y = rho * np.sin(phi)
    #        return (x, y)
#
 #       if ax == False:
  #          
   #         ax = plt.gca()
    #        ax.cla()
#
 #       ax.plot(-width / 2, 0, 'ok')
  #      ax.plot(width / 2, 0, 'ok')
#
 #       ax.plot([-width / 2, 0], [0, 0], 'k')
  #      ax.plot([width / 2, 0], [0, 0], 'k')
#
 #       ax.plot(-width / 2 + np.array([0, link1 * cos(theta1)]), [0, link1 * sin(theta1)], 'k')
  #      ax.plot(width / 2 + np.array([0, link1 * cos(theta2)]), [0, link1 * sin(theta2)], 'k')
#
 #       ax.plot(-width / 2 + link1 * cos(theta1) + np.array([0, link2 * cos(alpha1)]), \
  #              link1 * sin(theta1) + np.array([0, link2 * sin(alpha1)]), 'k');
   #     ax.plot(width / 2 + link1 * cos(theta2) + np.array([0, link2 * cos(alpha2)]), \
    #            np.array(link1 * sin(theta2) + np.array([0, link2 * sin(alpha2)])), 'k');
#
 #       ax.plot(width / 2 + link1 * cos(theta2) + link2 * cos(alpha2), \
  #              np.array(link1 * sin(theta2) + link2 * sin(alpha2)), 'ro');
#
 #       ax.axis([-(l1+l2), (l1+l2), -(l1+l2), (l1+l2)])
  #      ax.invert_yaxis()
#
        #plt.draw()
