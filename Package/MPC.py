import System_Dynamics
import numpy as np
from math import cos, sin

#formulate original state space equation
class ModelPredictiveControl(object):
    def __init__(self, A, B, C, f, v, W3, W4, x0, total):
        self.A=A
        self.B=B
        self.C=C
        self.f=f    # prediction horizon
        self.v=v    # control horizon
        self.W3=W3
        self.W4=W4
        self.total=total

        # State-space dimensions
        n_states = 10  # [x, y, z, vx, vy, vz, ax, ay, az, m]
        n_controls = 2  # [T1, T2]

        self.X = np.zeros((n_states, 1))
        self.U = np.zeros((n_controls, 1))

        self.A = np.zeros((n_states, n_states))
        self.B = np.zeros((n_states, n_controls))

        self.Xnew = np.dot(A, self.X) + np.dot(B, self.U)

    def formLiftedMatrices(self):
        #f is prediction horizon
        #v is control horizon

        f=self.f
        v=self.v
        r=self.r
        n=self.n
        m=self.m
        A=self.A
        B=self.B
        C=self.C
        
        O=np.zeros(shape=(f*r,n))

        #what is r?
        #lifted matrix with C* A^i wrt prediction horizon
        for i in range(f):
            if (i==0):
                powA=A
            else:
                powA=np.matmul(powA,A)
            O[i*r:(i+1)*r,:]=np.matmul(C,powA)
            #dimensions of blocks of A are rows of CA which is rxn dim, 
            #lifted matrix of O at row index i*r to i+1*r
        
        #lifted matrix M for U
        M=np.zeros(shape=(f*r,v*m))
        #inputs: control horizon, dim of u?

        for i in range(f):
            if (i<v):
                for j in range(i+1):
                    if (j==0):
                        powA = np.eye(n,n)
                    else:
                        powA=np.matmul(powA,A)
                    M[i*r:(i+1)*r,(i-j)*m:(i-j+1)*m]=np.matmul(C,np.matmul(powA,B))
            else:
                    for j in range(v):
                        # here we form the last entry
                        if j==0:
                            sumLast=np.zeros(shape=(n,n))
                            for s in range(i-v+2):
                                if (s == 0):
                                    powA=np.eye(n,n)
                                else:
                                    powA=np.matmul(powA,A)
                                sumLast=sumLast+powA
                            M[i*r:(i+1)*r,(v-1)*m:(v)*m]=np.matmul(C,np.matmul(sumLast,B))
                        else:
                            powA=np.matmul(powA,A)
                            M[i*r:(i+1)*r,(v-1-j)*m:(v-j)*m]=np.matmul(C,np.matmul(powA,B))
            
            
            tmp1=np.matmul(M.T,np.matmul(self.W4,M))
            tmp2=np.linalg.inv(tmp1+self.W3)
            gainMatrix=np.matmul(tmp2,np.matmul(M.T,self.W4))
            
            
            return O,M,gainMatrix
                    #from control horizon to prediction horizon (where u becomes a constant)
                    #MUCH NEEDED dim analysis

        #v is prediction horizion, m is number of vars in U?

        #rows outputs, columns inputs

    def g(self):
        alpha_1 = np.cross(self.r_vector, self.Thrust_vec) / ( self.Icm[self.clock] * self.n_thrust[0])
        alpha_2 = np.cross(self.r_vector, self.Thrust_vec) / ( self.Icm[self.clock] * self.n_thrust[1])
        alpha_3 = np.cross(self.r_vector, self.Thrust_vec) / ( self.I_axis[self.clock] * self.n_thrust[2])
        return [alpha_1,alpha_2,alpha_3]
    
    def B_matrix(self):
        B_matrix = np.zeros((18,3))
        
        acc_ground = (self.Thrust[self.clock] / self.Mass) * (self.R2U())
        #6th-8th index needs to be numpy arrays of 
        theta1 = self.X[9]  # yaw
        theta2 = self.X[10] # pitch
        theta3 = self.X[11] # roll
        B_matrix[6] = np.array([acc_ground * (cos(theta2) * cos(theta1)), acc_ground * (-cos(theta2) * sin(theta1)), acc_ground * (sin(theta2))])
        B_matrix[7] = np.array([acc_ground * ((-sin(theta3)*sin(theta2)*cos(theta1))+(cos(theta3))*sin(theta1)), acc_ground * ((sin(theta3)*sin(theta2)*cos(theta1))+(cos(theta3)*cos(theta1))), acc_ground * (sin(theta3)*cos(theta2))])
        B_matrix[8] = np.array([acc_ground * ((-cos(theta3)*sin(theta2)*cos(theta1))-(sin(theta3)*sin(theta1))), acc_ground * ((cos(theta3)*sin(theta2)*cos(theta1))-(sin(theta3)*cos(theta1))), acc_ground * (cos(theta3)*cos(theta2))])

        diagonal_elements = self.g()
        B_matrix[15] = np.array([diagonal_elements[0],0,0])
        B_matrix[16] = np.array([0,diagonal_elements[1],0])
        B_matrix[17] = np.array([0,0,diagonal_elements[2]])
        return B_matrix
    
    def control_to_theta(self):
        # TODO: Convert the U vector back into 2 angles for the motors controlling the thrust vector.
        pass