import PDDOConstants
import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import solve

class fourthOrderPDDODiscretization:
    def __init__(self):
        self.dx = 1/PDDOConstants.NX
        self.dy = 1/PDDOConstants.NY
        self.l1 = PDDOConstants.L1
        self.l2 = PDDOConstants.L2
        self.deltaX = PDDOConstants.HORIZON4*self.dx
        self.deltaY = PDDOConstants.HORIZON4*self.dy
        self.bVec40 = PDDOConstants.BVEC40 
        self.bVec04 = PDDOConstants.BVEC04
        self.horizon = PDDOConstants.HORIZON4
        self.kernelDim = PDDOConstants.KERNELDIM4
    
    def createPDDOKernelMesh(self):
        indexing = 'xy'
        #xCoords = np.linspace(self.dx/2,(self.horizon + 1)*self.dx , int(self.horizon*2 + 1))
        #yCoords = np.linspace(self.dy/2,(self.horizon + 1)*self.dy, int(self.horizon*2 + 1))
        
        xCoords = np.arange(self.dx/2, (self.horizon*2 + 1)*self.dx, self.dx)
        yCoords = np.arange(self.dy/2, (self.horizon*2 + 1)*self.dy, self.dy)
        xCoords, yCoords = np.meshgrid(xCoords, yCoords, indexing=indexing)
        xCoords = xCoords.reshape(-1, 1)
        yCoords = yCoords.reshape(-1, 1)
        self.PDDOKernelMesh = np.array([xCoords[:,0], yCoords[:,0]]).T
    
    def calculateXis(self):
        midPDDONodeCoords = self.PDDOKernelMesh[int((len(self.PDDOKernelMesh)-1)/2),:]
        self.xXis = midPDDONodeCoords[0]-self.PDDOKernelMesh[:,0]
        self.yXis = midPDDONodeCoords[1]-self.PDDOKernelMesh[:,1]
         
    def calculateGPolynomials(self):
        deltaMag = np.sqrt(self.deltaX**2 + self.deltaY**2)
        diffMat = np.zeros([15,15])
        g40 = []
        g04 = []

        for iNode in range(len(self.PDDOKernelMesh)):
            currentXXi = self.xXis[iNode]
            currentYXi = self.yXis[iNode]
            xiMag = np.sqrt(currentXXi**2+currentYXi**2)
            pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, (currentXXi/deltaMag)**2, \
                    (currentYXi/deltaMag)**2, (currentXXi/deltaMag)*(currentYXi/deltaMag), \
                    (currentXXi/deltaMag)**3, (currentYXi/deltaMag)**3, ((currentXXi/deltaMag)**2)*(currentYXi/deltaMag), \
                    (currentXXi/deltaMag)*((currentYXi/deltaMag)**2), (currentXXi/deltaMag)**4, (currentYXi/deltaMag)**4, \
                    ((currentXXi/deltaMag)**3)*(currentYXi/deltaMag), ((currentXXi/deltaMag)**2)*(currentYXi/deltaMag)**2, \
                    (currentXXi/deltaMag)*((currentYXi/deltaMag)**3)])
            weight = np.exp(-4*(xiMag/deltaMag)**2)
            diffMat += weight*np.outer(pList,pList)*self.dx*self.dy
        for iNode in range(len(self.PDDOKernelMesh)):
            currentXXi = self.xXis[iNode]
            currentYXi = self.yXis[iNode]
            xiMag = np.sqrt(currentXXi**2+currentYXi**2)
            pList = np.array([1, currentXXi/deltaMag, currentYXi/deltaMag, (currentXXi/deltaMag)**2, \
                    (currentYXi/deltaMag)**2, (currentXXi/deltaMag)*(currentYXi/deltaMag), \
                    (currentXXi/deltaMag)**3, (currentYXi/deltaMag)**3, ((currentXXi/deltaMag)**2)*(currentYXi/deltaMag), \
                    (currentXXi/deltaMag)*((currentYXi/deltaMag)**2), (currentXXi/deltaMag)**4, (currentYXi/deltaMag)**4, \
                    ((currentXXi/deltaMag)**3)*(currentYXi/deltaMag), ((currentXXi/deltaMag)**2)*(currentYXi/deltaMag)**2, \
                    (currentXXi/deltaMag)*((currentYXi/deltaMag)**3)])
            weight = np.exp(-4*(xiMag/deltaMag)**2)
            g40.append(weight*(np.inner(solve(diffMat,self.bVec40), pList)))
            g04.append(weight*(np.inner(solve(diffMat,self.bVec04), pList)))
        self.g40 = np.array(g40).reshape((self.kernelDim,self.kernelDim))
        self.g04 = np.array(g04).reshape((self.kernelDim,self.kernelDim))
    
    def normalizeGPolynomials(self):
        #self.g40 = np.array(self.g40.flatten()/np.linalg.norm(self.g40.flatten(),1)).reshape((self.kernelDim,self.kernelDim))
        #self.g04 = np.array(self.g04.flatten()/np.linalg.norm(self.g04.flatten(),1)).reshape((self.kernelDim,self.kernelDim))
        self.g40 = self.g40
        self.g04 = self.g04

    def combineGPolynomials(self):
        self.kernel = -(self.g40 + self.g04)

    def createPDDOKernel(self):
        self.calculateXis()
        self.calculateGPolynomials()
        self.normalizeGPolynomials()
        self.combineGPolynomials()

    def solve(self):
        self.createPDDOKernelMesh()
        self.createPDDOKernel()
