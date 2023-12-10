
import cv2
import numpy as np
import fourthOrderPDDODiscretization as PDDO4
import secondOrderPDDODiscretization as PDDO2
import firstOrderPDDODiscretization as PDDO1

def main():
    lena = cv2.imread('../data/lena.png')
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY)
    noisyLena = cv2.imread('../data/noisyLena.png')
    noisyLena = cv2.cvtColor(noisyLena, cv2.COLOR_BGR2GRAY)

    
    PDDOMethod1 = PDDO1.firstOrderPDDODiscretization()
    PDDOMethod1.solve()

    PDDOMethod2 = PDDO2.secondOrderPDDODiscretization()
    PDDOMethod2.solve()

    PDDOMethod4 = PDDO4.fourthOrderPDDODiscretization()
    PDDOMethod4.solve()

    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\g10.csv', PDDOMethod1.g10, delimiter=",") 
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\g01.csv', PDDOMethod1.g01, delimiter=",")
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\kernel1stOrder.csv', PDDOMethod1.kernel, delimiter=",")
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\PDDOKernelMesh1.csv', PDDOMethod1.PDDOKernelMesh, delimiter=",") 
    
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\g20.csv', PDDOMethod2.g20, delimiter=",")
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\g02.csv', PDDOMethod2.g02, delimiter=",")
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\kernel2ndOrder.csv', PDDOMethod2.kernel, delimiter=",")
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\PDDOKernelMesh2.csv', PDDOMethod2.PDDOKernelMesh, delimiter=",")

    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\g40.csv', PDDOMethod4.g40, delimiter=",")
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\g04.csv', PDDOMethod4.g04, delimiter=",")
    
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\kernel4thdOrder.csv', PDDOMethod4.kernel, delimiter=",")
    np.savetxt('C:\\Users\\docta\\Documents\\Thesis\\CalculatePDDOKernels\\data\\PDDOKernelMesh4.csv', PDDOMethod4.PDDOKernelMesh, delimiter=",")
    #cv2.imshow('Lean vs Noisy Lena',np.concatenate((lena, noisyLena), axis=1))

    #cv2.imshow('PDDO Edge Detection', filterIM1.filteredImage)
    

    #cv2.imshow('Noisy Lena',noisyLena)
    #cv2.waitKey(0)
    #print(np.shape(noisyLena))






    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
