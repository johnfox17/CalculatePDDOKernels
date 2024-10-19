
import cv2
import numpy as np
import fourthOrderPDDODiscretization as PDDO4
import secondOrderPDDODiscretization as PDDO2
import secondOrderPDDODiscretization_4 as PDDO2_4
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

    PDDOMethod2_4 = PDDO2_4.secondOrderPDDODiscretization()
    PDDOMethod2_4.solve()

    PDDOMethod4 = PDDO4.fourthOrderPDDODiscretization()
    PDDOMethod4.solve()

    np.savetxt('/home/doctajfox/Documents/Thesis/CalculatePDDOKernels/data/PDDOKernelMesh1stOrder.csv', PDDOMethod1.PDDOKernelMesh, delimiter=",")
    np.savetxt('/home/doctajfox/Documents/Thesis/CalculatePDDOKernels/data/PDDOKernelMesh2ndOrder.csv', PDDOMethod2.PDDOKernelMesh, delimiter=",")
    np.savetxt('/home/doctajfox/Documents/Thesis/CalculatePDDOKernels/data/PDDOKernelMesh2ndOrder_4.csv', PDDOMethod2_4.PDDOKernelMesh, delimiter=",")
    np.savetxt('/home/doctajfox/Documents/Thesis/CalculatePDDOKernels/data/PDDOKernelMesh4thdOrder.csv', PDDOMethod4.PDDOKernelMesh, delimiter=",")

    np.savetxt('/home/doctajfox/Documents/Thesis/CalculatePDDOKernels/data/kernel1stOrder.csv', PDDOMethod1.kernel, delimiter=",")    
    np.savetxt('/home/doctajfox/Documents/Thesis/CalculatePDDOKernels/data/kernel2ndOrder.csv', PDDOMethod2.kernel, delimiter=",")
    np.savetxt('/home/doctajfox/Documents/Thesis/CalculatePDDOKernels/data/kernel2ndOrder_4.csv', PDDOMethod2_4.kernel, delimiter=",")
    np.savetxt('/home/doctajfox/Documents/Thesis/CalculatePDDOKernels/data/kernel4thdOrder.csv', PDDOMethod4.kernel, delimiter=",")



if __name__ == "__main__":
    main()
