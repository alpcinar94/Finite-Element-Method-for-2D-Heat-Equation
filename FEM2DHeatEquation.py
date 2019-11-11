#ALP CINAR 2019
#UNIVERSITY OF CALIFORNIA, BERKELEY
#ME280A HW8 PROBLEM 5

import numpy as np
import pandas as pd
import openpyxl
import math
from sympy import *
import matplotlib.pyplot as plt

#PLEASE CHANGE THE NUMBER OF ELEMENTS IN THE MESH BY PLAYING WITH "numberOfElements" variable
numberOfElements = 256
nodesInElement = 4
numberOfDOF = (math.sqrt(numberOfElements)+1)*(math.sqrt(numberOfElements)+1)
numberOfDOF = int(numberOfDOF)

#Create the mapping matrix for assembly operation
#Create the ID matrix
ID_matrix = np.zeros((1,numberOfDOF))
for i in range(0,numberOfDOF):
    ID_matrix[0,i] = i+1
#Create the IX matrix
IX_matrix = np.zeros((4,numberOfElements))
additionHolder = 1
sumCorrector = math.sqrt(numberOfDOF) +1
for colChanger in range(0,numberOfElements):
    IX_matrix[0,colChanger] = colChanger + additionHolder
    IX_matrix[1, colChanger] = colChanger + additionHolder +1
    IX_matrix[2, colChanger] =colChanger + sumCorrector + additionHolder
    IX_matrix[3, colChanger] = colChanger + sumCorrector +additionHolder -1
    if (colChanger+1) % (math.sqrt(numberOfDOF)-1) == 0 and colChanger != 0:
        additionHolder = additionHolder + 1
#Create LM matrix
LM_matrix = np.zeros((4,numberOfElements))
for rowCounter in range(0,4):
    for colCounter in range(0,numberOfElements):
        indexOfConcern = IX_matrix[rowCounter,colCounter]
        indexOfConcern = int(indexOfConcern)
        valueOfConcern = ID_matrix[0,indexOfConcern-1]
        valueOfConcern = int(valueOfConcern)
        LM_matrix[rowCounter,colCounter] = valueOfConcern

#Initiate the global stiffness matrix with a size of numberOfNodes x numberOfNodes
globalStiffnessMatrix = np.zeros((numberOfDOF, numberOfDOF))
#Initiate the global forcing vector
globalForcingVector = np.zeros((numberOfDOF, 1))

#Calculation of the Jacobian matrix
jacobianMatrix = np.zeros((2,2))
elementLength = 10/(math.sqrt(numberOfElements))
jacobianMatrix[0,0] = 2*elementLength/4
jacobianMatrix[0,1] = 0
jacobianMatrix[1,0] = 0
jacobianMatrix[1,1] = 2*elementLength/4
inverseJacobianMatrix = np.linalg.inv(jacobianMatrix)
detJacobianMatrix = np.linalg.det(jacobianMatrix)
#Initiate the local stiffness matrix
localStiffnessMatrix = np.zeros((4,4))

#Implementation of the Gaussian Quadrature Scheme
def evalFunction(i,j,Je11,Je22):
    eval1 = 0
    eval2 = 0
    eval3 = 0
    eval4 = 0
    if i == 0 and j == 0:
        eval1 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval2 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval3 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval4 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))

    if i == 0 and j == 1:
        eval1 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval2 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval3 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval4 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))

    if i == 0 and j == 2:
        eval1 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval2 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval3 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval4 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))

    if i == 0 and j == 3:
        eval1 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval2 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval3 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval4 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))

    if i == 1 and j == 0:
        eval1 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval2 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval3 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval4 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))

    if i == 1 and j == 1:
        eval1 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval2 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval3 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval4 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))

    if i == 1 and j == 2:
        eval1 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval2 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval3 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval4 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))

    if i == 1 and j == 3:
        eval1 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval2 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval3 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval4 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(-1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))

    if i == 2 and j == 0:
        eval1 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval2 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval3 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval4 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))

    if i == 2 and j == 1:
        eval1 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval2 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval3 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval4 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))

    if i == 2 and j == 2:
        eval1 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval2 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval3 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval4 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))

    if i == 2 and j == 3:
        eval1 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval2 = (Je11*(1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval3 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval4 = (Je11*(1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))

    if i == 3 and j == 0:
        eval1 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval2 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval3 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval4 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))

    if i == 3 and j == 1:
        eval1 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval2 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))
        eval3 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(-1/4)*(1-math.sqrt(1/3)))
        eval4 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(-1/4)*(1+math.sqrt(1/3)))

    if i == 3 and j == 2:
        eval1 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval2 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval3 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval4 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))

    if i == 3 and j == 3:
        eval1 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval2 = (Je11*(-1/4)*(1-math.sqrt(1/3))) * (Je11*(-1/4)*(1-math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))
        eval3 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1+math.sqrt(1/3)))*(Je22*(1/4)*(1+math.sqrt(1/3)))
        eval4 = (Je11*(-1/4)*(1+math.sqrt(1/3))) * (Je11*(-1/4)*(1+math.sqrt(1/3))) + (Je22*(1/4)*(1-math.sqrt(1/3)))*(Je22*(1/4)*(1-math.sqrt(1/3)))

    return eval1 + eval2 + eval3 + eval4

#Calculation of the values of the elemental stiffness matrices
for elementCounter in range(0,numberOfElements):
    for i in range(0,4):
        for j in range(0,4):
            localStiffnessMatrix[i,j] = evalFunction(i,j,inverseJacobianMatrix[0,0],inverseJacobianMatrix[1,1])*detJacobianMatrix
            globalFirstCoor = LM_matrix[i,elementCounter]-1
            globalSecondCoor = LM_matrix[j,elementCounter]-1
            globalFirstCoor = int(globalFirstCoor)
            globalSecondCoor = int(globalSecondCoor)
            globalStiffnessMatrix[globalFirstCoor,globalSecondCoor] = localStiffnessMatrix[i,j] + globalStiffnessMatrix[globalFirstCoor,globalSecondCoor]

#Now, let's construct the matrix containing nodal unknowns
nodalValues = np.zeros((numberOfDOF,1))

#Application procedure of Dirichlet BC
colCounter = math.sqrt(numberOfElements)
rowCounter = math.sqrt(numberOfElements)
colCounter = int(colCounter)
rowCounter = int(rowCounter)

for rowModifier in range(0,rowCounter+1):
    for colModifier in range(0,colCounter+1):
        if rowModifier == 0:
            globalStiffnessMatrix[colModifier,:] = 0
            globalStiffnessMatrix[colModifier,colModifier] = 1
            boundaryValue = colModifier*elementLength*(10-colModifier*elementLength)
            globalForcingVector[colModifier] = boundaryValue
        if rowModifier == math.sqrt(numberOfElements):
            modifyNum = rowModifier*math.sqrt(numberOfDOF) + colModifier
            modifyNum = int(modifyNum)
            globalStiffnessMatrix[modifyNum,:] = 0
            globalStiffnessMatrix[modifyNum,modifyNum] = 1
        else:
            leftMostNodeNumber = rowModifier*math.sqrt(numberOfDOF)
            leftMostNodeNumber = int(leftMostNodeNumber)
            rightMostNodeNumber = rowModifier*math.sqrt(numberOfDOF) + math.sqrt(numberOfDOF) - 1
            rightMostNodeNumber = int(rightMostNodeNumber)
            globalStiffnessMatrix[leftMostNodeNumber,:] = 0
            globalStiffnessMatrix[rightMostNodeNumber,:] = 0
            globalStiffnessMatrix[leftMostNodeNumber,leftMostNodeNumber] = 1
            globalStiffnessMatrix[rightMostNodeNumber,rightMostNodeNumber] =1

inverseStiffnessMatrix = np.linalg.inv(globalStiffnessMatrix)
#Get the nodal values by multiplying the inverse of the stiffness matrix with forcing vector
nodalValues = np.matmul(inverseStiffnessMatrix,globalForcingVector)
#Print the nodal values of the solution
print(nodalValues)

#Let's write the nodal solutions to an Excel file named "solution"
#Convert the solution matrix into a dataframe
dataFrameForSolution = pd.DataFrame(nodalValues)
#Save to xlsx file
filepath = 'solution.xlsx'
dataFrameForSolution.to_excel(filepath, index=False)

#Do the contour plotting
n, m = 10,10
start = 0

if numberOfElements == 256:
    x_vals = np.arange(start, start+n+0.625, elementLength) #Arrenging the axis limits for N=256 elements
    y_vals = np.arange(start, start+m+0.625, elementLength)
else:
    x_vals = np.arange(start, start+n+1, elementLength)
    y_vals = np.arange(start, start+m+1, elementLength)

X, Y = np.meshgrid(x_vals, y_vals)
fig = plt.figure(figsize=(6,5))
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])

matrixSize = math.sqrt(numberOfDOF)
matrixSize = int(matrixSize)

nodalValuesReshaped = np.zeros((matrixSize,matrixSize))
elementTracker = 0
for rowNum in range(0,matrixSize):
    for colNum in range(0,matrixSize):
        nodalValuesReshaped[rowNum,colNum] = nodalValues[elementTracker,0]
        elementTracker = elementTracker +1

cp = plt.contourf(X, Y, nodalValuesReshaped)
plt.colorbar(cp)

ax.set_title('Contour Plot for Temperature Profile')
ax.set_xlabel('x (cm)')
ax.set_ylabel('y (cm)')
#Saving the plot to the directory where the .py is located at
plt.savefig("contourPlotResult")