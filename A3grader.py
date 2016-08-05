# Delete all variables defined so far (in notebook)
for name in dir():
    if not callable(globals()[name]) and not name.startswith('_'):
        del globals()[name]

import numpy as np
import os

# import A3mysolution as mine
# import imp
# imp.reload(mine)
# trainValidateTestKFolds = mine.trainValidateTestKFolds
# trainLinear = mine.trainLinear
# evaluateLinear = mine.evaluateLinear
# trainNN = mine.trainNN
# evaluateNN = mine.evaluateNN

import neuralnetworks as nn

def within(correct, attempt, diff):
    return np.abs((correct-attempt) / correct)  < diff

g = 0

for func in ['trainValidateTestKFolds', 'trainLinear',
              'evaluateLinear', 'trainNN', 'evaluateNN']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined.'.format(func))

X = np.arange(20).reshape((-1,1))
T = np.abs(X -10) + X
result = trainValidateTestKFolds(trainLinear,evaluateLinear,X,T,
                                 range(0,101,10),nFolds=5,shuffle=False)
correctResult = np.array(
    [[10.00,    3.158,   4.132,   2.414],
     [20.00,    4.368,   5.021,   3.641],
     [10.00,    3.245,   4.178,   5.030],
     [20.00,    4.448,   6.070,   2.024],
     [20.00,    2.426,   2.972,   10.886]])

result = np.array(result)
correct = True
print(' Testing: result = trainValidateTestKFolds(trainLinear,evaluateLinear,X,T,\n                  range(0,101,10),nFolds=5,shuffle=False)')

print(' Your result is')
for row in result:
    print('   {:3g}   {:.4g}   {:.4g}   {:.4g}'.format(*row))

if np.abs(result[:,0]-correctResult[:,0]).sum() > 20:
    print(' 0/20 points. First column, of best lambda values, is not correct.')
    correct = False
else:
    g += 20
    print('20/20 points. First column, of best lambda values, is correct.')
    
if np.abs(result[:,1:]-correctResult[:,1:]).sum() > 10:
    print(' 0/20 points. Columns of RMSE values are not correct.')
    correct = False
else:
    g += 20
    print('20/20 points. Columns of RMSE values are correct.')

if not correct:
    print(' Correct value of result is')
    for row in result:
        print('   {:3g}   {:.4g}   {:.4g}   {:.4g}'.format(*row))

print()
print(''' Testing:
   import itertools
   parms = list(itertools.product([[5],[5,5],[2,2,2]], [10,50,100,200]))
   te = []
   for rep in range(5):
       result = trainValidateTestKFolds(trainNN,evaluateNN,X,T,
                                        parms,
                                        nFolds=4,shuffle=False)
       resulte = np.array([r[1:] for r in result])
       meanTestRMSE = resulte[:,-1].mean()
       print('     ',meanTestRMSE)
       te.append(meanTestRMSE)''')


import itertools
parms = list(itertools.product([[5],[5,5],[2,2,2]], [10,50,100,200]))
te = []
for rep in range(5):
    result = trainValidateTestKFolds(trainNN,evaluateNN,X,T,
                                     parms,
                                     nFolds=4,shuffle=False)
    resulte = np.array([r[1:] for r in result])
    meanTestRMSE = resulte[:,-1].mean()
    print('     ',meanTestRMSE)
    te.append(meanTestRMSE)

if np.array(te).mean() > 5:
    print(' 0/40 points. Mean test RMSE should be less than 5, but it is not.')
else:
    g += 40
    print('40/40 points. Mean test RMSE is less than 5 as it should be.')


print('\n{} Grade is {}/100'.format(os.getcwd().split('/')[-1], g))
print('Up to 20 more points will be given based on the qualty of your descriptions of the method and the results.')
