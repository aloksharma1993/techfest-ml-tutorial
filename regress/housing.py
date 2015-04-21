#! /usr/bin/env python
#from sklearn.feature_selection import VarianceThreshold
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

"""
See

https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

1. Title: Boston Housing Data

2. Sources:
   (a) Origin:  This dataset was taken from the StatLib library which is
                maintained at Carnegie Mellon University.
   (b) Creator:  Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the 
                 demand for clean air', J. Environ. Economics & Management,
                 vol.5, 81-102, 1978.
   (c) Date: July 7, 1993

3. Past Usage:
   -   Used in Belsley, Kuh & Welsch, 'Regression diagnostics ...', Wiley, 
       1980.   N.B. Various transformations are used in the table on
       pages 244-261.
    -  Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning.
       In Proceedings on the Tenth International Conference of Machine 
       Learning, 236-243, University of Massachusetts, Amherst. Morgan
       Kaufmann.

4. Relevant Information:

   Concerns housing values in suburbs of Boston.

5. Number of Instances: 506

6. Number of Attributes: 13 continuous attributes (including "class"
                         attribute "MEDV"), 1 binary-valued attribute.

7. Attribute Information:

    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over 
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds 
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks 
                 by town
    13. LSTAT    % lower status of the population
    14. MEDV     Median value of owner-occupied homes in $1000's

8. Missing Attribute Values:  None.

Here is the first line
 0.00632  18.00   2.310  0  0.5380  6.5750  65.20  4.0900   1  296.0  15.30 396.90   4.98  24.00

"""

FIELDS = [ ['CRIM',      'per capita crime rate by town'],
    ['ZN' ,       'proportion of residential land zoned for lots over 25,000 sq.ft.'],
    ['INDUS',     'proportion of non-retail business acres per town'],
    ['CHAS',      'Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)'],
    ['NOX',       'nitric oxides concentration (parts per 10 million)'],
    ['RM',        'average number of rooms per dwelling'],
    ['AGE',       'proportion of owner-occupied units built prior to 1940'],
    ['DIS',       'weighted distances to five Boston employment centres'],
    ['RAD',       'index of accessibility to radial highways'],
    ['TAX',      'full-value property-tax rate per $10,000'],
    ['PTRATIO',  'pupil-teacher ratio by town'],
    ['B',        '1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town'],
    ['LSTAT',    '% lower status of the population'],
    ['MEDV',     'Median value of owner-occupied homes in $1000\'s']]

def main():
  """Read file estimate housing prices"""
  fp_house = open('../../../data/learn/housing.data', 'r')
  X_train = []
  y_train = []
  X_test = []
  y_test = []
  line_count = 0
  for line in fp_house:
    line = line.rstrip()
    line_splits = line.split()
    fields_in = {}
    count = 0
    for field in line_splits:
      field_name = FIELDS[count][0]
      fields_in[field_name] = float(field)
      count += 1
    # THE price
    y = fields_in['MEDV']

    # Dependent variables
    rooms = fields_in['RM']
    lower_stat = fields_in['LSTAT']
    crime = fields_in['CRIM']
    blacks = fields_in['B']
    large_plot = fields_in['ZN']

    next_x = [fields_in['B'], rooms, lower_stat, large_plot]

    # We train using first 100 lines
    if line_count < 100:
      X_train.append(next_x)
      y_train.append(y)
    else:
      # We test our model against lines 101-200
      X_test.append(next_x)
      y_test.append(y)
    line_count += 1
    if line_count > 200:
      break
  fp_house.close()

  clf = linear_model.LinearRegression()

  # CREATE model
  clf.fit (X_train, y_train)
  print('Variance score: %.2f' % clf.score(X_test, y_test))

  # LET's plot it
  plt.scatter([x[0] for x in X_test], y_test,  color='black')

  plt.xticks(())
  plt.yticks(())

  plt.show()
if __name__ == '__main__':
  main()
