FROM ipython/scipyserver

## Regression example
ADD regress/housing.ipynb /notebooks/housing.ipynb
ADD regress/regressor.png /notebooks/
ADD regress/colorful_row_houses.jpg /notebooks/
ADD regress/boston_roads.png /notebooks/

## classification example
ADD notebooks/classification_example.ipynb /notebooks/
