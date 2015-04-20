"""
Creates featurized train and test datasets from the train set available at
http://physionet.org/challenge/2012/.
This script expects the "Outcomes-a.txt" file on the Challenge website in the
same directory as this script. This script also expects the patient record files
to be in a folder called "set-a".
"""

import csv
import os
import random

from collections import Counter
from collections import defaultdict

import numpy as np

from sklearn.feature_extraction import DictVectorizer


TRAIN_FRACTION = 0.8

STATIC_FIELDS = ['Age', 'Gender', 'Height', 'ICUType']
FIELDS = ['Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol',
          'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT',
          'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'Na', 'NIDiasABP',
          'NIMAP', 'NISysABP', 'PaCO2', 'PaO2', 'pH', 'Platelets', 'RespRate',
          'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine', 'WBC',
          'Weight']  # Weight is both a static and a time-series field, so may be -1.
NAN_REPLACE = -1

def set_features_to_nan(fieldname):
  """
  Feature values to add to dataset if variable `fieldname` was not observed.
  """
  field_features = {}
  field_features['{}_min'.format(fieldname)] = NAN_REPLACE
  field_features['{}_max'.format(fieldname)] = NAN_REPLACE
  field_features['{}_mean'.format(fieldname)] = NAN_REPLACE
  field_features['{}_first'.format(fieldname)] = NAN_REPLACE
  field_features['{}_last'.format(fieldname)] = NAN_REPLACE
  field_features['{}_diff'.format(fieldname)] = NAN_REPLACE
  return field_features


def featurize(data):
  """ Create features from time-series data. """
  features = {}
  missing_weight = False
  for fieldname in STATIC_FIELDS:
    # Static fields use -1 to denote that the value was not measured.
    if data[fieldname][0][1] == -1:
      features[fieldname] = NAN_REPLACE
    else:
      features[fieldname] = float(data[fieldname][0][1])
  for fieldname in FIELDS:
    # Time-series fields may or may not be measured, but if they are present
    # in the dataset, then the value will be valid (i.e. nonnegative).
    if fieldname in data:
      values = [float(d[1]) for d in data[fieldname]]
      if -1 in values and fieldname == 'Weight':
        # Record that weight was missing for this record id.
        missing_weight = True
        field_features = set_features_to_nan(fieldname)
      else:
        field_features = {}
        field_features['{}_min'.format(fieldname)] = min(values)
        field_features['{}_max'.format(fieldname)] = max(values)
        field_features['{}_mean'.format(fieldname)] = np.mean(values)
        field_features['{}_first'.format(fieldname)] = values[0]
        field_features['{}_last'.format(fieldname)] = values[-1]
        field_features['{}_diff'.format(fieldname)] = values[-1] - values[0]
    else:
      field_features = set_features_to_nan(fieldname)
    features.update(field_features)
  return features, missing_weight


def read_record_files(outcomes):
  """ Reads in time series measurements from record file.

  :param outcomes: dictionary of patient record id to list of outcomes.
  :returns: nested dictionary of patient record id to dictionary of variable
    name to list of (measurement time, value) tuples.
    Example: {'123': { # patient record id
                'SysABP': [(03:27, 147), (04:57, 118)]  # variable SysABP
                ...
             }}
  """
  input_data = {}
  for record_id in outcomes.keys():
    filename = '{}.txt'.format(record_id)
    f = open(os.path.join('set-a', filename))
    reader = csv.reader(f)
    # skip header
    reader.next()
    # skip record id
    reader.next()
    data = defaultdict(list)
    for row in reader:
      data[row[1]].append((row[0], row[2]))
    input_data[record_id] = data
    f.close()
  return input_data


if __name__ == '__main__':

  ###########################
  # Read and featurize data #
  ###########################
  outcomes = {}
  with open('Outcomes-a.txt', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    # skip header
    reader.next()
    for row in reader:
      outcomes[row[0]] = row

  # Read patient record files.
  input_data = read_record_files(outcomes)

  # Create features and labels.
  features = []
  labels = []
  ids = []
  count_missing_weight = 0
  for id, data in input_data.iteritems():
    feats, missing_weight = featurize(data)
    if missing_weight:
      count_missing_weight += 1
    features.append(feats)
    labels.append(int(outcomes[id][5])) # in-hospital_death
    ids.append(id)

  v = DictVectorizer()
  features = v.fit_transform(features).toarray()
  labels = np.array([labels])

  ##########################
  # Create train/test sets #
  ##########################
  header = list(v.get_feature_names())
  header.append('In-hospital_death')

  # Number of records to put in training set.
  num_train = int(len(outcomes) * TRAIN_FRACTION)

  # Randomly shuffle data.
  indices = range(0, len(outcomes))
  random.shuffle(indices)

  # Training set.
  features_train = features[indices[0:num_train],:]
  labels_train = labels[:,indices[0:num_train]]
  # Join features and outcome.
  train_set = np.concatenate((features_train, labels_train.T), axis=1)
  # Write data to csv file.
  with open('train-a.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(train_set)

  # Test set.
  features_test = features[indices[num_train:],:]
  labels_test = labels[:,indices[num_train:]]
  test_set = np.concatenate((features_test, labels_test.T), axis=1)
  with open('test-a.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)
    writer.writerows(test_set)
