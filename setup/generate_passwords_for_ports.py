import csv
import random
import string


ports = range(60000, 60020)

if __name__ == '__main__':
  writer = csv.writer(open('setup/ports.csv', 'w'))
  for port in ports:
    password = ''.join([random.choice(string.hexdigits) for x in range(6)])
    writer.writerow([port, password])
    
