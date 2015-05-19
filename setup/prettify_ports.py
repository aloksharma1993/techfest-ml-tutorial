import os
import csv

if __name__ == '__main__':
  _, nodename, _, _, _ = os.uname()
  with open('setup/ports.csv') as infile:
    reader = csv.reader(infile)
    with open('setup/pretty_ports.csv', 'w') as outfile:
      writer = csv.writer(outfile)
      writer.writerow(['', '', ''])
      for port, password in reader:
        writer.writerow(['ipython notebook server url', '', 'password'])
        writer.writerow(['', '', ''])
        writer.writerow(['https://{node}:{port}'.format(node=nodename, port=port), '', password])
        writer.writerow(['', '', ''])
        writer.writerow(['', '', ''])
