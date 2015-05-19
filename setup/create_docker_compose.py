import csv

compose_yml_template = """server{port}:
  build: .
  ports:
    - {port}:8888
  environment:
    - PASSWORD={password}
"""

if __name__ == '__main__':
  reader = csv.reader(open('setup/ports.csv', 'r'))
  with open('multy_compose.yml', 'w') as outfile:
    for port, password in reader:
      outfile.write(compose_yml_template.format(port=port,
                                                password=password.strip()))
