### Requirements #
Have docker deamon running on you machine (boot2docker works)

### To edit the presentation #
1. make
2. `<ip of docker service>`:8888
  * with boot2docker this becomes `boot2docker ip`:8888
  * if you are running docker natively, I belive localhost:8888

### To show the presentation #
1. make dev_presentation
2. http://localhost:8000/
3. Press 's' to get the presenters view (there are notes to see)

### To print the presentation #
1. make dev_presentation
2. http://localhost:8000/print-pdf 
3. ctrl-p
