# Neural Tensor Network for Knowledge Base Completion with GRAKN.AI
-------------------------------------
## Credits for the (bulk of the) neural network file code go to Siddarth Agarwal. His GitHub repo for this project is https://github.com/siddharth-agrawal/Neural-Tensor-Network

## Credits for the neural tensor network project itself go to http://nlp.stanford.edu/~socherr/SocherChenManningNg_NIPS2013.pdf
-------------------------------------

To run this program, you will first need to have installed Grakn (this was written on 0.15.0). If you're coming here from the blog, then you probably already have grakn installed, but if you don't, you can use something like Homebrew to quickly get up and running, using the command
```
brew install grakn
```
You will then have to go into the Grakn directory and start the Grakn shell script, like so
```
/YOUR-GRAKN-DIRECTORY/bin/grakn.sh start
```
This will allow you to make queries through the Graql shell

## Running the program
You will need Python 2. for this project. I cannot guarantee compatibility with Python 3.

First, navigate to the project directory and type `pip install -r requirements.txt` to install all the modules necessary to run this project

Next you must go into `insertLexicon.py` and at the top of the file, replace the `PATH-TO-GRAKN` in
```
_GRAQL_PATH = "/PATH-TO-GRAKN/bin/graql.sh"
```
with the directory path and name that you used to start the Grakn engine itself. For example, on my machine this line looks like
```
_GRAQL_PATH = "/Users/nickpowell/Documents/Grakn/bin/graql.sh"
```
-------------------------------------
You can run either the `neuralTensorNetwork.py` file, or the `insertLexicon.py` file. `insertLexicon.py` is a stand-alone separate from the neural netwrok that loads the ontology, inserts entities and relations, and checks for initial inferences. `neuralTensorNetwork.py` does the work of `insertLexicon.py` as well as that of the neural network. In most scenarios you'd want to run `neuralTensorNetwork.py`. The rest of this section assumes this is the file you are executing.

If it is your first time running the program, you will need to run it with the `buildGraph` flag on in order to build the ontology and ruleset. You will also need to specify a Graql keyspace with `-k`. The shell command looks like this:
```
python ./neuralTensorNetwork.py -k insert_your_keyspace_here --buildGraph
```
If you have multiple versions of Python installed, you may want to specify the 2. version, for example
```
python2.7 ./neuralTensorNetwork.py -k insert_your_keyspace_here --buildGraph
```
Once you have loaded the ontology and ruleset for a particular keyspace, REMOVE the `--buildGraph` flag from your command and on every subsequent program execution to that keyspace, simply use
```
python ./neuralTensorNetwork.py -k insert_your_keyspace_here
```
If you try to add the ontology to a keyspace that already has the ontology loaded, you may encounter errors, especially if you have made changes to the ontology.gql file.
