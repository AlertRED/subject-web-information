# WebInformation - Machine learning
### Task
Your task is to display the top words from all the files 
which have positvie expression and negative expression and 
also mention the intensity along with them. after that if you 
pass any comment to your classifier then it must be smart enough 
to understand either it is positve comment or negative.
## Install
### Requirements

    pip install -r requirements.txt
* Python3.7
* Conda 4.8.3
## Datasets
When working on the project used dataset

http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

## Usage
### ML class
* train - Trains a model on data.

* test - Testing a model on data.

* save_model - Saves model with .pkl extension

* save_vectorizer - Saves vectors with .vec extension

* load_model - Loads a trained model from a file

* load_vectorizer - Loads vectors from a file

### Example code
```python
from main import Ml

dir_trains = "dataset\\train"
dir_tests = "dataset\\test"

ml = ML()
ml.train(dir_trains)
accuracy = ml.test(dir_tests)
print(accuracy)
```
    

