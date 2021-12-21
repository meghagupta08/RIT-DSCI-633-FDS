# SET EXPANDER

## Installing dependencies
* Recommended python version 3.7
* Execute the following command to install torch 1.4.0 for python 3.7 . 
It is required by nlp-architect module, and it is observed that on Windows OS with 
python 3.7, pip is only allowing torch version 1.7+. For the workaround,
 below command install the module from the pytorch repository.
    ```bash
    pip install https://download.pytorch.org/whl/cu101/torch-1.4.0-cp37-cp37m-win_amd64.whl
    ```
* Install nlp-architect module using `set_expander_requirements.txt` located
at `src/openkg/entity_extraction/set_expander_requirements.txt`. This file
contains dependency for nlp-architect along with `gensim==3.8.0`. If gensim
 version is not specified, it installs version 4.0.0 which is not compatible
with nlp-architect module.
    ```bash
    pip install -r src/openkg/entity_extraction/set_expander_requirements.txt
    ```

## Testing set expander
* A separate test file `test_set_expander.py` is added to test `get_entities` and `train` method of 
**SetExpander** class.
* This test file was not leveraging `EntityExtractionFactory` class to get
the instance of the `SetExpander` class, because during the installation of the dependencies,
 conflict was observed between nlp-architect and other existing dependencies (ex Flair). 
Due to which `EntityExtractionFactory` class was not loaded, as it was importing
those dependencies.
* In file `test_ner.py` proper implementation using `EntityExtractionFactory` is 
implemented and the code is commented. Once the dependency conflict is resolved,
 the code can be uncommented and can be used.
* To test the `SetExpander` class functionality, please run `test_set_expander.py`, with
appropriate test method.
* The `test_set_expander_get_entities` method will initially download the pretrained
model and untar it. This will be passed to SetExpander, for expanding the
seed terms. Once terms are expanded, the downloaded compressed pretrained model file
will be removed, keeping the extracted model file intact.
* The `test_set_expander_train` method will first download the marked corpus file
for training the model. The extracted marked corpus file is around **~15 GiB**.
* Training the model consumed around 16 GB of RAM and took around 1.5hours