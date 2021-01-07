# lstm_text_classifier
**lstm_text_classifier** is a text classifier based on lstm model.

## Requirement
python3  
torch==1.6.0  
pandas  
matplotlib  
torchtext==0.8.0  
sklearn    

## Usage
```shell
./lstm_text_classifier.sh <-i infile>
                          [-m max_length]
                          [-b batch_size]
                          [-n num_epochs]
                          [-l learning_rate]
                          [-o outdir]
```
**-i**: filename of raw input text. Input text shoud be in TSV format with 4 colomns as follows:  
&nbsp;&nbsp;&nbsp;&nbsp;ID*TAB*title*TAB*content*TAB*label  
&nbsp;&nbsp;&nbsp;&nbsp;labels should be intergers.  
**-m**: max length of sequence. Default=200  
**-b**: batch size for training, valid and test data. Default=32  
**-n**: number of epochs to use. Default=10  
**-r**: learning rate of model. Default=0.001  
**-o**: directory to save trained model. Default=./model  

## Example
Use following command to test:
```shell
./lstm_text_classifier.sh -i testdata/test.tsv
```
