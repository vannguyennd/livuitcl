# LiVu-ITCL

The source code samples for reproducing the experimental results mentioned in our paper "An Information-Theoretic and Contrastive Learning-based Approach for Identifying Code Statements Causing Software Vulnerability".

## Datasets 

We used three real-world datasets including the CWE-399 dataset with 1,010 and 1,313 vulnerable/non-vulnerable functions for resource management error vulnerabilities, the CWE-119 dataset with 5,582 and 5,099 vulnerable/non-vulnerable functions for the buffer error vulnerabilities, and a big C/C++ dataset provided by Fan et al. (2020) containing many types of vulnerabilities such as Out-of-bounds Write, Improper Input Validation, and Path
Traversal. For the CWE-399 and CWE-199 datasets collected by Li et al. (2018), we used the ones processed by Nguyen et al. (2021). Additionally, the Fan et al.’s dataset is considered as one of the largest vulnerability datasets that includes the ground truth at the statement level. The dataset is collected
from 348 open-source Github projects from 2002 to 2019. It consists of 188,636 C/C++ source code functions where a ratio of vulnerable functions is 5.7% (i.e., 10,900 vulnerable functions).

## Requirements 

We implemented our LiVu-ITCL method using Tensorflow (Abadi et al. 2016) (version 2.5) and Python (version 3.8). Other required packages are scikit-learn, numpy, scipy, and pickle.

## Running source code samples

Here, we provide the instructions for using the source code samples of our LiVu-ITCL method the on the Fan et al.’s dataset.

## Folders and files

The folder named “Fan_dataset” consists of all of the necessary files containing the Fan et al.’s dataset. The file named “Fan_data_train_evaluate.py” is the source code for our proposed LiVu-ITCL method in both training and evaluating processes. The file named “Fan_data_VCP_VCA_TopK_IFA.py” is the source code for computing the main measures of fine-grained vulnerability detection including VCP, VCA, Top-10 accuracy (Top-10 ACC), and IFA.

The file named “Utils.py” is a collection of supported Python functions used in the training and evaluating processes of the model.

## Train, evaluate and get the results

To train our model, please use the following command, for example, " python Fan_data_train_evaluate.py --lr=1e-4 --sigma=1e-1 --tau=0.5 --temp=0.5 --dim_dnn=300 --clusters=7 --train_epochs=5 --home_dir=./Fan_data_results/ --do_train ".

To evaluate our proposed LiVu-ITCL method performance, please use the following command, for example, " python Fan_data_train_evaluate.py --lr=1e-4 --sigma=1e-1 --tau=0.5 --temp=0.5 --dim_dnn=300 --clusters=7 --home_dir=./Fan_data_results/ --do_eval ".

To get the results for the main measures of fine-grained vulnerability detection including VCP, VCA, Top-10 accuracy (Top-10 ACC), and IFA, please use the following command, for example, " python Fan_data_VCP_VCA_TopK_IFA.py './Fan_data_results/' ".

## The model configuration 

For the LiVu-ITCL model configuration, please read the Model configuration section in the appendix of our paper for details.
