# Vietnamese Semantic Role Labelling (vnSRL) v1.0.0 - 2016-02-20
-----------------------------------------------------------------
Code by **Thai-Hoang Pham**, **Xuan-Khoai Pham**

## 1. Introduction

The **vnSRL** system is used to labelling semantic roles of arguments for each predicate in a Vietnamese sentence. This software is written by Python 2.x.

## 2. Installation

This software depends on NumPy, SciPy, Scikit-learn, Pandas, Pulp, ETE2, six Python packages for scientific computing. You must have them installed prior to using vnSRL.

The simple way to install them is using pip:

```sh
	# pip install -U numpy scipy scikit-learn pandas pulp ete2
```

## 3. Usage

### 3.1. Data

The input data's format of vnSRL is Penn Treebank format. For details, see sample data *100-sen.txt* in a directory **'data/input'**.

For classifying task, put your data in a directory **'data/input'**.

Your output file is stored in a directory **'data/output'**.

### 3.2. Command-line Usage

You can use vnSRL software by a following command:

```sh
	$ python vnSRL.py <input> <ilp> <embedding> <output>
```

Positional arguments:

* ``input``:       input file name
* ``ilp``:         integer linear programming post-processing (1 for using ilp and 0 for vice versa)
* ``embedding``:   word embedding file (skipgram or glove)
* ``output``:      output file name

For example, if you want to use this software to label the file input.txt, ilp for post-processing, glove for word embedding and the output file output.txt, you use the command:

```sh
	$ python vnSRL.py input.txt 1 glove output.txt
```

**Note**: In our experiment, integer linear programming method helps to improve the performance about 0.4% but takes very long time to run (about 50x longer).

## 4. References

[T-H. Pham, X-K. Pham, and P. Le-Hong, "Building a Semantic Role Labelling System for Vietnamese", Proceedings of the 10th International Conference on Digital Information Management, Jeju Islands, South Korea, IEEE, 10/2015.](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7381877)

## 5. Contact

Thai-Hoang Pham < phamthaihoang.hn@gmail.com >

FPT Technology Research Institute, FPT University

