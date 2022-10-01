# The Source Code for LSH-APG (Submitted to SIGMOD 2023 Second Round)
-----------------------------------------------------------------------------------------------------------------
## Introduction
This is a source code for the algorithm described in the paper **LSH-APG: Building Approximate Proximity Graphs Through Locality-Sensitive Hashing(Submitted to SIGMOD 2023 Second Round)**. We call it as **LG** project.

## Compilation

**LG** project is written by **C++** and can be complied by **g++** in **Linux** and **MSVC** in **Windows**. It adopt `openMP` for parallelism.


### Installation
#### Windows
We can use **Visual Studio 2019** to build the project with importing all the files in the directory `./cppCode/LSH-APG/src/`.

#### Linux
```bash
cd ./cppCode/LSH-APG
make
```
The excutable file is then in dbLSH directory, called as `lgo`

## Usage

### Command Usage

-------------------------------------------------------------------
> lgo datasetName
-------------------------------------------------------------------
(the first parameter specifies the procedure be executed and change)

### Parameter explanation

- datasetName  : dataset name
-------------------------------------------------------------------

FOR EXAMPLE, YOU CAN RUN THE FOLLOWING CODE IN COMMAND LINE AFTER BUILD ALL THE TOOLS:

```bash
cd ./cppCode/LSH-APG
./lgo audio
```

## Dataset

In our project, the format of the input file (such as `audio.data_new`, which is in `float` data type) is the same as that in [LSHBOX](https://github.com/RSIA-LIESMARS-WHU/LSHBOX). It is a binary file, which is organized as the following format:

>{Bytes of the data type (int)} {The size of the vectors (int)} {The dimension of the vectors (int)} {All of the binary vector, arranged in turn (float)}


For your application, you should also transform your dataset into this binary format, then rename it as `[datasetName].data_new` and put it in the directory `./dataset`.

A sample dataset `audio.data_new` has been put in the directory `./dataset`.
Also, you can get it, `audio.data`, from [here](http://www.cs.princeton.edu/cass/audio.tar.gz)(if so, rename it as `audio.data_new`). If the link is invalid, you can also get it from [data](https://github.com/RSIA-LIESMARS-WHU/LSHBOX-sample-data).

For the datasets we use, you can get the raw data from following links: [MNIST](http://yann.lecun.com/exdb/mnist/index.html), [Deep1M](https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz), [GIST](http://corpus-texmex.irisa.fr/), [TinyImages80M](https://hyper.ai/tracker/download?torrent=6552), [SIFT](http://corpus-texmex.irisa.fr/). Next, you should transform your raw dataset into the mentioned binary format, then rename it is `[datasetName].data_new` and put it in the directory `./dataset`.
