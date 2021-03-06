---
layout: post
category : [blog]
tags : [GPGPU, setup]
title: 'Installing cuda toolkit and related R packages'
summary: setting up drivers and packages
author: "Dmitriy Selivanov"
license: GPL (>= 2)
redirect_from:
  - /blog/2015/06/04/installing-cuda-toolkit-and-gputools/
src: 2015-06-04-installing-cuda-toolkit-and-gputools.Rmd
---

The main purpose of this post is to keep all steps of installing cuda toolkit (and R related packages) and in one place. Also I hope this may be useful for someone.

## Installing cuda toolkit ( Ubuntu )
First of all we need to install **nvidia cuda toolkti**. I'am on latest ubuntu 15.04, but found [this article](http://www.r-tutor.com/gpu-computing/cuda-installation/cuda7.0-ubuntu) well suited for me. But there are few additions:  

1. It is very important to have no nvidia drivers before installation ( first I corrupted my system and have to reinstall it :-( ). So I recommend to switch to real terminal (`ctrl + alt + f1`), remove all nvidia stuff `sudo apt-get purge nvidia-*` and then follow steps from article above.  

2. This will install cuda toolkit and corresponding nvidia drivers.

{% highlight bash %}
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1410/x86_64/cuda-repo-ubuntu1410_7.0-28_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1410_7.0-28_amd64.deb
sudo apt-get update
sudo apt-get install cuda
{% endhighlight %}

3. After installation we need to modify our `.bashrc` file. Add following lines:

{% highlight bash %}
export CUDA_HOME=/usr/local/cuda-7.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64

PATH=${CUDA_HOME}/bin:${PATH}
PATH=${CUDA_HOME}/bin/nvcc:${PATH}
export PATH
{% endhighlight %}

Note, that I added path to `nvcc` compiler.

## Installing gputools
First simply try:

{% highlight r %}
install.packages('gputools', repos = 'http://cran.rstudio.com/')
{% endhighlight %}
After that I recieved:

> Unsupported gpu architecture 'compute_10'

Solving this issue I found this [link](https://devtalk.nvidia.com/default/topic/606195/-solved-nvcc-fatal-unsupported-gpu-architecture-compute_21-/) useful. 
I have gt525m card and have compute capability 2.1. You can verify your GPU capabilities [here](https://developer.nvidia.com/cuda-gpus). 
So I downloaded gputools source package:

{% highlight bash %}
cd ~
wget http://cran.r-project.org/src/contrib/gputools_0.28.tar.gz
tar -zxvf gputools_0.28.tar.gz
{% endhighlight %}
and replace following string

{% highlight bash %}
NVCC := $(CUDA_HOME)/bin/nvcc -gencode arch=compute_10,code=sm_10 -gencode arch=compute_13,code=sm_13 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30
{% endhighlight %}
in `gputools/src/Makefile` by

{% highlight bash %}
NVCC := $(CUDA_HOME)/bin/nvcc -gencode arch=compute_20,code=sm_21
{% endhighlight %}
Next try to gzip it back and install from source:

{% highlight r %}
install.packages("~/gputools.tar.gz", repos = NULL, type = "source")
{% endhighlight %}
Than I recieved:

> rinterface.cu:1:14: fatal error: R.h: No such file or directory #include<R.h>

We have to adjust R header dir location. First of all look for `R.h`:

{% highlight bash %}
locate \/R.h
{% endhighlight %}
replace string `R_INC := $(R_HOME)/include` in `gputools/src/config.mk` string by found path:
```
R_INC := /usr/share/R/include
```


In case we recieve error regarding shared `libcublas.so` we also need to adjust links for `libcublas` shared library:

{% highlight bash %}
sudo ln -s /usr/local/cuda/lib64/libcublas.so.7.0 /usr/lib/libcublas.so.7.0
{% endhighlight %}
thanks to this [thread](http://stackoverflow.com/questions/10808958/why-cant-libcudart-so-4-be-found-when-compiling-the-cuda-samples-under-ubuntu).

## Testing performance
here is simple benchmark:

{% highlight r %}
library(gputools)
N <- 1e3
m <- matrix(sample(100, size = N*N, replace = T), nrow = N)
system.time(dist(m))
system.time(gpuDist(m))
{% endhighlight %}
