---
layout: post
category : blog
tags : [R, LSH, functional programming]
---
{% include JB/setup %}  

## Introduction
In the next series of posts I will try to explain base concepts **Locality Sensitive Hashing technique**.  

Note, that I will try to follow general functional programming style. So I will use R's [Higher-Order Functions](https://stat.ethz.ch/R-manual/R-devel/library/base/html/funprog.html) instead of traditional **R's _\*apply_** functions family (I suppose this post will be more readable for non R users). Also I will use **brilliant pipe operator** ```%>%``` from [magrittr](http://cran.r-project.org/web/packages/magrittr/) package. We will start with basic concepts, but end with very efficient implementation in R (it is about 100 times faster than python implementations I found).

## The problem
Imagine the following interesting problem. We have two **very large** social netwotks (for example **facebook and google+**), which have hundreds of millions of profiles and we want to determine profiles owned by same person. One reasonable approach is to assume that these people have nearly same or at least highly overlapped sets of friends in both networks. One well known measure for determining degree of similarity of sets is [Jaccard Index](http://en.wikipedia.org/wiki/Jaccard_index):  
$$J(SET_1, SET_2) = {|SET_1 \cap SET_2|\over |SET_1 \cup SET_2| }$$

Set operations are computationally cheap and straightforward solution seems quite good. But let's try to estimate computational time for duplicates detection for only people with name "John Smith". Imagine that in average each person has 100 friends:

{% highlight r %}
# for reproducible results
set.seed(seed = 17)
library('microbenchmark')
# we will use brilliant pipe operator %>%
library('magrittr')
jaccard <- function(x, y) {
  set_intersection <- length(intersect(x, y))
  set_union <- length(union(x, y))
  return(set_intersection / set_union)
}
# generate "lastnames"
lastnames <- Map(function(x) paste(sample(letters, 3), collapse = ''), 1:1e5) %>% unique
print(head(lastnames))
{% endhighlight %}



{% highlight text %}
## [[1]]
## [1] "eyl"
## 
## [[2]]
## [1] "ukm"
## 
## [[3]]
## [1] "fes"
## 
## [[4]]
## [1] "fka"
## 
## [[5]]
## [1] "vuw"
## 
## [[6]]
## [1] "ypg"
{% endhighlight %}



{% highlight r %}
friends_set_1 <- sample(lastnames, 100, replace = F)
friends_set_2 <- sample(lastnames, 100, replace = F)
microbenchmark(jaccard(friends_set_1, friends_set_2))
{% endhighlight %}



{% highlight text %}
## Unit: microseconds
##                                   expr    min     lq     mean  median
##  jaccard(friends_set_1, friends_set_2) 45.646 47.417 50.72362 48.4045
##       uq     max neval
##  49.9435 150.343   100
{% endhighlight %}
One operation takes 50 microseconds in average (on my machine). If we have 100000 of peoples with name _John Smith_ and we have to compare all pairs, total computation **will take more than 100 hours**!

{% highlight r %}
hours <- (50 * 1e-6) * 1e5 * 1e5 / 60 / 60
hours
{% endhighlight %}



{% highlight text %}
## [1] 138.8889
{% endhighlight %}
Of course this is unappropriate because of $$O(n^2)$$ complexity of our brute-force algorithm.  

## Minhashing
To solve this kind problem we will use [Locality-sensitive hashing]((http://en.wikipedia.org/wiki/Locality-sensitive_hashing)) - a method of performing probabilistic dimension reduction of high-dimensional data. It provides good tradeoff between accuracy and computational time and roughly speaking has $$O(n)$$ complexity.  
I will explain one scheme of **LSH**, called [MinHash](http://en.wikipedia.org/wiki/MinHash).  
The intuition of the method is the following: we will try to hash the input items so that similar items are mapped to the same buckets with high probability (the number of buckets being much smaller than the universe of possible input items).  
Let's construct simple example:

{% highlight r %}
set1 <- c('SMITH', 'JOHNSON', 'WILLIAMS', 'BROWN')
set2 <- c('SMITH', 'JOHNSON', 'BROWN')
set3 <- c('THOMAS', 'MARTINEZ', 'DAVIS')
set_list <- list(set1, set2, set3)
{% endhighlight %}
Now we have 3 sets to compare and identify profiles, related to same "John Smith". From these sets we will construct matrix which encode relations between sets:

{% highlight r %}
sets_dict <- unlist(set_list) %>% unique

m <- Map(f = function(set, dict) as.integer(dict %in% set), 
         set_list, 
         MoreArgs = list(dict = sets_dict)) %>% 
  do.call(what = cbind, .)

# This is equal to more traditional R's sapply call:
# m <- sapply(set_list, FUN = function(set, dict) as.integer(dict %in% set), dict = sets_dict, simplify = T)

dimnames(m) <- list(sets_dict, paste('set', 1:length(set_list), sep = '_'))
print(m)
{% endhighlight %}



{% highlight text %}
##          set_1 set_2 set_3
## SMITH        1     1     0
## JOHNSON      1     1     0
## WILLIAMS     1     0     0
## BROWN        1     1     0
## THOMAS       0     0     1
## MARTINEZ     0     0     1
## DAVIS        0     0     1
{% endhighlight %}
Let's call this matrix **input-matrix**.
In our representation similarity of two sets from source array equal to the similarity of two corresponding columns with non-zero rows:  

name | set_1 | set_2 | intersecton | union  
--|--|---|---|----
SMITH|1|1|+|+
JOHNSON|1|1|+|+  
WILLIAMS|1|0|-|+  
BROWN|1|1|+|+  
THOMAS|0|0|-|-  
MARTINEZ|0|0|-|-  
DAVIS|0|0|-|-  

From table above we can conclude, that **jaccard index between set\_1 and set\_2 is 0.75**.  
Let's check:

{% highlight r %}
print(jaccard(set1, set2))
{% endhighlight %}



{% highlight text %}
## [1] 0.75
{% endhighlight %}



{% highlight r %}
column_jaccard <-  function(c1, c2) {
  non_zero <- which(c1 | c2)
  column_intersect <- sum(c1[non_zero] & c2[non_zero])
  column_union <- length(non_zero)
  return(column_intersect / column_union)
}
isTRUE(jaccard(set1, set2) == column_jaccard(m[, 1], m[, 2]))
{% endhighlight %}



{% highlight text %}
## [1] TRUE
{% endhighlight %}
All the magic starts here. Suppose random permutation of rows of the **input-matrix** `m`. And let's define **minhash function** $$h(c)$$ = # of first row in which column $$c == 1$$. If we will use $$N$$ **independent** permutations we will end with $$N$$ minhash functions. So we can construct **signature-matrix** from **input-matrix** using these minhash functions. Below we will do it not very efficiently with 2 nested ```for``` loops. But the logic should be very clear.

{% highlight r %}
# for our toy example we will pick N = 4
N <- 4
sm <- matrix(data = NA_integer_, nrow = N, ncol = ncol(m))
perms <- matrix(data = NA_integer_, nrow = nrow(m), ncol = N)
# calculate indexes for non-zero entries for each column
non_zero_row_indexes <- apply(m, MARGIN = 2, FUN = function(x) which (x != 0) )
for (i in 1 : N) {
  # calculate permutations
  perm <- sample(nrow(m))
  perms[, i] <- perm
  # fill row of signature matrix
  for (j in 1:ncol(m))
    sm[i, j] <-  min(perm[non_zero_row_indexes[[j]]])
}
print(sm)
{% endhighlight %}



{% highlight text %}
##      [,1] [,2] [,3]
## [1,]    3    3    1
## [2,]    1    1    3
## [3,]    1    1    2
## [4,]    1    1    4
{% endhighlight %}
You can see how we obtain **signature-matrix** matrix after "minhash transformation". Permutations and corresponding signatures marked with same colors:

|perm_1|perm_2|perm_3|perm_4|set_1| set_2| set_3|
 |--|--|--|--|--|--|--|
 <span style="background-color:lightgreen">4 </span>| <span style="background-color:orange">1 </span>| <span style="background-color:lightblue">4 </span>| <span style="background-color:yellow">6 </span>| 1 | 1 | 0 |
 <span style="background-color:lightgreen">3 </span>| <span style="background-color:orange">4 </span>| <span style="background-color:lightblue">1 </span>| <span style="background-color:yellow">1 </span>| 1 | 1 | 0 |
 <span style="background-color:lightgreen">7 </span>| <span style="background-color:orange">6 </span>| <span style="background-color:lightblue">6 </span>| <span style="background-color:yellow">2 </span>| 1 | 0 | 0 |
 <span style="background-color:lightgreen">6 </span>| <span style="background-color:orange">2 </span>| <span style="background-color:lightblue">7 </span>| <span style="background-color:yellow">3 </span>| 1 | 1 | 0 |
 <span style="background-color:lightgreen">5 </span>| <span style="background-color:orange">3 </span>| <span style="background-color:lightblue">2 </span>| <span style="background-color:yellow">5 </span>| 0 | 0 | 1 |
 <span style="background-color:lightgreen">2 </span>| <span style="background-color:orange">5 </span>| <span style="background-color:lightblue">3 </span>| <span style="background-color:yellow">7 </span>| 0 | 0 | 1 |
 <span style="background-color:lightgreen">1 </span>| <span style="background-color:orange">7 </span>| <span style="background-color:lightblue">5 </span>| <span style="background-color:yellow">4 </span>| 0 | 0 | 1 |


|set_1| set_2| set_3|
|--|--|--|
|<span style="background-color:lightgreen">3</span>|<span style="background-color:lightgreen">3</span>|<span style="background-color:lightgreen">1</span>|
|<span style="background-color:orange">1</span>|<span style="background-color:orange">1</span>|<span style="background-color:orange">3</span>|
|<span style="background-color:lightblue">1</span>|<span style="background-color:lightblue">1</span>|<span style="background-color:lightblue">2</span>|
|<span style="background-color:yellow">1</span>|<span style="background-color:yellow">1</span>|<span style="background-color:yellow">4</span>|

You can notice that set_1 and set_2 signatures are very similar and signature of set_3 dissimilar with set_1 and set_2.

{% highlight r %}
jaccard_signatures <-  function(c1, c2) {
  column_intersect <- sum(c1 == c2)
  column_union <- length(c1)
  return(column_intersect / column_union)
}
print(jaccard_signatures(sm[, 1], sm[, 2]))
{% endhighlight %}



{% highlight text %}
## [1] 1
{% endhighlight %}



{% highlight r %}
print(jaccard_signatures(sm[, 1], sm[, 3]))
{% endhighlight %}



{% highlight text %}
## [1] 0
{% endhighlight %}
Intuition is very straighforward. Let's look down the permuted columns $$c_1$$ and $$c_2$$ until we detect **1**.  

  * If in both columns we find ones - (1, 1), then $$h(c_1) = h(c_2)$$. 
  * In case (0, 1) or (1, 0) $$h(c_1) \neq h(c_2)$$. So the probability over all permutations of rows that $$h(c_1) = h(c_2)$$ is the same as $$J(c_1, c_2)$$.  

Moreover there exist theoretical guaranties for estimation of Jaccard similarity: for any constant $$\varepsilon > 0$$ there is a constant $$k = O(1/\varepsilon^2)$$
such that the expected error of the estimate is at most $$\varepsilon$$. 

### Implementation and bottlenecks
Suppose **input-matrix** is very big, say ```1e9``` rows. It is quite hard computationally to permute 1 billion rows. Plus you need to store these entries and access these values. It is common to use following scheme instead:  

  * Pick $$N$$ independent hash functions $$h_i(c)$$ instead of $$N$$ premutations, $$i = 1..N$$.  
  * For each column $$c$$ and each hash function $$h_i$$, keep a "slot" $$M(i, c)$$.  
  * $$M(i, c)$$ will become the smallest value of $$h_i(r)$$ for which column $$c$$ has 1 in row $$r$$. I.e., $$h_i(r)$$ gives order of rows for $$i^{th}$$ permutation.  

So we end up with following **ALGORITHM(1)** from excellent [Mining of Massive Datasets](http://www.mmds.org) book:
{% highlight text %}
for each row r do begin
  for each hash function hi do
    compute hi (r);
  for each column c
    if c has 1 in row r
      for each hash function hi do
        if hi(r) is smaller than M(i, c) then
          M(i, c) := hi(r);
end;
{% endhighlight %}
I **highly recommend** to watch video about minhashing from Stanford [Mining Massive Datasets](https://class.coursera.org/mmds-001) course.

<div align="center"><iframe width="854" height="510" src="http://www.youtube.com/embed/pqZh-Uu9VSk" frameborder="0" allowfullscreen></iframe></div>


## Summary
Let's summarize what we have learned from first part of tutorial:  

* We can construct **input-matrix** from given list of sets. But actually we didn't exploit the fact, that **input-matrix** is **very sparse** and construct it as R's regular dense matrix. It is very computationally and RAM inefficient. 
* We can construct **dense** signature-matrix from **input-matrix**. But we only implemented algorithm that is based on permutations and also not very efficient. 
* We understand **theorethical guaranties** of our algorithm. They are proportional to number of **independent** hash functions we will pick. But how will we actually construct this family of functions? How can we efficiently increase number of functions in our family when needed?
* Our **signature-matrix** has small **fixed** number of rows. Each column represents input set and $$J(c_1, c_2)$$ ~ $$J(set_1, set_2)$$. But we **still have $$O(n^2)$$ complexity**, because we need to compair each pair to find duplicate candidates.

In the next posts I will describe how to efficently construct and store **input-matrix** in **sparse** format.
Then we will discuss how to **construct family of hash functions**. After that we will implement **fast vectorized** version of **ALGORITHM(1)**. And finally we will see how to use **Locality Sensitive Hashing** to determine candidate pairs for similar sets in $$O(n)$$ time. Stay tuned!
