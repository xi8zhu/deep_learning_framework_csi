# 基础知识
#### SGCS和GCS的区别
+ GCS: Generalized Cosine Similarity
$$

$$
+ SGCS: Squared Generalized Cosine Similarity
$$
SGCS(W, W') = \frac{1}{N_{sample}N_{sb}}\sum_{i=1}^{N_{sample}}\sum_{k=1}^{N_{sb}}(\frac{||w_{k,i}^Hw'_{k,i}||_2}{||w_{k,i}||_2||w'_{k,i}||_2})^2
$$