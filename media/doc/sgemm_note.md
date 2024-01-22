### Environment

* ubuntu 22.04
* Cuda 12.0
* device RTX3080
 
### kernel1：Computing multiple elements in a single thread

&emsp;&emsp;A result block in matrix C is obtained by the dot product of the row of the matrix block in A and the column of the matrix block in B, so that each `block` in the device is responsible for the calculation of a block in matrix C, so is need to read the blocks in A and B from the global memory along `dotIndex` to the shared memory for calculation.Notice the boundary conditions: when the shared memory size is larger than the matrix block, the larger part needs to be set to 0.

&emsp;&emsp;Here set `BM` and `BN` to be 128, and `BK` to be 8.

![](image/eachBlockCompute.png)
*<center>fig.1 The job of each block</center>*

&emsp;&emsp;Each `block` is responsible for the calculation of a block in the matrix C:

```C++
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BM;
```

&emsp;&emsp;Load the global memory data in to the shared memory, take A as an example, each thread loads in an element, all threads load a batch of elements at a time and loads `SMEMLoadRowStrideA` lines in each batch:


![](image/loadToSMEM.png)*<center>fig.2 Load from global memory to shared memory</center>*

&emsp;&emsp;Create a load index as follows:

```C++
    int SMEMLoadARow = threadIdx.x / BK;
    int SMEMLoadACol = threadIdx.x % BK;
    int SMEMLoadAStride = blockDim.x / BK;
    int SMEMLoadBRow = threadIdx.x / BN;
    int SMEMLoadBCol = threadIdx.x % BN;
    int SMEMLoadBStride = blockDim.x / BN;
```

&emsp;&emsp;Multiplication between matrix blocks is considered after the matrix blocks have been loaded into shared memory. Each thread is responsible for the calculation of `TM` * `TN` elements. The outer product of the column vector of length `TM` in `sharedA` and the row vector of length `TN` in `sharedB` is performed, and then the result of the outer product is added along `innerIndex` to obtain a size of` TM` * `TN` matrix block. Here `TM` and `TN` are taken as 8.

![](image/eachThreadCompute.png)
*<center>fig.3 The job of each thread</center>*

&emsp;&emsp;Finally, write the calculation result back to C.

[kernel1 repo](https://github.com/xcwang1999/gemm/blob/main/sgemm/src/sgemm_kernel1.cu)

&emsp;&emsp;Performance comparison between kernel1 and cublas.

![](image/kernel1.png)
*<center>fig.4 The performance of kernel1</center>*

### Optimized on the basis of kernel1
- Kernel2: Use float4 vector access
  
&emsp;&emsp;The size of each shared memory sharedA is `BM` * `BK`, here is 128 * 8 = 1024, each thread block has 256 threads, and each thread takes a float4, which can just cover the entire shared memory. The global memory needs to be written to the shared memory through the register. 

&emsp;&emsp;First load the data into the register:

```C++
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
...
    float ldgRegA[4] = {0.0};
    float ldgRegB[4] = {0.0};
    const int SMEMLoadARow = threadIdx.x / (BK / 4);
    const int SMEMLoadACol = threadIdx.x % (BK / 4);
    const int SMEMLoadBRow = threadIdx.x / (BN / 4);
    const int SMEMLoadBCol = threadIdx.x % (BN / 4);
    FETCH_FLOAT4(ldgRegA[0]) =
            FETCH_FLOAT4(A[INDEX(SMEMLoadARow, SMEMLoadACol * 4, K)]);
    FETCH_FLOAT4(ldgRegB[0]) =
           FETCH_FLOAT4(B[INDEX(SMEMLoadBRow, SMEMLoadBCol * 4, N)]);
```
![](image/kernel2.png)
*<center>fig.5 The performance of kernel2</center>*
- Transpose elements in `shardA`

&emsp;&emsp;As shown in Figure 3, the two rows of threads with threadRow = 0 and threadRow = 1 are in the same thread warp, and they will cause a bank conflict in the shared memory. In the process of reading from the register to the shared memory, the value stored in shardA is transposed, and the bank conflict in sharedA is eliminated by a non-coalesced loads.
```C++
    sharedA[INDEX(SMEMLoadACol*4 + 0, SMEMLoadARow, BM)] = ldgRegA[0];
    sharedA[INDEX(SMEMLoadACol*4 + 1, SMEMLoadARow, BM)] = ldgRegA[1];
    sharedA[INDEX(SMEMLoadACol*4 + 2, SMEMLoadARow, BM)] = ldgRegA[2];
    sharedA[INDEX(SMEMLoadACol*4 + 3, SMEMLoadARow, BM)] = ldgRegA[3];
    FETCH_FLOAT4(sharedB[INDEX(SMEMLoadBRow, SMEMLoadBCol*4, BN)]) =
                FETCH_FLOAT4(ldgRegB[0]);
```
![](image/kernel3.png)
*<center>fig.6 The performance of kernel3</center>*
- Reduce bank conflicts in loading `sharedB`


&emsp;&emsp;As shown in Figure 5, each square represents a float4, and each thread reads two adjacent float4s successively. In this case, threads 0, 4, 8, 12... will simultaneously access bank0, threads 1, 5, and 9 , 13... will access bank1 at the same time, causing bank conflicts.

![](image/bankConflictInSharedB.png)
*<center>fig.7 Bank conflict in sharedB</center>*

&emsp;&emsp;The method of reducing bank conflicts is shown in Figure 7. Each thread successively accesses two float4s separated by BK/2. This means that threads 0, 8, 16...will access bank0 at the same time, reducing bank conflicts by half.

![](image/reduceBankConflictInSharedB.png)
*<center>fig.8 Reduce bank conflict in sharedB</center>*

&emsp;&emsp;However, in sharedA, after a transposition, the index interval of threads accessing the same bank exceeds 32 (the number of threads in a warp), so no bank conflict occurs when loading shardA.

![](image/noBankConflictInSharedA.png)
*<center>fig.9 No bank conflict in sharedA</center>*

```C++
    for(int innerIndex=0; innerIndex < BK; innerIndex++){
        FETCH_FLOAT4(vectorOuterProdA[0]) =
                FETCH_FLOAT4(sharedA[INDEX(innerIndex, threadRow * TM + 0, BM)]);
        FETCH_FLOAT4(vectorOuterProdA[4]) =
                FETCH_FLOAT4(sharedA[INDEX(innerIndex, threadRow * TM + 4, BM)]);

        FETCH_FLOAT4(vectorOuterProdB[0]) =
                FETCH_FLOAT4(sharedB[INDEX(innerIndex, threadCol * TN / 2, BN)]);
        FETCH_FLOAT4(vectorOuterProdB[4]) =
                FETCH_FLOAT4(sharedB[INDEX(innerIndex, threadCol * TN / 2 + BN / 2, BN)]);
        for(int rstEachThreadRow=0; rstEachThreadRow < TM; rstEachThreadRow++)
            for(int rstEachThreadCol=0; rstEachThreadCol < TN; rstEachThreadCol++)
                rstEachThread[INDEX(rstEachThreadRow, rstEachThreadCol, TN)] +=
                        vectorOuterProdA[rstEachThreadRow] * vectorOuterProdB[rstEachThreadCol];
    }
    __syncthreads();
```

[kernel4 repo](https://github.com/xcwang1999/gemm/blob/main/sgemm/src/sgemm_kernel4.cu)

&emsp;&emsp;Finally, write the calculation result back to C.
```C++
    for(int resEachThreadRow=0; resEachThreadRow<TM; resEachThreadRow++)
        FETCH_FLOAT4(C[INDEX(threadRow*TM + resEachThreadRow, threadCol*TN/2, N)]) =
                FETCH_FLOAT4(rstEachThread[INDEX(resEachThreadRow, 0, TN)]);

    for(int resEachThreadRow=0; resEachThreadRow<TM; resEachThreadRow++)
        FETCH_FLOAT4(C[INDEX(threadRow*TM + resEachThreadRow, threadCol*TN/2 + BN/2, N)]) =
                FETCH_FLOAT4(rstEachThread[INDEX(resEachThreadRow, 4, TN)]);
```
&emsp;&emsp;Performance comparison between cublas and kernel4.

![](image/kernel4.png)
*<center>fig.10 The performance of kernel4</center>*

&emsp;&emsp;Note: Due to GPU memory alignment, using float4 vector memory access cannot load into a matrix whose dimension is not a multiple of 4, so there is no boundary judgment in kernel2, 3, 4.

### kernel5: Double buffer
&emsp;&emsp;Double buffering needs to open up double the shared memory space. When calculating the first layer of memory, asynchronously read the data that needs to be calculated in the next step into the second layer of memory.

![](image/loopFlow.png)
*<center>fig.11 The work flow of double buffering</center>*

[kernel5 repo](https://github.com/xcwang1999/gemm/blob/main/sgemm/src/sgemm_kernel5.cu)

![](image/kernel5.png)
*<center>fig.12 The performance of kernel3</center>*

&emsp;&emsp;Also due to the float4 vectorize memory access, kernel5 cannot handle matrices with dimensions other than a multiple of 4.

###kernel4: Abandon float4 vectorize access

&emsp;&emsp; In kernel6, I gave up using float4, and instead, a thread continuously reads 4 floats.
```C++
regA[0] = A[INDEX(SMEMLoadARow, SMEMLoadACol* 4 + dotOffset + 0, K)];
regA[1] = A[INDEX(SMEMLoadARow, SMEMLoadACol* 4 + dotOffset + 1, K)];
regA[2] = A[INDEX(SMEMLoadARow, SMEMLoadACol* 4 + dotOffset + 2, K)];
regA[3] = A[INDEX(SMEMLoadARow, SMEMLoadACol* 4 + dotOffset + 3, K)];
regB[0] = B[INDEX(SMEMLoadBRow + dotOffset, SMEMLoadBCol* 4 + 0, N)];
regB[1] = B[INDEX(SMEMLoadBRow + dotOffset, SMEMLoadBCol* 4 + 1, N)];
regB[2] = B[INDEX(SMEMLoadBRow + dotOffset, SMEMLoadBCol* 4 + 2, N)];
regB[3] = B[INDEX(SMEMLoadBRow + dotOffset, SMEMLoadBCol* 4 + 3, N)];
```

&emsp;&emsp;The boundary judgment condition is added, so that kernel6 can calculate any dimension matrix, but the performance has declined.

![](image/kernel6_1024.png)
*<center>fig.11 The performance of kernel6</center>*

![](image/kernel6_1111.png)
*<center>fig.12 The performance of kernel6</center>*

I don't think it's wise.

[kernel6 repo](https://github.com/xcwang1999/gemm/blob/main/sgemm/src/sgemm_kernel6.cu)


### Kernel7 :Dealing with memory alignment issues with mallocPitch

&emsp;&emsp;I use mallocPitch to allocate two-dimensional, aligned memory for matrices, so that matrices of any dimension can be calculated, and they can also be accessed with float4 vectorization。


[kernel7 repo](https://github.com/xcwang1999/gemm/blob/main/sgemm/src/sgemm_kernel7.cu)

![](image/kernel7_1024.png)
*<center>fig.12 The performance of kernel7</center>*

![](image/kernel7_1111.png)
*<center>fig.13 The performance of kernel7</center>*

Please correct me if I am wrong.