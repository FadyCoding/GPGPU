/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 2: Addition de vecteurs
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{
	__global__ void sumArraysCUDA(const int n, const int *const dev_a, const int *const dev_b, int *const dev_res)
	{
		/// TODO
		// grimDim.x
		int x = gridDim.x*blockDim.x;
		int max_op = 1+ (n / x);
		int idx = blockDim.x * blockIdx.x + threadIdx.x;

		for (int i = 0; i < max_op; i++)
		{
			int current = idx * max_op + i;
			
			if (current < n){
				dev_res[current] = dev_a[current] + dev_b[current];
			}
			
		}
		
	
	}

    void studentJob(const int size, const int *const a, const int *const b, int *const res)
	{
		ChronoGPU chrGPU;

		// 3 arrays for GPU
		int *dev_a = NULL;
		int *dev_b = NULL;
		int *dev_res = NULL;

		// Allocate arrays on device (input and ouput)
		const size_t bytes = size * sizeof(int);
		std::cout << "taille vecteur : " << size << std::endl;
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();
		
		/// TODO
		cudaMalloc((void **) &dev_a, bytes);
		cudaMalloc((void **) &dev_b, bytes);
		cudaMalloc((void **) &dev_res, bytes);

		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Copy data from host to device (input arrays) 
		/// TODO
		cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_res, res, bytes, cudaMemcpyHostToDevice);

		// Launch kernel
		/// TODO
		sumArraysCUDA<<<1024, 256>>>(size, dev_a, dev_b, dev_res);

		// Copy data from device to host (output array)  
		/// TODO
		cudaMemcpy(res, dev_res, bytes, cudaMemcpyDeviceToHost);

		// Free arrays on device
		/// TODO
		cudaFree(dev_a);
		cudaFree(dev_b);
		cudaFree(dev_res);
		
	}
}

