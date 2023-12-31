 /*
* TP 3 - Réduction CUDA
* --------------------------
* Mémoire paratagée, synchronisation, optimisation
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"

namespace IMAC
{
	// ==================================================== EX 1
    __global__
    void maxReduce_ex1(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		// shared memory array
		extern __shared__ uint shared_data[];

		// Indexes 
		int t_id = threadIdx.x;
		int id = blockDim.x * blockIdx.x + t_id;

		// Load shared memory from global memory
		if(id < size)
		{
			shared_data[t_id] = dev_array[id];
		}
		__syncthreads();

		for(unsigned int s = 1; s + t_id < blockDim.x; t_id<<=1, s<<=1)
		{
			shared_data[t_id] = umax(shared_data[t_id], shared_data[s+t_id]);
			__syncthreads();
		}

		if (t_id == 0)
		{
			dev_partialMax[blockIdx.x] = shared_data[0];
		}
		

	}

	    __global__
    void maxReduce_ex2(const uint *const dev_array, const uint size, uint *const dev_partialMax)
	{
		// shared memory array
		extern __shared__ uint shared_data[];

		// Indexes 
		int t_id = threadIdx.x;
		int id = blockDim.x * blockIdx.x + t_id;

		// Load shared memory from global memory
		if(id < size)
		{
			shared_data[t_id] = dev_array[id];
		}
		__syncthreads();

		for (unsigned int s=blockDim.x/2; s>0; s>>=1)
		{
			if (t_id< s)
			{
				shared_data[t_id] += shared_data[t_id + s];
			}
			__syncthreads();
		}
		
		if (t_id == 0)
		{
			dev_partialMax[blockIdx.x] = shared_data[0];
		}
	}

	void studentJob(const std::vector<uint> &array, const uint resCPU /* Just for comparison */, const uint nbIterations)
    {
		uint *dev_array = NULL;
        const size_t bytes = array.size() * sizeof(uint);

		// Allocate array on GPU
		HANDLE_ERROR( cudaMalloc( (void**)&dev_array, bytes ) );
		// Copy data from host to device
		HANDLE_ERROR( cudaMemcpy( dev_array, array.data(), bytes, cudaMemcpyHostToDevice ) );

		std::cout << "Test with " << nbIterations << " iterations" << std::endl;

		std::cout << "========== Ex 1 " << std::endl;
		uint res1 = 0; // result
		// Launch reduction and get timing
		float2 timing1 = reduce<KERNEL_EX1>(nbIterations, dev_array, array.size(), res1);
		
        std::cout << " -> Done: ";
        printTiming(timing1);
		compare(res1, resCPU); // Compare results

		std::cout << "========== Ex 2 " << std::endl;
		uint res2 = 0; // result
		// Launch reduction and get timing
		float2 timing2 = reduce<KERNEL_EX2>(nbIterations, dev_array, array.size(), res2);
		
        std::cout << " -> Done: ";
        printTiming(timing2);
		compare(res2, resCPU);

		std::cout << "========== Ex 3 " << std::endl;
		uint res3 = 0; // result
		// Launch reduction and get timing
		float2 timing3 = reduce<KERNEL_EX3>(nbIterations, dev_array, array.size(), res3);
		
        std::cout << " -> Done: ";
        printTiming(timing3);
		compare(res3, resCPU);

		std::cout << "========== Ex 4 " << std::endl;
		uint res4 = 0; // result
		// Launch reduction and get timing
		float2 timing4 = reduce<KERNEL_EX4>(nbIterations, dev_array, array.size(), res4);
		
        std::cout << " -> Done: ";
        printTiming(timing4);
		compare(res4, resCPU);

		std::cout << "========== Ex 5 " << std::endl;
		uint res5 = 0; // result
		// Launch reduction and get timing
		float2 timing5 = reduce<KERNEL_EX5>(nbIterations, dev_array, array.size(), res5);
		
        std::cout << " -> Done: ";
        printTiming(timing5);
		compare(res5, resCPU);

		// Free array on GPU
		cudaFree( dev_array );
    }

	void printTiming(const float2 timing)
	{
		std::cout << ( timing.x < 1.f ? 1e3f * timing.x : timing.x ) << " us on device and ";
		std::cout << ( timing.y < 1.f ? 1e3f * timing.y : timing.y ) << " us on host." << std::endl;
	}

    void compare(const uint resGPU, const uint resCPU)
	{
		if (resGPU == resCPU)
		{
			std::cout << "Well done ! " << resGPU << " == " << resCPU << " !!!" << std::endl;
		}
		else
		{
			std::cout << "You failed ! " << resGPU << " != " << resCPU << " !!!" << std::endl;
		}
	}
}
