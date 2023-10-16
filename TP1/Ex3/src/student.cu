/*
* TP 1 - Premiers pas en CUDA
* --------------------------
* Ex 3: Filtre d'images sepia
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"


namespace IMAC
{
	__global__ void filter_img(uchar *dev_input, uchar * const dev_output){
		

		int id = (threadIdx.x + blockIdx.x * blockDim.x) * 3;
		const uchar inR = dev_input[id];
		const uchar inG = dev_input[id + 1];
		const uchar inB = dev_input[id + 2];
		dev_output[id] = static_cast<uchar>( fminf( 255.f, ( inR * .393f + inG * .769f + inB * .189f ) ) );
		dev_output[id + 1] = static_cast<uchar>( fminf( 255.f, ( inR * .349f + inG * .686f + inB * .168f ) ) );
		dev_output[id + 2] = static_cast<uchar>( fminf( 255.f, ( inR * .272f + inG * .534f + inB * .131f ) ) );
	}


	void studentJob(const std::vector<uchar> &input, const uint width, const uint height, std::vector<uchar> &output)
	{
		ChronoGPU chrGPU;

		// 2 arrays for GPU
		uchar *dev_input = NULL;
		uchar *dev_output = NULL;
		
		/// TODOOOOOOOOOOOOOO
		const uint size = width * height;
		const size_t bytes = input.size() * sizeof(uchar);
		std::cout << "taille image : " << size << std::endl;
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();

		//Allocation
		cudaMalloc((void **) &dev_input, bytes);
		cudaMalloc((void **) &dev_output, bytes);

		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
	
		//Transfer
		chrGPU.start();
		// Copy data from host to device
		cudaMemcpy(dev_input, input.data(), bytes, cudaMemcpyHostToDevice);

		chrGPU.stop();
		std::cout 	<< "-> Transfer : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Launch Kernel
		filter_img<<<(width/32 + 1) * height/32, 1024>>>(dev_input, dev_output);
		
		// Copy data from device to host (output array)
		cudaMemcpy(output.data(), dev_output, bytes, cudaMemcpyDeviceToHost);

		//Free
		chrGPU.start();

		cudaFree(dev_input);
		cudaFree(dev_output);

		chrGPU.stop();
		std::cout 	<< "-> Free : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;


	}
}
