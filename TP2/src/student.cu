/*
* TP 2 - Convolution d'images
* --------------------------
* Mémoire constante et textures
*
* File: student.cu
* Author: Maxime MARIA
*/

#include "student.hpp"
#include "chronoGPU.hpp"

namespace IMAC
{

// ================================================== For image comparison
	std::ostream &operator <<(std::ostream &os, const uchar4 &c)
	{
		os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";  
    	return os; 
	}

	void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b)
	{
		bool error = false;
		if (a.size() != b.size())
		{
			std::cout << "Size is different !" << std::endl;
			error = true;
		}
		else
		{
			for (uint i = 0; i < a.size(); ++i)
			{
				// Floating precision can cause small difference between host and device
				if (	std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2 
					|| std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2)
				{
					std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - " << std::abs(a[i].x - b[i].x) << std::endl;
					error = true;
					break;
				}
			}
		}
		if (error)
		{
			std::cout << " -> You failed, retry!" << std::endl;
		}
		else
		{
			std::cout << " -> Well done!" << std::endl;
		}
	}

	__device__ float clamp(const float val, const float min , const float max) 
	{
		return fminf(max, fmaxf(min, val));
	}

	
	__global__ void convCPU(uchar4 *input, const uint imgWidth, const uint imgHeight, 
					const float *matConv, const uint matSize, 
					uchar4 *output)
	{

		for ( uint y = 0; y < blockDim.y; ++y )
		{
			for ( uint x = 0; x < blockDim.x; ++x ) 
			{
				float3 sum = make_float3(0.f,0.f,0.f);
				
				// Apply convolution
				for ( uint j = 0; j < matSize; ++j ) 
				{
					for ( uint i = 0; i < matSize; ++i ) 
					{
						int dX = x + i - matSize / 2;
						int dY = y + j - matSize / 2;

						// Handle borders
						if ( dX < 0 ) 
							dX = 0;

						if ( dX >= imgWidth ) 
							dX = imgWidth - 1;

						if ( dY < 0 ) 
							dY = 0;

						if ( dY >= imgHeight ) 
							dY = imgHeight - 1;

						const int idMat		= j * matSize + i;
						const int idPixel	= dY * imgWidth + dX;
						sum.x += (float)input[idPixel].x * matConv[idMat];
						sum.y += (float)input[idPixel].y * matConv[idMat];
						sum.z += (float)input[idPixel].z * matConv[idMat];
					}
				}
				const int idOut = y * imgWidth + x;
				output[idOut].x = (uchar)clamp( sum.x, 0.f, 255.f );
				output[idOut].y = (uchar)clamp( sum.y, 0.f, 255.f );
				output[idOut].z = (uchar)clamp( sum.z, 0.f, 255.f );
				output[idOut].w = 255;
			}
		}
	}
	
// ==================================================

    void studentJob(const std::vector<uchar4> &inputImg, // Input image
					const uint imgWidth, const uint imgHeight, // Image size
                    const std::vector<float> &matConv, // Convolution matrix (square)
					const uint matSize, // Matrix size (width or height)
					const std::vector<uchar4> &resultCPU, // Just for comparison
                    std::vector<uchar4> &output // Output image
					)
	{
		/// TODO !!!!!!!!!!!!!!!
		ChronoGPU chrGPU;

		// Arrays for GPU
		uchar4 *dev_input_img = NULL;  
		uchar4 *dev_output_img =NULL;
		float *dev_convMat = NULL;

		const uint img_size = imgWidth * imgHeight;
		const uint matcSize = img_size;
		const size_t bytes = inputImg.size() * sizeof(uchar4);
		std::cout << "taille image : " << img_size << std::endl;
		std::cout 	<< "Allocating input (3 arrays): " 
					<< ( ( 3 * bytes ) >> 20 ) << " MB on Device" << std::endl;
		chrGPU.start();

		// Allocation
		cudaMalloc((void **) &dev_input_img, bytes);
		cudaMalloc((void **) &dev_output_img, bytes);
		cudaMalloc((void**) &dev_convMat, bytes);

		chrGPU.stop();
		std::cout 	<< "-> Done : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;
		
		// Transfer
		chrGPU.start();
		// Data from Host to Device
		cudaMemcpy(dev_input_img, inputImg.data(), bytes, cudaMemcpyHostToDevice);
		chrGPU.stop();
		std::cout 	<< "-> Transfer : " << chrGPU.elapsedTime() << " ms" << std::endl << std::endl;

		// Launch kernel
		convCPU<<<1024, 1024>>>(dev_input_img, imgWidth, imgHeight, dev_convMat, matcSize, dev_output_img);

		// Output. Device to Host
		cudaMemcpy(output.data(), dev_output_img, bytes, cudaMemcpyDeviceToHost);

		// Free
		chrGPU.start();

		cudaFree(dev_input_img);
		cudaFree(dev_output_img);
		cudaFree(dev_convMat);

	}
}
