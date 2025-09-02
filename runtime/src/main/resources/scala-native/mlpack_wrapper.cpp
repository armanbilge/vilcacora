#include <mlpack/core.hpp>
#include <algorithm>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <iostream>
#include <cmath>

using FConvolution = mlpack::ConvolutionType<
    mlpack::NaiveConvolution<mlpack::ValidConvolution>,
    mlpack::NaiveConvolution<mlpack::FullConvolution>,
    mlpack::NaiveConvolution<mlpack::ValidConvolution>,
    arma::fmat>;

using FMaxPooling = mlpack::MaxPoolingType<arma::fmat>;

extern "C" {


void F_perform_convolution_direct(
    // Input parameters 
    size_t num_output_maps, size_t kernel_height, size_t kernel_width,
    size_t stride_height, size_t stride_width, int auto_pad, int use_bias,
    
    // Data pointers
    const float* input_ptr, size_t input_height, size_t input_width, size_t input_channels,
    const float* kernel_ptr, const float* bias_ptr,
    
    // Direct output - Scala allocates, C++ writes directly
    float* output_ptr, size_t* output_height, size_t* output_width, size_t* output_channels)
{
    // Create input VIEW (zero copy)
    arma::fmat inputData(const_cast<float*>(input_ptr), 
                        input_height * input_width * input_channels, 1, false, false);

    // Create layer directly
    FConvolution convLayer(num_output_maps, kernel_width, kernel_height,
                          stride_width, stride_height, 0, 0,
                          (auto_pad == 1 ? "same" : "valid"), use_bias != 0);

    convLayer.InputDimensions() = {input_height, input_width, input_channels};
    convLayer.ComputeOutputDimensions();

    // Set weights/bias directly (zero allocation)
    arma::fcube weightCube(const_cast<float*>(kernel_ptr), 
                          kernel_width, kernel_height, 
                          num_output_maps * input_channels, false, false);
    convLayer.Weight() = weightCube;
    
    if (use_bias) {
        arma::fmat biasMatrix(const_cast<float*>(bias_ptr), num_output_maps, 1, false, false);
        convLayer.Bias() = biasMatrix;
    }

    // Get output dimensions
    const auto& outputDims = convLayer.OutputDimensions();
    *output_height = outputDims[0];
    *output_width = outputDims[1]; 
    *output_channels = outputDims[2];
    
    // Create output VIEW directly into Scala's pre-allocated memory (zero copy)
    size_t total_elements = outputDims[0] * outputDims[1] * outputDims[2];
    arma::fmat outputView(output_ptr, total_elements, 1, false, false);

    // Perform operation - writes directly to Scala's memory
    convLayer.Forward(inputData, outputView);
}

void perform_convolution_direct(// Input parameters 
    size_t num_output_maps, size_t kernel_height, size_t kernel_width,
    size_t stride_height, size_t stride_width, int auto_pad, int use_bias,
    
    // Data pointers
    const double* input_ptr, size_t input_height, size_t input_width, size_t input_channels,
    const double* kernel_ptr, const double* bias_ptr,
    
    // Direct output - Scala allocates, C++ writes directly
    double* output_ptr, size_t* output_height, size_t* output_width, size_t* output_channels)
    {
    // Create input VIEW (zero copy)
    arma::mat inputData(const_cast<double*>(input_ptr), 
                        input_height * input_width * input_channels, 1, false, false);

    // Create layer directly
    mlpack::Convolution convLayer(num_output_maps, kernel_width, kernel_height,
                          stride_width, stride_height, 0, 0,
                          (auto_pad == 1 ? "same" : "valid"), use_bias != 0);

    convLayer.InputDimensions() = {input_height, input_width, input_channels};
    convLayer.ComputeOutputDimensions();

    // Set weights/bias directly (zero allocation)
    arma::cube weightCube(const_cast<double*>(kernel_ptr), 
                          kernel_width, kernel_height, 
                          num_output_maps * input_channels, false, false);
    convLayer.Weight() = weightCube;
    
    if (use_bias) {
        arma::mat biasMatrix(const_cast<double*>(bias_ptr), num_output_maps, 1, false, false);
        convLayer.Bias() = biasMatrix;
    }

    // Get output dimensions
    const auto& outputDims = convLayer.OutputDimensions();
    *output_height = outputDims[0];
    *output_width = outputDims[1]; 
    *output_channels = outputDims[2];
    
    // Create output VIEW directly into Scala's pre-allocated memory (zero copy)
    size_t total_elements = outputDims[0] * outputDims[1] * outputDims[2];
    arma::mat outputView(output_ptr, total_elements, 1, false, false);

    // Perform operation - writes directly to Scala's memory
    convLayer.Forward(inputData, outputView);
}


void F_perform_maxpooling_direct(
    // Parameters 
    size_t kernel_height, size_t kernel_width, size_t stride_height, size_t stride_width,
    
    // Data pointers  
    const float* input_ptr, size_t input_height, size_t input_width, size_t input_channels,
    
    // Direct output
    float* output_ptr, size_t* output_height, size_t* output_width, size_t* output_channels)
{
    FMaxPooling maxPooling(kernel_width, kernel_height, stride_width, stride_height);
    maxPooling.InputDimensions() = {input_height, input_width, input_channels};
    maxPooling.ComputeOutputDimensions();
    
    const auto& outputDims = maxPooling.OutputDimensions();
    *output_height = outputDims[0];
    *output_width = outputDims[1];
    *output_channels = outputDims[2];

    // Create views (zero copy)
    arma::fmat inputView(const_cast<float*>(input_ptr), 
                        input_height * input_width * input_channels, 1, false, false);
    
    size_t total_elements = outputDims[0] * outputDims[1] * outputDims[2];
    arma::fmat outputView(output_ptr, total_elements, 1, false, false);

    maxPooling.Forward(inputView, outputView);
}

void perform_maxpooling_direct(
    // Parameters
    size_t kernel_height, size_t kernel_width, size_t stride_height, size_t stride_width,
    
    // Data pointers  
    const double* input_ptr, size_t input_height, size_t input_width, size_t input_channels,
    
    // Direct output
    double* output_ptr, size_t* output_height, size_t* output_width, size_t* output_channels)
{
    mlpack::MaxPooling maxPooling(kernel_width, kernel_height, stride_width, stride_height);
    maxPooling.InputDimensions() = {input_height, input_width, input_channels};
    maxPooling.ComputeOutputDimensions();
    
    const auto& outputDims = maxPooling.OutputDimensions();
    *output_height = outputDims[0];
    *output_width = outputDims[1];
    *output_channels = outputDims[2];

    // Create views (zero copy)
    arma::mat inputView(const_cast<double*>(input_ptr), 
                        input_height * input_width * input_channels, 1, false, false);
    
    size_t total_elements = outputDims[0] * outputDims[1] * outputDims[2];
    arma::mat outputView(output_ptr, total_elements, 1, false, false);

    maxPooling.Forward(inputView, outputView);
}

void F_perform_softmax_direct(
    const float* input_ptr, size_t input_size,
    float* output_ptr) // Scala pre-allocates same size as input
{
    // Direct computation - no intermediate allocations
    float max_val = *std::max_element(input_ptr, input_ptr + input_size);
    
    float sum = 0.0f;
    for (size_t i = 0; i < input_size; ++i) {
        output_ptr[i] = std::exp(input_ptr[i] - max_val);
        sum += output_ptr[i];
    }
    
    for (size_t i = 0; i < input_size; ++i) {
        output_ptr[i] /= sum;
    }
}
void perform_softmax_direct(
    const double* input_ptr, size_t input_size,
    double* output_ptr) // Scala pre-allocates same size as input
{
    // Direct computation - no intermediate allocations
    double max_val = *std::max_element(input_ptr, input_ptr + input_size);
    
    double sum = 0.0;
    for (size_t i = 0; i < input_size; ++i) {
        output_ptr[i] = std::exp(input_ptr[i] - max_val);
        sum += output_ptr[i];
    }
    
    for (size_t i = 0; i < input_size; ++i) {
        output_ptr[i] /= sum;
    }

} 
}// extern "C"
