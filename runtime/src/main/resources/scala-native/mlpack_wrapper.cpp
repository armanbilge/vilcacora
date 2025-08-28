#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/convolution.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <mlpack/methods/ann/layer/softmax.hpp>

#include <string>
#include <vector>
#include <stdexcept>
#include <tuple>
#include <cmath> // For std::floor

// --- PART 1: C++ STRUCTS MATCHING THE SCALA NATIVE FFI ---
// These structs are the C++ counterparts to your CStruct definitions.
using FConvolution  = mlpack::ConvolutionType<
                        mlpack::NaiveConvolution<mlpack::ValidConvolution>,
                        mlpack::NaiveConvolution<mlpack::FullConvolution>,
                        mlpack::NaiveConvolution<mlpack::ValidConvolution>,
                        arma::fmat>;

using FMaxPooling   = mlpack::MaxPoolingType<arma::fmat>;
using FSoftmax      = mlpack::SoftmaxType<arma::fmat>;
struct OpResult {
    double* data;
    size_t height;
    size_t width;
    size_t channels;
};
struct FOpResult {
    float* data;
    size_t height;
    size_t width;
    size_t channels;
};
struct ConvolutionParameters {
    size_t num_output_maps;
    size_t kernel_height;
    size_t kernel_width;
    size_t stride_height;
    size_t stride_width;
    int auto_pad; // 0 for VALID, 1 for SAME
};


struct MaxPoolingParameters {
    size_t kernel_height;
    size_t kernel_width;
    size_t stride_height;
    size_t stride_width;
};


// The 'extern "C"' block prevents C++ name mangling, which is essential
// for the Scala Native linker to find these functions.
extern "C" {

// --- PART 2: IMPLEMENTATION OF YOUR @extern OBJECT ---

// --- Factory and Memory Management Functions ---

ConvolutionParameters* create_convolution_parameters(
    size_t num_output_maps, size_t kernel_height, size_t kernel_width,
    size_t stride_height, size_t stride_width, int auto_pad) {
    ConvolutionParameters* p = new ConvolutionParameters();
    p->num_output_maps = num_output_maps;
    p->kernel_height = kernel_height;
    p->kernel_width = kernel_width;
    p->stride_height = stride_height;
    p->stride_width = stride_width;
    p->auto_pad = auto_pad;
    return p;
}

void free_convolution_parameters(ConvolutionParameters* params) {
    delete params;
}

MaxPoolingParameters* create_maxpooling_parameters(
    size_t kernel_height, size_t kernel_width,
    size_t stride_height, size_t stride_width) {
    MaxPoolingParameters* p = new MaxPoolingParameters();
    p->kernel_height = kernel_height;
    p->kernel_width = kernel_width;
    p->stride_height = stride_height;
    p->stride_width = stride_width;
    return p;
}

void free_maxpooling_parameters(MaxPoolingParameters* params) {
    delete params;
}

void free_op_result(OpResult* result) {
    if (result) {
        delete[] result->data; // Free the data array first
        delete result;        // Then free the struct itself
    }
}
void free_Fop_result(FOpResult* result) {
    if (result) {
        delete[] result->data; // Free the data array first
        delete result;        // Then free the struct itself
    }
}
void free_converted_memory(double* ptr) {
    delete[] ptr;
}
void F_free_converted_memory(float* ptr) {
    delete[] ptr;
}


// --- Data Layout Conversion Helpers ---

// Converts an NCHW (row-major) tensor to MLPack's format.
double* convert_row_to_column_major(
    const double* input_ptr, size_t height, size_t width, size_t channels) {
    size_t total_elements = height * width * channels;
    double* output_ptr = new double[total_elements];
    size_t k = 0;
    for (size_t c = 0; c < channels; ++c) {
        for (size_t w = 0; w < width; ++w) {
            for (size_t h = 0; h < height; ++h) {
                // Read from row-major index, write to column-major index
                output_ptr[k++] = input_ptr[c * (height * width) + h * width + w];
            }
        }
    }
    return output_ptr;
}
float* F_convert_row_to_column_major(
    const float* input_ptr, size_t height, size_t width, size_t channels) {
    size_t total_elements = height * width * channels;
    float* output_ptr = new float[total_elements];
    size_t k = 0;
    for (size_t c = 0; c < channels; ++c) {
        for (size_t w = 0; w < width; ++w) {
            for (size_t h = 0; h < height; ++h) {
                // Read from row-major index, write to column-major index
                output_ptr[k++] = input_ptr[c * (height * width) + h * width + w];
            }
        }
    }
    return output_ptr;
}
// Converts MLPack's output back to NCHW (row-major).
double* convert_column_to_row_major(
    const double* input_ptr, size_t height, size_t width, size_t channels) {
    size_t total_elements = height * width * channels;
    double* output_ptr = new double[total_elements];
    size_t k = 0;
    for (size_t c = 0; c < channels; ++c) {
        for (size_t w = 0; w < width; ++w) {
            for (size_t h = 0; h < height; ++h) {
                // Read from column-major index, write to row-major index
                output_ptr[c * (height * width) + h * width + w] = input_ptr[k++];
            }
        }
    }
    return output_ptr;
}
float* F_convert_column_to_row_major(
    const float* input_ptr, size_t height, size_t width, size_t channels) {
    size_t total_elements = height * width * channels;
    float* output_ptr = new float[total_elements];
    size_t k = 0;
    for (size_t c = 0; c < channels; ++c) {
        for (size_t w = 0; w < width; ++w) {
            for (size_t h = 0; h < height; ++h) {
                // Read from column-major index, write to row-major index
                output_ptr[c * (height * width) + h * width + w] = input_ptr[k++];
            }
        }
    }
    return output_ptr;
}

double* convert_oihw_to_mlpack_format(
    const double* kernel_ptr, size_t output_channels, size_t input_channels,
    size_t kernel_height, size_t kernel_width) {
    size_t total_elements = output_channels * input_channels * kernel_height * kernel_width;
    double* output_ptr = new double[total_elements];
    size_t k = 0;
    size_t kernel_plane_size = kernel_height * kernel_width;

    for (size_t o = 0; o < output_channels; ++o) {
        for (size_t i = 0; i < input_channels; ++i) {
            
            for (size_t w = 0; w < kernel_width; ++w) {   // width first
                for (size_t h = 0; h < kernel_height; ++h) { // then height
                    size_t row_major_idx = o * (input_channels * kernel_plane_size) +
                                           i * (kernel_plane_size) +
                                           h * kernel_width + w;
                    output_ptr[k++] = kernel_ptr[row_major_idx];
                }
            }
        }
    }
    return output_ptr;
}

float* F_convert_oihw_to_mlpack_format(
    const float* kernel_ptr, size_t output_channels, size_t input_channels,
    size_t kernel_height, size_t kernel_width) {
    size_t total_elements = output_channels * input_channels * kernel_height * kernel_width;
    float* output_ptr = new float[total_elements];
    size_t k = 0;
    size_t kernel_plane_size = kernel_height * kernel_width;

    for (size_t o = 0; o < output_channels; ++o) {
        for (size_t i = 0; i < input_channels; ++i) {
            
            for (size_t w = 0; w < kernel_width; ++w) {   // width first
                for (size_t h = 0; h < kernel_height; ++h) { // then height
                    size_t row_major_idx = o * (input_channels * kernel_plane_size) +
                                           i * (kernel_plane_size) +
                                           h * kernel_width + w;
                    output_ptr[k++] = kernel_ptr[row_major_idx];
                }
            }
        }
    }
    return output_ptr;
}
// Helper to calculate output dimension based on the standard formula
size_t calculate_output_size(size_t inputSize, size_t kernelSize, size_t stride, size_t padA, size_t padB) {
    return static_cast<size_t>(std::floor(static_cast<double>(inputSize - kernelSize + padA + padB) / stride)) + 1;
}

// --- Core Operation Functions ---

OpResult* perform_convolution(
    const ConvolutionParameters* params,
    const double* input_ptr, size_t inputHeight, size_t inputWidth, size_t inputChannels,
    const double* kernel_ptr, const double* bias_ptr, int use_bias_int)
{
    bool use_bias = (use_bias_int != 0);
    arma::mat inputData(const_cast<double*>(input_ptr), inputHeight * inputWidth * inputChannels, 1, true);
    size_t totalKernelElements = params->num_output_maps * inputChannels * params->kernel_height * params->kernel_width;
    arma::mat kernelWeights(const_cast<double*>(kernel_ptr), totalKernelElements, 1, true);
    
    arma::mat biasData;
    if (use_bias) {
        biasData = arma::mat(const_cast<double*>(bias_ptr), params->num_output_maps, 1, true);
    }
    
    mlpack::FFN<> model;
    std::string padding_type = (params->auto_pad == 1) ? "same" : "none";
    
    model.Add<mlpack::Convolution>(params->num_output_maps, params->kernel_width, params->kernel_height,
        params->stride_width, params->stride_height, 0, 0, padding_type, use_bias);
    
    model.InputDimensions() = {inputWidth, inputHeight, inputChannels};

    size_t totalParams = kernelWeights.n_elem + (use_bias ? biasData.n_elem : 0);
    model.Parameters().set_size(totalParams, 1);
    model.Parameters().submat(0, 0, kernelWeights.n_elem - 1, 0) = kernelWeights;
    if (use_bias) {
        model.Parameters().submat(kernelWeights.n_elem, 0, totalParams - 1, 0) = biasData;
    }
    
    arma::mat outputArma;
    model.Predict(inputData, outputArma);
    
    // Create the result struct on the heap and copy data to a new buffer
    OpResult* result = new OpResult();
    result->channels = params->num_output_maps;
    if (params->auto_pad == 1) { // SAME
        result->height = inputHeight;
        result->width = inputWidth;
    } else { // VALID
        result->height = calculate_output_size(inputHeight, params->kernel_height, params->stride_height, 0, 0);
        result->width = calculate_output_size(inputWidth, params->kernel_width, params->stride_width, 0, 0);
    }
    
    result->data = new double[outputArma.n_elem];
    std::memcpy(result->data, outputArma.memptr(), outputArma.n_elem * sizeof(double));
    
    return result;
}
FOpResult* F_perform_convolution(
    const ConvolutionParameters* params,
    const float* input_ptr, size_t inputHeight, size_t inputWidth, size_t inputChannels,
    const float* kernel_ptr, const float* bias_ptr, int use_bias_int)
{
    bool use_bias = (use_bias_int != 0);
    arma::fmat inputData(const_cast<float*>(input_ptr), inputHeight * inputWidth * inputChannels, 1, true);
    size_t totalKernelElements = params->num_output_maps * inputChannels * params->kernel_height * params->kernel_width;
    arma::fmat kernelWeights(const_cast<float*>(kernel_ptr), totalKernelElements, 1, true);
    
    arma::fmat biasData;
    if (use_bias) {
        biasData = arma::fmat(const_cast<float*>(bias_ptr), params->num_output_maps, 1, true);
    }
    
    mlpack::FFN<mlpack::NegativeLogLikelihood,
            mlpack::RandomInitialization,
            arma::fmat> model;
    std::string padding_type = (params->auto_pad == 1) ? "same" : "none";
    
    model.Add<FConvolution>(params->num_output_maps, params->kernel_width, params->kernel_height,
        params->stride_width, params->stride_height, 0, 0, padding_type, use_bias);
    
    model.InputDimensions() = {inputWidth, inputHeight, inputChannels};

    size_t totalParams = kernelWeights.n_elem + (use_bias ? biasData.n_elem : 0);
    model.Parameters().set_size(totalParams, 1);
    model.Parameters().submat(0, 0, kernelWeights.n_elem - 1, 0) = kernelWeights;
    if (use_bias) {
        model.Parameters().submat(kernelWeights.n_elem, 0, totalParams - 1, 0) = biasData;
    }
    
    arma::fmat outputArma;
    model.Predict(inputData, outputArma);
    
    // Create the result struct on the heap and copy data to a new buffer
    FOpResult* result = new FOpResult();
    result->channels = params->num_output_maps;
    if (params->auto_pad == 1) { // SAME
        result->height = inputHeight;
        result->width = inputWidth;
    } else { // VALID
        result->height = calculate_output_size(inputHeight, params->kernel_height, params->stride_height, 0, 0);
        result->width = calculate_output_size(inputWidth, params->kernel_width, params->stride_width, 0, 0);
    }
    
    result->data = new float[outputArma.n_elem];
    std::memcpy(result->data, outputArma.memptr(), outputArma.n_elem * sizeof(float));
    
    return result;
}

OpResult* perform_maxpooling(
    const MaxPoolingParameters* params,
    const double* input_ptr, size_t inputHeight, size_t inputWidth, size_t inputChannels)
{
    arma::mat inputData(const_cast<double*>(input_ptr), inputHeight * inputWidth * inputChannels, 1, true);
    
    mlpack::FFN<> model;
    model.Add<mlpack::MaxPooling>(params->kernel_width, params->kernel_height, params->stride_width, params->stride_height);
    model.InputDimensions() = {inputWidth, inputHeight, inputChannels};

    arma::mat outputArma;
    model.Predict(inputData, outputArma);

    OpResult* result = new OpResult();
    result->channels = inputChannels;
    result->height = calculate_output_size(inputHeight, params->kernel_height, params->stride_height, 0, 0);
    result->width = calculate_output_size(inputWidth, params->kernel_width, params->stride_width, 0, 0);

    result->data = new double[outputArma.n_elem];
    std::memcpy(result->data, outputArma.memptr(), outputArma.n_elem * sizeof(double));

    return result;
}
FOpResult* F_perform_maxpooling(
    const MaxPoolingParameters* params,
    const float* input_ptr, size_t inputHeight, size_t inputWidth, size_t inputChannels)
{
    arma::fmat inputData(const_cast<float*>(input_ptr), inputHeight * inputWidth * inputChannels, 1, true);
    
    mlpack::FFN<mlpack::NegativeLogLikelihood,
            mlpack::RandomInitialization,
            arma::fmat> model;
    model.Add<FMaxPooling>(params->kernel_width, params->kernel_height, params->stride_width, params->stride_height);
    model.InputDimensions() = {inputWidth, inputHeight, inputChannels};

    arma::fmat outputArma;
    model.Predict(inputData, outputArma);

    FOpResult* result = new FOpResult();
    result->channels = inputChannels;
    result->height = calculate_output_size(inputHeight, params->kernel_height, params->stride_height, 0, 0);
    result->width = calculate_output_size(inputWidth, params->kernel_width, params->stride_width, 0, 0);

    result->data = new float[outputArma.n_elem];
    std::memcpy(result->data, outputArma.memptr(), outputArma.n_elem * sizeof(float));

    return result;

}
OpResult* perform_softmax(
    const double* input_ptr, size_t input_height, size_t input_width, size_t input_channels) {
    
    size_t total_elements = input_height * input_width * input_channels;
    arma::mat inputData(const_cast<double*>(input_ptr), total_elements, 1, true);
    
    mlpack::FFN<> model;
    model.Add<mlpack::SoftmaxType<>>();
    model.InputDimensions() = {total_elements};
    
    arma::mat outputArma;
    model.Predict(inputData, outputArma);
    
    // Create the result struct on the heap and copy data to a new buffer
    OpResult* result = new OpResult();
    result->height = input_height;
    result->width = input_width; 
    result->channels = input_channels;
    result->data = new double[outputArma.n_elem];
    std::memcpy(result->data, outputArma.memptr(), outputArma.n_elem * sizeof(double));
    
    return result;
}

FOpResult* F_perform_softmax(
    const float* input_ptr, size_t input_height, size_t input_width, size_t input_channels) {
    
    size_t total_elements = input_height * input_width * input_channels;
    arma::fmat inputData(const_cast<float*>(input_ptr), total_elements, 1, true);
    
    mlpack::FFN<mlpack::NegativeLogLikelihood, mlpack::RandomInitialization, arma::fmat> model;
    model.Add<FSoftmax>();
    model.InputDimensions() = {total_elements};
    
    arma::fmat outputArma;
    model.Predict(inputData, outputArma);
    
    // Create the result struct on the heap and copy data to a new buffer
    FOpResult* result = new FOpResult();
    result->height = input_height;
    result->width = input_width;
    result->channels = input_channels;
    result->data = new float[outputArma.n_elem];
    std::memcpy(result->data, outputArma.memptr(), outputArma.n_elem * sizeof(float));
    
    return result;
}
} // extern "C"