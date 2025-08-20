#include <vector>
#include <cstring>
#include <memory>

#include "tiny_cnn/tiny_cnn.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

extern "C" {

// Calculate output dimension after conv/pool operation
int calculate_output_size(int input_size, int kernel_size, int stride, int padding) {
    return (input_size + 2 * padding - kernel_size) / stride + 1;
}

// Single inference convolution - NO BATCHING
int conv2d_single_inference(
    const float* input_data,    // Single input tensor
    const float* weights,       // Convolution weights
    const float* bias,          // Bias values
    float* output,              // Output buffer
    // Input dimensions (single sample)
    int input_height,
    int input_width, 
    int input_channels,
    // Convolution parameters
    int kernel_height,
    int kernel_width,
    int output_channels,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    // Create tiny-cnn layer
    padding pad_type = (pad_h > 0 || pad_w > 0) ? padding::same : padding::valid;

    convolutional_layer<identity> conv_layer(
        input_width, input_height, kernel_width, 
        input_channels, output_channels, pad_type,
        true, // has bias
        stride_w, stride_h
    );

    // Get parameter references
    auto& W = conv_layer.weight();
    auto& b = conv_layer.bias();
    size_t weight_size = W.size();
    size_t bias_size = b.size();

    // Set weights and bias
    for (size_t i = 0; i < weight_size; ++i) {
        W[i] = weights[i];
    }
    for (size_t i = 0; i < bias_size; ++i) {
        b[i] = bias[i];
    }

    // Convert single input to tiny-cnn format
    int input_size = input_channels * input_height * input_width;
    vec_t input_vec(input_size);

    for (int i = 0; i < input_size; i++) {
        input_vec[i] = static_cast<float>(input_data[i]);
    }

    // Forward propagation
    const vec_t& result = conv_layer.forward_propagation(input_vec, 0);

    // Copy result to output buffer
    for (size_t i = 0; i < result.size(); i++) {
        output[i] = static_cast<float>(result[i]);
    }

    return 0; // Success
}

// Single inference max pooling - NO BATCHING
int maxpool2d_single_inference(
    const float* input_data,    // Single input tensor
    float* output,              // Output buffer
    // Input dimensions (single sample)
    int input_height,
    int input_width,
    int channels,
    // Pooling parameters
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    // Create tiny-cnn max pooling layer
    max_pooling_layer<identity> pool_layer(
        input_width, input_height, channels,
        kernel_width, stride_w
    );

    // Convert single input to tiny-cnn format
    int input_size = channels * input_height * input_width;
    vec_t input_vec(input_size);

    for (int i = 0; i < input_size; i++) {
        input_vec[i] = static_cast<float>(input_data[i]);
    }

    // Forward propagation
    const vec_t& result = pool_layer.forward_propagation(input_vec, 0);

    // Copy result to output buffer
    for (size_t i = 0; i < result.size(); i++) {
        output[i] = static_cast<float>(result[i]);
    }

    return 0; // Success
}

} // extern "C"
