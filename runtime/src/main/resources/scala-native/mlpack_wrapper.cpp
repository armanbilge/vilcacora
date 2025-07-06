/*
 * Copyright 2023 Arman Bilge
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <mlpack/prereqs.hpp> // Includes Armadillo
#include <iostream>



// --- ONNX Add (Element-wise addition of two matrices/tensors) ---
void onnx_add_matrix_double(
    double* A_data, size_t rowsA, size_t colsA,
    double* B_data, size_t rowsB, size_t colsB,
    double* C_data) {

    arma::mat A(A_data, rowsA, colsA, false, true);
    arma::mat B(B_data, rowsB, colsB, false, true);

    if (rowsA != rowsB || colsA != colsB) {
        std::cerr << "Error: Matrices must have same dimensions for ONNX Add." << std::endl;
        return;
    }
    arma::mat C = A + B;
    for (size_t i = 0; i < C.n_elem; ++i) {
        C_data[i] = C(i);
    }
}

// --- ONNX Mul (Element-wise multiplication of two matrices/tensors) ---
void onnx_mul_matrix_double(
    double* A_data, size_t rowsA, size_t colsA,
    double* B_data, size_t rowsB, size_t colsB,
    double* D_data) {

    arma::mat A(A_data, rowsA, colsA, false, true);
    arma::mat B(B_data, rowsB, colsB, false, true);

    if (rowsA != rowsB || colsA != colsB) {
        std::cerr << "Error: Matrices must have same dimensions for ONNX Mul." << std::endl;
        return;
    }
    arma::mat D = A % B; // Element-wise multiplication
    for (size_t i = 0; i < D.n_elem; ++i) {
        D_data[i] = D(i);
    }
}

// --- ONNX Cast (from double to float) ---
void onnx_cast_double_to_float(
    double* input_data, size_t rows, size_t cols,
    float* output_data) {

    arma::mat input_mat(input_data, rows, cols, false, true);
    arma::fmat output_fmat = arma::conv_to<arma::fmat>::from(input_mat);
    for (size_t i = 0; i < output_fmat.n_elem; ++i) {
        output_data[i] = output_fmat(i);
    }
}

// --- ONNX Cast (from float to double) ---
void onnx_cast_float_to_double(
    float* input_data, size_t rows, size_t cols,
    double* output_data) {

    arma::fmat input_fmat(input_data, rows, cols, false, true);
    arma::mat output_mat = arma::conv_to<arma::mat>::from(input_fmat);
    for (size_t i = 0; i < output_mat.n_elem; ++i) {
        output_data[i] = output_mat(i);
    }
}

