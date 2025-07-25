#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "svm.h"

// Wrapper functions for LibSVM struct creation and management
extern "C" {

// Create and initialize svm_parameter
struct svm_parameter* create_svm_param(
    int svm_type, 
    int kernel_type, 
    int degree, 
    double gamma, 
    double coef0
) {
    struct svm_parameter *param = (struct svm_parameter *)malloc(sizeof(struct svm_parameter));
    if (!param) return NULL;
    
    // Initialize all fields to safe defaults
    memset(param, 0, sizeof(struct svm_parameter));
    
    // Set provided values
    param->svm_type = svm_type;
    param->kernel_type = kernel_type;
    param->degree = degree;
    param->gamma = gamma;
    param->coef0 = coef0;
    

    
    return param;
}

// Create svm_model with proper initialization
struct svm_model* create_svm_model(
    struct svm_parameter *param,
    int nr_class,
    int l,
    double *support_vectors,  // flattened array [l * num_features]
    int num_features,
    double *coefficients,     // flattened array [(nr_class-1) * l]  
    double *rho,             // array [nr_class*(nr_class-1)/2]
    int *class_labels,       // array [nr_class]
    int *n_sv_per_class      // array [nr_class]
) {
    struct svm_model *model = (struct svm_model *)malloc(sizeof(struct svm_model));
    if (!model) return NULL;
    
    // Initialize all fields
    memset(model, 0, sizeof(struct svm_model));
    
    // Basic model properties
    model->param = *param;  // Copy parameter struct
    model->nr_class = nr_class;
    model->l = l;
    
    // Allocate and populate support vectors
    model->SV = (struct svm_node **)malloc(sizeof(struct svm_node*) * l);
    for (int i = 0; i < l; i++) {
        model->SV[i] = (struct svm_node *)malloc(sizeof(struct svm_node) * (num_features + 1));
        
        // Copy feature values
        for (int j = 0; j < num_features; j++) {
            model->SV[i][j].index = j + 1;  // 1-indexed
            model->SV[i][j].value = support_vectors[i * num_features + j];
        }
        // Terminator node
        model->SV[i][num_features].index = -1;
        model->SV[i][num_features].value = 0.0;
    }
    
    // Allocate and populate coefficients
    model->sv_coef = (double **)malloc(sizeof(double*) * (nr_class - 1));
    for (int i = 0; i < nr_class - 1; i++) {
        model->sv_coef[i] = (double *)malloc(sizeof(double) * l);
        for (int j = 0; j < l; j++) {
            model->sv_coef[i][j] = coefficients[i * l + j];
        }
    }
    
    // Copy rho (bias terms)
    int rho_size = nr_class * (nr_class - 1) / 2;
    model->rho = (double *)malloc(sizeof(double) * rho_size);
    memcpy(model->rho, rho, sizeof(double) * rho_size);
    
    // Copy class labels
    model->label = (int *)malloc(sizeof(int) * nr_class);
    memcpy(model->label, class_labels, sizeof(int) * nr_class);
    
    // Copy number of SVs per class
    model->nSV = (int *)malloc(sizeof(int) * nr_class);
    memcpy(model->nSV, n_sv_per_class, sizeof(int) * nr_class);
    
    // Initialize other fields
    model->probA = NULL;
    model->probB = NULL; 
    model->sv_indices = NULL;
    model->free_sv = 1;
    
    return model;
}

// Prediction with per-class scores
int svm_predict_with_scores(
    struct svm_model *model,
    double *features,        // input features [num_features]
    int num_features,
    double *class_scores     // output scores [nr_class] 
) {
    // Create input svm_node array
    struct svm_node *x = (struct svm_node *)malloc(sizeof(struct svm_node) * (num_features + 1));
    
    for (int i = 0; i < num_features; i++) {
        x[i].index = i + 1;
        x[i].value = features[i];
    }
    x[num_features].index = -1;  // terminator
    
    // Get decision values from LibSVM
    int nr_class = model->nr_class;
    int dec_values_count = (nr_class * (nr_class - 1)) / 2;
    double *dec_values = (double *)malloc(sizeof(double) * dec_values_count);
    
    double predicted_label = svm_predict_values(model, x, dec_values);
    
    // Convert to per-class scores
    if (nr_class == 2) {
        // Binary case
        class_scores[0] = -dec_values[0];
        class_scores[1] = dec_values[0];
    } else {
        // Multiclass: OvO to OvR conversion
        int *votes = (int *)calloc(nr_class, sizeof(int));
        double *conf = (double *)calloc(nr_class, sizeof(double));
        
        int k = 0;
        for (int i = 0; i < nr_class; i++) {
            for (int j = i + 1; j < nr_class; j++) {
                double margin = dec_values[k];
                
                if (margin > 0) {
                    votes[i] += 1;
                } else {
                    votes[j] += 1;
                }
                
                conf[i] -= margin;
                conf[j] += margin;
                k++;
            }
        }
        
        // Apply tie-breaking and final scores
        for (int c = 0; c < nr_class; c++) {
            double tconf = conf[c] / (3.0 * (fabs(conf[c]) + 1.0));
            class_scores[c] = (double)votes[c] + tconf;
        }
        
        free(votes);
        free(conf);
    }
    
    free(x);
    free(dec_values);
    
    return (int)predicted_label;
}

// Debug function to print model details
void debug_model_info(struct svm_model *model) {
    printf("=== SVM Model Debug Info ===\n");
    printf("nr_class: %d\n", model->nr_class);
    printf("l (num support vectors): %d\n", model->l);
    printf("kernel_type: %d\n", model->param.kernel_type);
    printf("gamma: %f\n", model->param.gamma);
    printf("coef0: %f\n", model->param.coef0);
    printf("degree: %d\n", model->param.degree);
    
    printf("Class labels: ");
    for (int i = 0; i < model->nr_class; i++) {
        printf("%d ", model->label[i]);
    }
    printf("\n");
    
    printf("Number of SVs per class: ");
    for (int i = 0; i < model->nr_class; i++) {
        printf("%d ", model->nSV[i]);
    }
    printf("\n");
    
    printf("Rho values: ");
    int rho_size = model->nr_class * (model->nr_class - 1) / 2;
    for (int i = 0; i < rho_size; i++) {
        printf("%f ", model->rho[i]);
    }
    printf("\n");
    printf("===========================\n");
}

} // extern "C"
