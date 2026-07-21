/******************************************************************************
* File Name:   ml_validation.c
*
* Description: This file contains the implementation of the validation of the
*              machine learning model.
*
* Related Document: See README.md
*
*
*******************************************************************************
* (c) 2025-2026, Infineon Technologies AG, or an affiliate of Infineon
* Technologies AG. All rights reserved.
* This software, associated documentation and materials ("Software") is
* owned by Infineon Technologies AG or one of its affiliates ("Infineon")
* and is protected by and subject to worldwide patent protection, worldwide
* copyright laws, and international treaty provisions. Therefore, you may use
* this Software only as provided in the license agreement accompanying the
* software package from which you obtained this Software. If no license
* agreement applies, then any use, reproduction, modification, translation, or
* compilation of this Software is prohibited without the express written
* permission of Infineon.
*
* Disclaimer: UNLESS OTHERWISE EXPRESSLY AGREED WITH INFINEON, THIS SOFTWARE
* IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, ALL WARRANTIES OF NON-INFRINGEMENT OF
* THIRD-PARTY RIGHTS AND IMPLIED WARRANTIES SUCH AS WARRANTIES OF FITNESS FOR A
* SPECIFIC USE/PURPOSE OR MERCHANTABILITY.
* Infineon reserves the right to make changes to the Software without notice.
* You are responsible for properly designing, programming, and testing the
* functionality and safety of your intended application of the Software, as
* well as complying with any legal requirements related to its use. Infineon
* does not guarantee that the Software will be free from intrusion, data theft
* or loss, or other breaches ("Security Breaches"), and Infineon shall have
* no liability arising out of any Security Breaches. Unless otherwise
* explicitly approved by Infineon, the Software may not be used in any
* application where a failure of the Product or any consequences of the use
* thereof can reasonably be expected to result in personal injury.
*******************************************************************************/
#include "ml_validation.h"

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#ifndef USE_STREAM_DATA
/* Include regression files */
#include MTB_ML_INCLUDE_MODEL_X_DATA_FILE(MODEL_NAME)
#include MTB_ML_INCLUDE_MODEL_Y_DATA_FILE(MODEL_NAME)
#endif

/*******************************************************************************
* Constants
*******************************************************************************/
#define SUCCESS_RATE       (98.0f)

/* Timeout value for streaming */
#define DEFAULT_TIMEOUT_MS (5000u)

/*******************************************************************************
* Global Variables
*******************************************************************************/
/* NN Model Object */
static mtb_ml_model_t *model_obj;

/* Output/result buffer for the inference engine */
static MTB_ML_DATA_T *result_buffer;

#ifndef USE_STREAM_DATA
/* Model Output Size */
static int model_output_size;

/*******************************************************************************
* Function Name: calculate_mae
********************************************************************************
* Summary:
*   This function finds the MAE (Mean Absolute Error) between the predicted
*   values and the reference values.
*
* Parameters:
*   pred:   Pointer to the first element of the predicted values
*   ref:    Pointer to the first element of the reference values
*   size:   Total number of elements in the MAE calculation
*
* Return:
*   float: MAE calculation, or CALCULATE_MAE_ERROR if input parameters are
*          invalid
*******************************************************************************/
static float calculate_mae(const float *pred, const float *ref, int size)
{
    if (pred == NULL || ref == NULL || size <= 0)
    {
        return CALCULATE_MAE_ERROR;
    }

    float mae_current = 0.0f;
    for (int k = 0; k < size; k++)
    {
        float diff = pred[k] - ref[k];
        mae_current += (diff < 0.0f) ? -diff : diff;
    }
    return mae_current / (float)size;
}

/*******************************************************************************
* Function Name: dequantize_reference
********************************************************************************
* Summary:
*   Dequantize a slice of the reference (Y) regression data into floating point
*   using the model output tensor's quantization parameters. If the model output
*   is already floating point, the values are copied as-is.
*
* Parameters:
*   ref:        Pointer to the quantized reference values
*   dst:        Destination floating point buffer
*   size:       Number of elements to dequantize
*   scale:      Quantization scale factor of the output tensor
*   zero_point: Quantization zero point of the output tensor
*   type_size:  Size in bytes of a single output element
*
* Return:
*   void
*******************************************************************************/
static void dequantize_reference(const MTB_ML_DATA_T *ref, float *dst, int size,
                                 float scale, int zero_point, int type_size)
{
    for (int k = 0; k < size; k++)
    {
        if (type_size == sizeof(float))
        {
            dst[k] = ((const float *)ref)[k];
        }
        else
        {
            dst[k] = ((float)ref[k] - (float)zero_point) * scale;
        }
    }
}
#endif /* USE_STREAM_DATA */

/*******************************************************************************
* Function Name: ml_validation_init
********************************************************************************
* Summary:
*   Initialize the Neural Network based on the given model and setup to start
*   regression of the model and profiling configuration.
*
* Parameters:
*   profile_cfg: profiling configuration
*   model_bin: pointer to the model data
*
* Return:
*   cy_rslt_t: the status of the initialization.
*******************************************************************************/
cy_rslt_t ml_validation_init(mtb_ml_profile_config_t profile_cfg,
                             mtb_ml_model_bin_t *model_bin)
{
    cy_rslt_t result;

    /* Initialize the neural network */
    result = mtb_ml_model_init(model_bin, NULL, &model_obj);
    if (CY_RSLT_SUCCESS != result)
    {
        printf("MTB ML initialization failure: %lu\r\n", (unsigned long) result);
        return result;
    }

    mtb_ml_model_profile_config(model_obj, profile_cfg);

    /* Allocate output buffer for inference results */
    result_buffer = (MTB_ML_DATA_T *) malloc(model_obj->output_concat_bytes);
    if (result_buffer == NULL)
    {
        printf("ERROR: Allocating memory for result_buffer\r\n");
        mtb_ml_model_deinit(model_obj);
        return MTB_ML_RESULT_ALLOC_ERR;
    }

    /* Print information about the model */
    mtb_ml_utils_print_model_info(model_obj);

    return CY_RSLT_SUCCESS;
}

#ifndef USE_STREAM_DATA
/*******************************************************************************
* Function Name: ml_validation_local_task
********************************************************************************
* Summary:
*   Run the Neural Network Inference Engine based on the local data. For each
*   output tensor the classification accuracy (argmax match) and the Mean
*   Absolute Error (MAE) against the reference data are computed and printed.
*
* Parameters:
*   void
*
* Return:
*   cy_rslt_t: the status of the task execution.
*******************************************************************************/
cy_rslt_t ml_validation_local_task(void)
{
    /* Regression pointers */
    MTB_ML_DATA_T  *input_reference;
    MTB_ML_DATA_T  *output_reference;

    uint32_t     num_loop;
    cy_rslt_t    result;
    int          file_input_size;
    int          model_input_size = (int)(model_obj->input_concat_bytes / model_obj->input_type_size);
    int          output_count = model_obj->output_count;
    model_output_size = (int)(model_obj->output_concat_bytes / model_obj->output_type_size);

    /* Float conversion buffers (sized to the largest output tensor) */
    float *output_float_buffer = NULL;
    float *ref_float_buffer = NULL;
    int    max_output_elements = 0;

    /* Per-output metrics */
    uint32_t *correct_result = NULL;
    float    *mae_sum = NULL;
    float    *mae_max = NULL;

    /* Parse input data information */
    mtb_ml_x_file_header_t *x_file_header = (mtb_ml_x_file_header_t *) MTB_ML_MODEL_X_DATA_BIN(MODEL_NAME);

    /* Point to regression data */
    input_reference  = (MTB_ML_DATA_T *) (((uint32_t) x_file_header) + sizeof(*x_file_header));
    output_reference = (MTB_ML_DATA_T *) MTB_ML_MODEL_Y_DATA_BIN(MODEL_NAME);

    /* Get the number of loops for this regression */
    num_loop = x_file_header->num_of_samples;

    /* Get the number of inputs of the NN */
    file_input_size = x_file_header->input_size;

#if defined(RNN_STREAMING)
    model_obj->recurrent_ts_size = x_file_header->recurrent_ts_size;

    /* Check if it is a non-RNN model */
    if (model_obj->recurrent_ts_size <= 0)
    {
        printf("This is not a RNN model (%d). Set the NN_RNN_MODEL variable to NO in the Makefile, aborting...\r\n",
            model_obj->recurrent_ts_size);
        return MTB_ML_RESULT_MISMATCH_DATA_TYPE;
    }

    /* If using RNN Model, check if the data time steps matches */
    if ((file_input_size/model_obj->recurrent_ts_size) != model_input_size)
    {
        printf("Data size error, file input size=%d, model input size=%d recurrent time series size=%d, aborting...\r\n",
            file_input_size, model_input_size, model_obj->recurrent_ts_size);
        return MTB_ML_RESULT_MISMATCH_DATA_TYPE;
    }

    /* Scratch buffer for one RNN frame; */
    MTB_ML_DATA_T *input_slice = (MTB_ML_DATA_T *) malloc(model_input_size * model_obj->input_type_size);
    if (input_slice == NULL)
    {
        printf("ERROR: Allocating memory for input slice\r\n");
        return MTB_ML_RESULT_ALLOC_ERR;
    }
#else
    /* Check if the file input size matches the model input size */
    if (file_input_size != model_input_size)
    {
        printf("Input buffer size error, file input size=%d, model input size=%d, aborting...\r\n",
                file_input_size, model_input_size);
        return MTB_ML_RESULT_MISMATCH_DATA_TYPE;
    }
#endif /* RNN_STREAMING */

    /* Determine the largest output tensor to size the float buffers */
    for (int i = 0; i < output_count; i++)
    {
        if ((int)model_obj->outputs[i].elements > max_output_elements)
        {
            max_output_elements = (int)model_obj->outputs[i].elements;
        }
    }

    output_float_buffer = (float *) malloc(max_output_elements * sizeof(float));
    ref_float_buffer    = (float *) malloc(max_output_elements * sizeof(float));
    correct_result      = (uint32_t *) calloc(output_count, sizeof(uint32_t));
    mae_sum             = (float *) calloc(output_count, sizeof(float));
    mae_max             = (float *) calloc(output_count, sizeof(float));

    if ((output_float_buffer == NULL) || (ref_float_buffer == NULL) ||
        (correct_result == NULL) || (mae_sum == NULL) || (mae_max == NULL))
    {
        printf("Failed to allocate memory for validation buffers\r\n");
        free(output_float_buffer);
        free(ref_float_buffer);
        free(correct_result);
        free(mae_sum);
        free(mae_max);
#if defined(RNN_STREAMING)
        free(input_slice);
#endif
        return MTB_ML_RESULT_ALLOC_ERR;
    }

    /* The following loop runs for number of examples used in regression */
    for (uint32_t j = 0; j < num_loop; j++)
    {
#if defined(RNN_STREAMING)
        result = mtb_ml_model_rnn_reset_all_parameters(model_obj);
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            printf("ERROR: failed to reset model parameters\r\n");
            goto cleanup;
        }

        for (int i = 0; i < model_obj->recurrent_ts_size; i++)
        {
            /* Input data is 2D array squashed to 1D array by Coretools */
            for (int z = 0; z < model_input_size; z++)
            {
                input_slice[z] = input_reference[i*model_input_size+z];
            }

            /* Load input data into each input tensor for multi-input models */
            for (int input_idx = 0; input_idx < model_obj->input_count; input_idx++)
            {
                MTB_ML_DATA_T *input_tensor_data = (MTB_ML_DATA_T *)((uint8_t *)input_slice + model_obj->inputs[input_idx].concat_offset_bytes);
                result = mtb_ml_model_inputs(model_obj, input_tensor_data, input_idx);
                if (MTB_ML_RESULT_SUCCESS != result)
                {
                    goto cleanup;
                }
            }

            result = mtb_ml_model_invoke(model_obj);
            if (MTB_ML_RESULT_SUCCESS != result)
            {
                goto cleanup;
            }
        }
#else
        /* Load input data into each input tensor for multi-input models */
        for (int input_idx = 0; input_idx < model_obj->input_count; input_idx++)
        {
            MTB_ML_DATA_T *input_tensor_data = (MTB_ML_DATA_T *)((uint8_t *)input_reference + model_obj->inputs[input_idx].concat_offset_bytes);
            result = mtb_ml_model_inputs(model_obj, input_tensor_data, input_idx);
            if (MTB_ML_RESULT_SUCCESS != result)
            {
                goto cleanup;
            }
        }

        /* Invoke the model */
        result = mtb_ml_model_invoke(model_obj);
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            goto cleanup;
        }
#endif /* RNN_STREAMING */

        /* Concatenate outputs into result_buffer allocated in ml_validation_init() */
        result = mtb_ml_model_load_output(model_obj, &result_buffer);
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            goto cleanup;
        }

        /* Calculate accuracy and MAE evaluation per-output */
        for (int i = 0; i < output_count; i++)
        {
            int elements = (int)model_obj->outputs[i].elements;

            /* Dequantize the model output tensor to floating point */
            result = mtb_ml_utils_model_dequantize_tensor(model_obj, i,
                                                          output_float_buffer,
                                                          (size_t)elements);
            if (MTB_ML_RESULT_SUCCESS != result)
            {
                goto cleanup;
            }

            /* Dequantize the matching reference slice to floating point */
            MTB_ML_DATA_T *ref_ptr = output_reference +
                (model_obj->outputs[i].concat_offset_bytes / model_obj->output_type_size);
            dequantize_reference(ref_ptr, ref_float_buffer, elements,
                                 model_obj->outputs[i].scale,
                                 model_obj->outputs[i].zero_point,
                                 model_obj->output_type_size);

            /* Check if the classification with the highest confidence matches */
            int res_max_idx = mtb_ml_utils_find_max_flt(output_float_buffer, elements);
            int ref_max_idx = mtb_ml_utils_find_max_flt(ref_float_buffer, elements);
            if (res_max_idx == ref_max_idx)
            {
                correct_result[i]++;
            }

            /* Calculate the MAE between model output and reference data */
            float mae_current = calculate_mae(output_float_buffer, ref_float_buffer, elements);
            if (CALCULATE_MAE_ERROR == mae_current)
            {
                printf("Invalid arguments passed to calculate_mae\r\n");
                result = MTB_ML_RESULT_BAD_ARG;
                goto cleanup;
            }

            mae_sum[i] += mae_current;
            if (mae_current > mae_max[i])
            {
                mae_max[i] = mae_current;
            }
        }

        /* Advance to next regression sample */
        input_reference  += file_input_size;
        output_reference += model_output_size;
    }

    /* Print model profiling results */
    mtb_ml_model_profile_log(model_obj);

    /* Print MAE and accuracy results */
    printf("***************************************************\r\n");
    for (int i = 0; i < output_count; i++)
    {
        float success_rate = (num_loop != 0) ?
            ((float)correct_result[i] * 100.0f / (float)num_loop) : 0.0f;
        float mae_avg = (num_loop != 0) ?
            (mae_sum[i] / (float)num_loop) : 0.0f;
        printf("data_out_%d: %s accuracy=%3.2f%%, total_cnt=%d\r\n",
               i,
               (success_rate >= SUCCESS_RATE) ? "PASS" : "FAIL",
               success_rate, (int)num_loop);
        printf("  Average Mean Absolute Error: %.8f\r\n", mae_avg);
        printf("  Maximum Mean Absolute Error: %.8f\r\n", mae_max[i]);
        if (i < output_count - 1)
        {
            printf("\r\n");
        }
    }
    printf("***************************************************\r\n");

    result = CY_RSLT_SUCCESS;

cleanup:
    free(output_float_buffer);
    free(ref_float_buffer);
    free(correct_result);
    free(mae_sum);
    free(mae_max);
#if defined(RNN_STREAMING)
    free(input_slice);
#endif /* RNN_STREAMING */

    return result;
}
#endif /* USE_STREAM_DATA */

#ifdef USE_STREAM_DATA
/*******************************************************************************
* Function Name: ml_validation_stream_task
********************************************************************************
* Summary:
*   Run the Neural Network Inference Engine based on the stream data. The
*   regression data is streamed from the DEEPCRAFT(TM) Model Converter over
*   UART and the inference results are sent back to the host, which computes
*   the accuracy metrics.
*
* Parameters:
*   iface: pointer to the streaming interface
*
* Return:
*   cy_rslt_t: the status of the task execution.
*******************************************************************************/
cy_rslt_t ml_validation_stream_task(mtb_ml_stream_interface_t *iface)
{
    cy_rslt_t result = MTB_ML_RESULT_SUCCESS;

    /* Initialize the streaming interface */
    result = mtb_ml_stream_init(iface, model_obj);
    if (CY_RSLT_SUCCESS != result)
    {
        printf("MTB ML streaming init failure: %lu\r\n", (unsigned long) result);
        return result;
    }

    /* Alloc RX buf */
    MTB_ML_DATA_T *rx_buf = (MTB_ML_DATA_T *) malloc(iface->input_size * model_obj->input_type_size);
    if (!rx_buf)
    {
        printf("ERROR: Allocating memory for rx_buf\r\n");
        return MTB_ML_RESULT_ALLOC_ERR;
    }

    /* Set timeout values */
    /* Calculate timeout as ten times of 1 second per 10KB */
    uint32_t rx_timeout_ms = (1000 * iface->input_size * model_obj->input_type_size)/(1024);
    if (rx_timeout_ms < DEFAULT_TIMEOUT_MS)
    {
        rx_timeout_ms = DEFAULT_TIMEOUT_MS;
    }
    uint32_t tx_timeout_ms = (1000 * model_obj->output_size * model_obj->output_type_size)/(1024);
    if (tx_timeout_ms < DEFAULT_TIMEOUT_MS)
    {
        tx_timeout_ms = DEFAULT_TIMEOUT_MS;
    }

    /* For non-streaming models data is sent by frames/samples, but for streaming frame slicing is applied.
    Number of slices can be calculated as: num_of_samples * recurrent_ts_size */
    int num_of_inferences = iface->x_data_info.num_of_samples;
#if defined(RNN_STREAMING)
    model_obj->recurrent_ts_size = iface->x_data_info.recurrent_ts_size;
    num_of_inferences *= model_obj->recurrent_ts_size;
#endif /* RNN_STREAMING */

    /* Do frame-by-frame (sample == frame) inference */
    for (int i = 0; i < num_of_inferences; i++)
    {
        /* Reset RNN model parameters per frame, does nothing if not RNN model.
        For RNN streaming models reset needs to be applied only when whole frame was streamed */
#if defined(RNN_STREAMING)
        if ((i % model_obj->recurrent_ts_size) == 0)
#endif /* RNN_STREAMING */
        {
            result = mtb_ml_model_rnn_reset_all_parameters(model_obj);
        }
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            printf("ERROR: failed to reset model parameters\r\n");
            free(rx_buf);
            return MTB_ML_RESULT_INFERENCE_ERROR;
        }

        /* Get input data */
        result = mtb_ml_stream_input_data(iface, rx_buf, rx_timeout_ms);
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            printf("ERROR: Failed to receive input data from host.\r\n");
            break;
        }

        /* Load input data into each input tensor for multi-input models */
        for (int input_idx = 0; input_idx < model_obj->input_count; input_idx++)
        {
            MTB_ML_DATA_T *input_tensor_data = (MTB_ML_DATA_T *)((uint8_t *)rx_buf + model_obj->inputs[input_idx].concat_offset_bytes);
            result = mtb_ml_model_inputs(model_obj, input_tensor_data, input_idx);
            if (MTB_ML_RESULT_SUCCESS != result)
            {
                free(rx_buf);
                return result;
            }
        }

        /* Invoke the model */
        result = mtb_ml_model_invoke(model_obj);
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            free(rx_buf);
            return result;
        }

        /* Concatenate outputs into result_buffer allocated in ml_validation_init() */
        result = mtb_ml_model_load_output(model_obj, &result_buffer);
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            free(rx_buf);
            return result;
        }

        /* Send output data.
        For RNN streaming models data needs to be sent only when whole frame was streamed */
#if defined(RNN_STREAMING)
        if (((i+1) % model_obj->recurrent_ts_size) == 0)
#endif /* RNN_STREAMING */
        {
            result = mtb_ml_stream_output_data(iface, result_buffer, tx_timeout_ms);
        }
        if (MTB_ML_RESULT_SUCCESS != result)
        {
            printf("ERROR: Failed to send output data to host\r\n");
            free(rx_buf);
            return MTB_ML_RESULT_COMM_ERROR;
        }
    }

    /* Free allocated memory */
    free(rx_buf);

    /* Generate profiling log if it is enabled */
    result = mtb_ml_model_profile_log(model_obj);
    if (MTB_ML_RESULT_SUCCESS != result)
    {
        printf("ERROR: Failed to generate profile log.\r\n");
        return MTB_ML_RESULT_BAD_MODEL;
    }

    return mtb_ml_inform_host_done(iface, DEFAULT_TIMEOUT_MS);
}
#endif /* USE_STREAM_DATA */

/* [] END OF FILE */
