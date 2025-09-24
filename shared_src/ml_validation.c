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
* Copyright 2025, Cypress Semiconductor Corporation (an Infineon company) or
* an affiliate of Cypress Semiconductor Corporation.  All rights reserved.
*
* This software, including source code, documentation and related
* materials ("Software") is owned by Cypress Semiconductor Corporation
* or one of its affiliates ("Cypress") and is protected by and subject to
* worldwide patent protection (United States and foreign),
* United States copyright laws and international treaty provisions.
* Therefore, you may use this Software only as provided in the license
* agreement accompanying the software package from which you
* obtained this Software ("EULA").
* If no EULA applies, Cypress hereby grants you a personal, non-exclusive,
* non-transferable license to copy, modify, and compile the Software
* source code solely for use in connection with Cypress's
* integrated circuit products.  Any reproduction, modification, translation,
* compilation, or representation of this Software except as specified
* above is prohibited without the express written permission of Cypress.
*
* Disclaimer: THIS SOFTWARE IS PROVIDED AS-IS, WITH NO WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, NONINFRINGEMENT, IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. Cypress
* reserves the right to make changes to the Software without notice. Cypress
* does not assume any liability arising out of the application or use of the
* Software or any product or circuit described in the Software. Cypress does
* not authorize its products for use in any products where a malfunction or
* failure of the Cypress product may reasonably be expected to result in
* significant property damage, injury or death ("High Risk Product"). By
* including Cypress's product in a High Risk Product, the manufacturer
* of such system or application assumes all risk of such use and in doing
* so agrees to indemnify Cypress against all liability.
*******************************************************************************/
#include "ml_validation.h"

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "elapsed_timer.h"

#ifdef ML_DEEPCRAFT_INT8
#include "TEST_MODEL_int8x8.h"

#if IMAI_DATA_IN_TYPE_ID == IMAGINET_TYPES_INT8
#include "TEST_MODEL_int8x8_data_input.h"
#include "TEST_MODEL_int8x8_data_output.h"
#endif

#if IMAI_DATA_IN_TYPE_ID == IMAGINET_TYPES_FLOAT32
#include "TEST_MODEL_float_data_input.h"
#include "TEST_MODEL_float_data_output.h"
#endif
#endif

#ifdef ML_DEEPCRAFT_FLOAT
#include "TEST_MODEL_float.h"
#include "TEST_MODEL_float_data_input.h"
#include "TEST_MODEL_float_data_output.h"
#endif

/*******************************************************************************
* Constants
*******************************************************************************/
#define SUCCESS_RATE       (98.0f)


/*******************************************************************************
* Global Variables
*******************************************************************************/
static uint64_t region_start[IMAI_REGIONS_COUNT] = {0};
static uint64_t region_profile[IMAI_REGIONS_COUNT] = {0};
static uint64_t region_profile_sum[IMAI_REGIONS_COUNT] = {0};

/*******************************************************************************
* Function Name: IMAI_hook_region
********************************************************************************
* Summary:
*   Create a hook that is used during profiling to enable the timing of
*    the model.
*
* Parameters:
*   entered: Determines if the region was entered or exited
*   region_id: What part of the pipeline is being timed
*
* Return:
*   void
*******************************************************************************/
void IMAI_hook_region(bool entered, int32_t region_id)
{
    uint64_t ticks = 0;
  
    /* Assign the current tick count*/
    elapsed_timer_get_tick(&ticks);
    
    /* Get the time entering and exiting a region */
    if (entered)
    {
        region_start[region_id] = ticks;
    }
    else 
    {
        region_profile[region_id] = ticks - region_start[region_id];
        region_profile_sum[region_id] += region_profile[region_id];
    }
}

/*******************************************************************************
* Function Name: ml_validation_init
********************************************************************************
* Summary:
*   Initialize the Neural Network based on the given model and setup to start
*   regression of the model and profiling configuration.
*
* Parameters:
*   void
*
* Return:
*   cy_rslt_t: the status of the initialization.
*******************************************************************************/
cy_rslt_t ml_validation_init(void)
{
    cy_rslt_t result;
    
    result = IMAI_init();
    
    if (IMAI_RET_SUCCESS != result)
    {
        printf("ERROR: initialization of the model failed!\r\n");
        return result;
    }
    
    IMAI_mtb_models_print_info();
    
    result = elapsed_timer_init();
    
    if(CY_RSLT_SUCCESS != result)
    {
        printf("ERROR: initialization of elapsed timer failed!\r\n");
        return result;
    }
    
    return result;
}


/*******************************************************************************
* Function Name: find_max_output
********************************************************************************
* Summary:
*   This function finds the maximum value in an array and return its index.
*
* Parameters:
*   in: Pointer to the model output data
*   size: Number of elements to compare
*
* Return:
*   int: The index of maximum value, -1 if input parameter is invalid
*******************************************************************************/
int find_max_output(const IMAI_DATA_OUT_TYPE * in, int size)
{
    int loop_count, max_idx = -1;
    IMAI_DATA_OUT_TYPE val, max_val;
    
    
    /* Check for valid inputs */
    if (in != NULL && size > 0)
    {
        loop_count = size - 1;
        max_idx = 0;
        max_val = *in++;

        /* Compare each value in the array */
        while(loop_count > 0)
        {
            val = *in++;
            if (val > max_val)
            {
                max_idx = size - loop_count;
                max_val = val;
            }
            loop_count--;
        }
    }
    return max_idx;
}

/*******************************************************************************
* Function Name: ml_validation_local_task
********************************************************************************
* Summary:
*   Run the Neural Network Inference Engine based on the local data.
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
    IMAI_DATA_IN_TYPE   *input_reference;
    IMAI_DATA_OUT_TYPE  *output_reference;
    
    /* Stores model output */
    IMAI_DATA_IN_TYPE   result_buffer[IMAI_DATA_OUT_COUNT];
        
    uint32_t correct_result = 0;
    float    success_rate;
    bool     test_result;
    
    if (IMAI_DATA_IN_COUNT != IMAI_DATA_INPUT_SAMPLE_SIZE)
    {
        printf("Model input size does not match regression sample size\r\n");
        return ML_INPUT_SIZE_MISMATCH;
    }
    
    if (IMAI_DATA_OUT_COUNT != IMAI_DATA_OUTPUT_SAMPLE_SIZE)
    {
        printf("Model output size does not match regression output size\r\n");
        return ML_OUTPUT_SIZE_MISMATCH;
    }

    /* Point to regression data */
    input_reference = (IMAI_DATA_IN_TYPE *) IMAI_data_input;
    output_reference = (IMAI_DATA_OUT_TYPE *) IMAI_data_output;
    
    /* The following loop runs for number of examples used in regression */
    for (int j = 0; j < IMAI_DATA_INPUT_SAMPLES; j++)
    {
        /* Feed the regression data to the model */
        IMAI_compute((const IMAI_DATA_IN_TYPE*)input_reference, result_buffer);
        
         /* Check if the results are accurate enough */
        if (find_max_output(result_buffer, IMAI_DATA_OUTPUT_SAMPLE_SIZE) ==
            find_max_output(output_reference, IMAI_DATA_OUTPUT_SAMPLE_SIZE))
        {
            correct_result++;
        }
        
        /* Increment buffers */
        input_reference  += IMAI_DATA_INPUT_SAMPLE_SIZE;
        output_reference += IMAI_DATA_OUTPUT_SAMPLE_SIZE;
    }

    success_rate = ((float) correct_result) * 100.0f / ((float) IMAI_DATA_INPUT_SAMPLES);
    test_result = (success_rate >= SUCCESS_RATE);
    
    /* Print model profiling results */
    IMAI_mtb_models_profile_log();
    
    printf("***************************************************\r\n");
    if (test_result == true)
    {
        printf("PASS with accuracy percentage =%3.2f, total_cnt=%d", success_rate, (int) IMAI_DATA_INPUT_SAMPLES);
    }
    else
    {
        printf("FAIL with accuracy percentage =%3.2f, total_cnt=%d", success_rate, (int) IMAI_DATA_INPUT_SAMPLES);
    }
    printf("\r\n***************************************************\r\n");

    return CY_RSLT_SUCCESS;
}

/* [] END OF FILE */