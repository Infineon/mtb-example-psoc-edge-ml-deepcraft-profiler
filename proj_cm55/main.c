/*****************************************************************************
* File Name        : main.c
*
* Description      : This source file contains the main routine for CM55 CPU
*
* Related Document : See README.md
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

/*******************************************************************************
* Header Files
*******************************************************************************/
#include "cybsp.h"

#ifdef ML_DEEPCRAFT_CM55
#include "ml_validation.h"
#include "app_common.h"
#endif

/*******************************************************************************
* Macros
*******************************************************************************/
/* Default UART baudrate */
#define UART_DEFAULT_STREAM_BAUD_RATE   (115200u)

/* Check if the UART transmission has completed */
#define WAIT_FOR_TX_COMPLETE()   while (!(Cy_SCB_UART_IsTxComplete \
                                                       (CYBSP_DEBUG_UART_HW)))

/*******************************************************************************
* Function Prototypes
*******************************************************************************/

#ifdef ML_DEEPCRAFT_CM55
/*****************************************************************************
* Function Name: cm55_ml_profiler_task
******************************************************************************
 * Summary:
 * This is the ML task for CM55. It does...
 *    1. Print welcome message
 *    2. Initialize the system for ML profiling (timer and model)
 *    3. Execute the inference engine
 *
 * Parameters:
 *  void
 *
 * Return:
 *  void

*
*****************************************************************************/
static void cm55_ml_profiler_task(void)
{
    cy_rslt_t result;
    
    app_retarget_io_init(UART_DEFAULT_STREAM_BAUD_RATE);
    
    /* \x1b[2J\x1b[;H - ANSI ESC sequence for clear screen */
    printf("\x1b[2J\x1b[;H");
    
    printf("****************** "
           "PSOC Edge MCU: Machine Learning DEEPCRAFT Profiler on CM55+U55 "
           "****************** \r\n\n");

    /* Initialize model and the timer used to determine inference cycles */
    result = ml_validation_init();
    if(CY_RSLT_SUCCESS != result)
    {
        printf("ERROR: initialization of the ML validation failed!\r\n");
        handle_error();
    }

    /* Start profiling the model */
    result = ml_validation_local_task();
    if (CY_RSLT_SUCCESS == result)
    {
        printf("\n\rProfiling completed!\n\r");
    }
    else
    {
        printf("\n\rProfiling task failed!\n\r");
    }
}
#endif
/*****************************************************************************
* Function Name: main
******************************************************************************
* Summary:
* This is the main function for CM55 application. 
* 
* It sets up a machine learning model to be profiled. It can use local 
* regression data.
* 
* Parameters:
*  void
*
* Return:
*  int
*
*****************************************************************************/
int main(void)
{
    cy_rslt_t result;

    /* Initialize the device and board peripherals */
    result = cybsp_init();

    /* Board init failed. Stop program execution */
    if (CY_RSLT_SUCCESS != result)
    {
        CY_ASSERT(0);
    }

    /* Enable global interrupts */
    __enable_irq();

#ifdef ML_DEEPCRAFT_CM55
    /* Start the profiling task if cm55 is set as the ML_DEEPCRAFT_CPU
    in the common.mk file */
    cm55_ml_profiler_task();
    
    WAIT_FOR_TX_COMPLETE();
#endif

    for (;;)
    {
        Cy_SysPm_CpuEnterDeepSleep(CY_SYSPM_WAIT_FOR_INTERRUPT);
    }
    
}

/* [] END OF FILE */
