/******************************************************************************
* File Name:   ml_profiler.c
*
* Description: This file contains the shared DEEPCRAFT(TM) ML profiler task
*              that initializes the ML middleware, model, and timer and runs
*              the local or streaming validation. It is shared by the core
*              projects so that each main.c only needs to handle the core
*              specific boot and then call ml_profiler_run().
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
#include "ml_profiler.h"
#include "ml_validation.h"
#include "elapsed_timer.h"
#include "app_common.h"

#include "cybsp.h"
#include "cy_pdl.h"

#include <stdio.h>

#include MTB_ML_INCLUDE_MODEL_FILE(MODEL_NAME)

/*******************************************************************************
* Macros
*******************************************************************************/
/* Default UART baudrate */
#ifdef USE_STREAM_DATA
    #define UART_DEFAULT_STREAM_BAUD_RATE   (1000000u)
#else
    #define UART_DEFAULT_STREAM_BAUD_RATE   (115200u)
#endif /* USE_STREAM_DATA */

/* Choose which profiling to enable. Options:
 *  MTB_ML_PROFILE_DISABLE
 *  MTB_ML_PROFILE_ENABLE_MODEL
 *  MTB_ML_LOG_ENABLE_MODEL_LOG
 */
#define PROFILE_CONFIGURATION       MTB_ML_PROFILE_ENABLE_MODEL

/* Check if the UART transmission has completed */
#define WAIT_FOR_TX_COMPLETE()   while (!(Cy_SCB_UART_IsTxComplete \
                                                       (CYBSP_DEBUG_UART_HW)))

/* The delay time in milliseconds used to wait for device to print UART messages */
#define DELAY_TIME_MS                 (50u)

/* MTB ML Block priority if using NPU */
#define MTB_ML_PRIORITY               (3)

/*******************************************************************************
* Function Name: ml_profiler_run
********************************************************************************
* Summary:
*   See ml_profiler.h for a full description.
*
* Parameters:
*   core_name: label shown in the banner (e.g. "CM33+NNLite" or "CM55+U55")
*
* Return:
*   void
*******************************************************************************/
void ml_profiler_run(const char *core_name)
{
    cy_rslt_t result;

    mtb_ml_model_bin_t model_bin = {MTB_ML_MODEL_BIN_DATA(MODEL_NAME)};

    app_retarget_io_init(UART_DEFAULT_STREAM_BAUD_RATE);

#ifdef USE_STREAM_DATA
    /* Data streaming object */
    mtb_data_streaming_interface_t data_stream_obj;
    mtb_data_streaming_context_t *context = (mtb_data_streaming_context_t *) &(data_stream_obj.context);
    context->obj_inst.uart = &mtb_ml_retarget_io_uart_obj;

    /* ML stream objects */
    mtb_ml_stream_tag_t stream_tag;
    mtb_ml_stream_interface_t stream_interface = {
        .interface_obj = &data_stream_obj,
        .stream_tag = &stream_tag,
    };
#endif /* USE_STREAM_DATA */

    /* \x1b[2J\x1b[;H - ANSI ESC sequence for clear screen */
    printf("\x1b[2J\x1b[;H");

    printf("****************** "
           "PSOC Edge MCU: Machine Learning DEEPCRAFT Profiler on %s "
           "****************** \r\n\n", core_name);

    /* Initialize the ModusToolbox ML middleware */
    mtb_ml_init(MTB_ML_PRIORITY);

    /* Initialize the model and profiling configuration */
    result = ml_validation_init(PROFILE_CONFIGURATION, &model_bin);
    if(CY_RSLT_SUCCESS != result)
    {
        printf("ERROR: initialization of the ML validation failed!\r\n");
        handle_error();
    }

    /* Initialize the timer used to determine inference cycles */
    result = elapsed_timer_init();
    if(CY_RSLT_SUCCESS != result)
    {
        printf("ERROR: initialization of elapsed timer failed!\r\n");
        handle_error();
    }

    for (;;)
    {
#ifdef USE_STREAM_DATA
        result = ml_validation_stream_task(&stream_interface);
#else
        result = ml_validation_local_task();
#endif /* USE_STREAM_DATA */

        if (CY_RSLT_SUCCESS == result)
        {
            printf("\n\rProfiling completed!\n\r");
        }
        else
        {
            printf("\n\rProfiling task failed!\n\r");
        }

#ifndef USE_STREAM_DATA
        /* Wait for the device to print out the results */
        Cy_SysLib_Delay(DELAY_TIME_MS);

        /* Only run the local regression once */
        WAIT_FOR_TX_COMPLETE();
        while (1)
        {
            Cy_SysPm_CpuEnterDeepSleep(CY_SYSPM_WAIT_FOR_INTERRUPT);
        }
#endif /* USE_STREAM_DATA */

        printf("Restarting...\r\n");
    }
}

/* [] END OF FILE */
