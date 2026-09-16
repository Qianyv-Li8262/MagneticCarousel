if(NOT DEFINED GENERATOR)
    message(FATAL_ERROR "GENERATOR is required")
endif()

if(NOT DEFINED OUTPUT)
    message(FATAL_ERROR "OUTPUT is required")
endif()

if(NOT DEFINED TARGET)
    message(FATAL_ERROR "TARGET is required")
endif()

get_filename_component(OUTPUT_DIR "${OUTPUT}" DIRECTORY)
file(MAKE_DIRECTORY "${OUTPUT_DIR}")

execute_process(
    COMMAND "${GENERATOR}" "--${TARGET}"
    RESULT_VARIABLE GENERATOR_RESULT
    OUTPUT_FILE "${OUTPUT}"
    ERROR_VARIABLE GENERATOR_ERROR
)

if(NOT GENERATOR_RESULT EQUAL 0)
    file(REMOVE "${OUTPUT}")
    message(
        FATAL_ERROR
        "Coefficient generation failed:\n${GENERATOR_ERROR}"
    )
endif()