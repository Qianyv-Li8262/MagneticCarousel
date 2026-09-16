if(NOT DEFINED GENERATOR)
    message(FATAL_ERROR "GENERATOR was not provided")
endif()

if(NOT DEFINED OUTPUT)
    message(FATAL_ERROR "OUTPUT was not provided")
endif()

get_filename_component(
    OUTPUT_DIR
    "${OUTPUT}"
    DIRECTORY
)

file(MAKE_DIRECTORY "${OUTPUT_DIR}")

execute_process(
    COMMAND "${GENERATOR}" --cpu
    OUTPUT_FILE "${OUTPUT}"
    RESULT_VARIABLE generator_result
)

if(NOT generator_result STREQUAL "0")
    file(REMOVE "${OUTPUT}")
    message(
        FATAL_ERROR
        "Field coefficient generator failed: ${generator_result}"
    )
endif()