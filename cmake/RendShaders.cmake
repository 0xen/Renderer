# Offline HLSL compilation through dxc from the Vulkan SDK: SPIR-V for the
# Vulkan backend and, on Windows, DXIL for the D3D12 backend from the SAME
# sources (assets/shaders/backend.hlsli hides the differences). Shared by
# the engine's own shaders (assets/) and by any project that consumes the
# engine and ships its own HLSL — there is no runtime shader compilation.
find_program(REND_DXC_EXECUTABLE
    NAMES dxc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    DOC "DirectX Shader Compiler (emits the SPIR-V artifacts for the Vulkan backend)")
if(NOT REND_DXC_EXECUTABLE)
    message(FATAL_ERROR "dxc not found; set VULKAN_SDK or REND_DXC_EXECUTABLE")
endif()
message(STATUS "Shader compiler: ${REND_DXC_EXECUTABLE}")

# DXIL artifacts (<name>.<stage>.dxil beside every .spv) for the D3D12
# backend. The same dxc emits them; REND_DXIL=OFF skips the rule.
if(WIN32)
    option(REND_DXIL "Also compile every shader stage to DXIL for the D3D12 backend" ON)
else()
    set(REND_DXIL OFF)
endif()

# rend_add_shaders(<target>
#     SOURCE_DIR <dir>        directory holding the .hlsl sources
#     OUTPUT_DIR <dir>        where the .spv artifacts land (generator
#                             expressions such as $<CONFIG> are fine)
#     STAGES <stage>...       "source|entry|profile|artifact[|-Ddefines]"
#     [INCLUDES <file>...]    shared .hlsli every stage depends on, relative
#                             to SOURCE_DIR)
#
# Creates an ALL custom target named <target> that compiles every stage;
# make executables that load the artifacts depend on it with
# add_dependencies(). dxc resolves #include paths relative to the source
# file itself, so INCLUDES only declares rebuild dependencies.
function(rend_add_shaders target)
    cmake_parse_arguments(ARG "" "SOURCE_DIR;OUTPUT_DIR" "STAGES;INCLUDES" ${ARGN})
    if(NOT ARG_SOURCE_DIR OR NOT ARG_OUTPUT_DIR OR NOT ARG_STAGES)
        message(FATAL_ERROR "rend_add_shaders(${target}): SOURCE_DIR, OUTPUT_DIR and STAGES are required")
    endif()

    set(include_paths "")
    foreach(include IN LISTS ARG_INCLUDES)
        list(APPEND include_paths "${ARG_SOURCE_DIR}/${include}")
    endforeach()

    set(artifacts "")
    set(sources "")
    foreach(stage IN LISTS ARG_STAGES)
        string(REPLACE "|" ";" parts "${stage}")
        list(LENGTH parts partCount)
        if(partCount LESS 4)
            message(FATAL_ERROR "rend_add_shaders(${target}): malformed stage '${stage}'")
        endif()
        list(GET parts 0 source)
        list(GET parts 1 entrypoint)
        list(GET parts 2 profile)
        list(GET parts 3 artifact)
        set(defines "")
        if(partCount GREATER 4)
            list(GET parts 4 defines)
        endif()

        set(source_path "${ARG_SOURCE_DIR}/${source}")
        set(artifact_path "${ARG_OUTPUT_DIR}/${artifact}")

        add_custom_command(
            OUTPUT "${artifact_path}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_OUTPUT_DIR}"
            # -fspv-entrypoint-name renames the emitted entry point to "main"
            # so every stage looks the same to the pipeline layer; it must
            # follow -E, which still names the HLSL function.
            COMMAND "${REND_DXC_EXECUTABLE}"
                    -spirv -fspv-target-env=vulkan1.3
                    -T ${profile} -E ${entrypoint} -fspv-entrypoint-name=main
                    ${defines}
                    -Fo "${artifact_path}" "${source_path}"
            DEPENDS "${source_path}" ${include_paths}
            COMMENT "dxc ${source} [${profile} ${entrypoint}] -> ${artifact}"
            VERBATIM)
        list(APPEND artifacts "${artifact_path}")
        list(APPEND sources "${source_path}")

        if(REND_DXIL)
            # Same stage to DXIL: no -spirv, same entry/profile/defines.
            # The [[vk::*]] attributes stay in the sources for the SPIR-V
            # build and are ignored here (warning silenced).
            string(REGEX REPLACE "[.]spv$" ".dxil" dxil_artifact "${artifact}")
            set(dxil_path "${ARG_OUTPUT_DIR}/${dxil_artifact}")
            add_custom_command(
                OUTPUT "${dxil_path}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_OUTPUT_DIR}"
                COMMAND "${REND_DXC_EXECUTABLE}"
                        -T ${profile} -E ${entrypoint} -Wno-ignored-attributes
                        ${defines}
                        -Fo "${dxil_path}" "${source_path}"
                DEPENDS "${source_path}" ${include_paths}
                COMMENT "dxc ${source} [${profile} ${entrypoint}] -> ${dxil_artifact}"
                VERBATIM)
            list(APPEND artifacts "${dxil_path}")
        endif()
    endforeach()
    list(REMOVE_DUPLICATES sources)

    # The .hlsl sources are listed for IDE browsing only — Visual Studio
    # would otherwise hand them to its built-in FXC rule instead of dxc.
    set_source_files_properties(${sources} ${include_paths} PROPERTIES VS_TOOL_OVERRIDE "None")
    add_custom_target(${target} ALL DEPENDS ${artifacts} SOURCES ${sources} ${include_paths})
endfunction()
