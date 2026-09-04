# Fetches pugixml (XML parser used by the assetio scene loader), pinned.
include(FetchContent)

FetchContent_Declare(
    pugixml
    GIT_REPOSITORY https://github.com/zeux/pugixml.git
    GIT_TAG v1.16
    GIT_SHALLOW ON
)
FetchContent_MakeAvailable(pugixml)
if(TARGET pugixml-static)
    set_target_properties(pugixml-static PROPERTIES FOLDER "third_party")
endif()
