cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(MyLungs)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/MyLungs")
set(EXTENSION_CATEGORY "Visual Segmentation")
set(EXTENSION_CONTRIBUTORS "Rafal (AGH), Dawid (AGH)")
set(EXTENSION_DESCRIPTION "Plugin for segmenation of lungs from CT scan of the whole body.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/MyLungs.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/MyLungs/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(Lungs_script)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
