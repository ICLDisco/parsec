# Sources that belong to the comm framework itself.  Backend-specific files are
# discovered from each module directory through its ValidateModule.CMake file.
set(MCA_${COMPONENT}_SOURCES mca/comm/comm.c)

# Install the framework header so in-tree and future out-of-tree comm components
# can build against the same component/module contract.
set_property(TARGET parsec
             APPEND PROPERTY
                    PUBLIC_HEADER_H mca/comm/comm.h)
