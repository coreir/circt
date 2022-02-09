add_circt_dialect_library(CIRCTSMT
  #SMTFolds.cpp
  SMTOps.cpp
  #SMTAnalysis.cpp
  SMTDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/SMT

  DEPENDS
  CIRCTHW
  MLIRSMTIncGen
  #MLIRSMTEnumsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRInferTypeOpInterface
   )

#add_dependencies(circt-headers MLIRSMTIncGen MLIRSMTEnumsIncGen)
add_dependencies(circt-headers MLIRSMTIncGen)
