add_circt_conversion_library(CIRCTSMTToComb
        SMTToComb.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTComb
  CIRCTSMT
  )
