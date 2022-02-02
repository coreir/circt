add_circt_dialect(SMT smt)
add_circt_doc(SMT  -gen-dialect-doc SMT Dialects/)

set(LLVM_TARGET_DEFINITIONS SMT.td)
#mlir_tablegen(SMTEnums.h.inc -gen-enum-decls)
#mlir_tablegen(SMTEnums.cpp.inc -gen-enum-defs)
#add_public_tablegen_target(MLIRSMTEnumsIncGen)
