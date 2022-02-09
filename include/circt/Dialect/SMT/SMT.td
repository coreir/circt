//===- SMT.td - SMT dialect definition --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the SMT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SMT_TD
#define SMT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def SMTDialect : Dialect {
  let name = "smt";

  let summary = "Types and operations for smt dialect";
  let description = [{
    This dialect defines the `smt` dialect(s), which are 1:1 combinational operators with SMT
  }];
  //let hasConstantMaterializer = 1;
  //let hasConstantMaterializer = 0;
  let cppNamespace = "::circt::smt";
}

// Base class for the operation in this dialect.
class SMTOp<string mnemonic, list<OpTrait> traits = []> :
    Op<SMTDialect, mnemonic, traits>;

include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/SMT/SMTDefOps.td"

#endif // SMT_TD
