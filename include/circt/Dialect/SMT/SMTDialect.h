//===- SMTDialect.h - SMT dialect declaration -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SMT MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTDIALECT_H
#define CIRCT_DIALECT_SMT_SMTDIALECT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"

// Pull in the Dialect definition.
#include "circt/Dialect/SMT/SMTDialect.h.inc"

// Pull in all enum type definitions and utility function declarations.
// #include "circt/Dialect/SMT/SMTEnums.h.inc"

#endif // CIRCT_DIALECT_SMT_SMTDIALECT_H
