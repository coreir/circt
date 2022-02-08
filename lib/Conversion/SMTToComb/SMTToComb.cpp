//===- AffineToStaticlogic.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SMTToComb.h"
#include "../PassDetail.h"
#include "circt/Analysis/DependenceAnalysis.h"
#include "circt/Analysis/SchedulingAnalysis.h"
#include "circt/Dialect/SMT/SMTDialect.h"
#include "circt/Scheduling/Algorithms.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Scheduling/Problems.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LoopUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <mlir/IR/BuiltinDialect.h>

#define DEBUG_TYPE "affine-to-staticlogic"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::memref;
using namespace mlir::scf;
using namespace circt;
using namespace circt::analysis;
using namespace circt::scheduling;
using namespace circt::staticlogic;

namespace {

struct SMTToComb
    : public SMTToCombBase<SMTToComb> {
  void runOnFunction() override {}
};

} // namespace

std::unique_ptr<mlir::Pass> circt::createSMTToComb() {
  return std::make_unique<SMTToComb>();
}
