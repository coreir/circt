//===- CalyxDialect.cpp - Implement the Calyx dialect ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Calyx dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Calyx/CalyxDialect.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Scheduling/Interfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace circt::calyx;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

namespace {

// We implement the OpAsmDialectInterface so that Calyx dialect operations
// automatically interpret the name attribute on operations as their SSA name.
struct CalyxOpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  /// Get a special name to use when printing the given operation. See
  /// OpAsmInterface.td#getAsmResultNames for usage details and documentation.
  void getAsmResultNames(Operation *op,
                         OpAsmSetValueNameFn setNameFn) const override {}
};

struct CalyxOperatorTypesInterface : public scheduling::OperatorTypesInterface {
  using scheduling::OperatorTypesInterface::OperatorTypesInterface;

  LogicalResult populateOperatorTypes(scheduling::Problem &problem,
                                      Block *block) const override;
};

} // end anonymous namespace

LogicalResult
CalyxOperatorTypesInterface::populateOperatorTypes(scheduling::Problem &problem,
                                                   Block *block) const {
  // Load the Calyx operator library into the problem. This is a very minimal
  // set of arithmetic and memory operators for now. This should ultimately be
  // pulled out into some sort of dialect interface.
  scheduling::Problem::OperatorType combOpr =
      problem.getOrInsertOperatorType("comb");
  problem.setLatency(combOpr, 0);

  scheduling::Problem::OperatorType seqOpr =
      problem.getOrInsertOperatorType("seq");
  problem.setLatency(seqOpr, 1);

  scheduling::Problem::OperatorType tcOpr =
      problem.getOrInsertOperatorType("threecycle");
  problem.setLatency(tcOpr, 3);

  Operation *unsupported;
  WalkResult result = block->walk([&](Operation *op) {
    return TypeSwitch<Operation *, WalkResult>(op)
        .Case<AffineYieldOp, arith::AddIOp, arith::ConstantOp, arith::CmpIOp,
              arith::IndexCastOp, memref::AllocaOp, scf::IfOp, scf::YieldOp>(
            [&](Operation *combOp) {
              // Some known combinational ops.
              problem.setLinkedOperatorType(combOp, combOpr);
              return WalkResult::advance();
            })
        .Case<AffineLoadOp, AffineStoreOp, memref::LoadOp, memref::StoreOp>(
            [&](Operation *seqOp) {
              // Some known sequential ops. In certain cases, reads may be
              // combinational in Calyx, but taking advantage of that is left as
              // a future enhancement.
              problem.setLinkedOperatorType(seqOp, seqOpr);
              return WalkResult::advance();
            })
        .Case<arith::MulIOp>([&](Operation *mcOp) {
          // Some known three-cycle ops.
          problem.setLinkedOperatorType(mcOp, tcOpr);
          return WalkResult::advance();
        })
        .Default([&](Operation *badOp) {
          unsupported = op;
          return WalkResult::interrupt();
        });
  });

  if (result.wasInterrupted())
    return block->getParentOp()->emitError("no operator type for operation: ")
           << *unsupported;

  return success();
}

void CalyxDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/Calyx/Calyx.cpp.inc"
      >();

  // Register interface implementations.
  addInterfaces<CalyxOpAsmDialectInterface, CalyxOperatorTypesInterface>();
}

// Provide implementations for the enums and attributes we use.
#include "circt/Dialect/Calyx/CalyxAttrs.cpp.inc"
#include "circt/Dialect/Calyx/CalyxDialect.cpp.inc"
#include "circt/Dialect/Calyx/CalyxEnums.cpp.inc"
