//===- Interfaces.h - Library of scheduling interfaces ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a library of scheduling interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SCHEDULING_INTERFACES_H
#define CIRCT_SCHEDULING_INTERFACES_H

#include "circt/Scheduling/Problems.h"
#include "mlir/IR/DialectInterface.h"
#include "mlir/Support/LogicalResult.h"

namespace circt {
namespace scheduling {

/// Dialect interface to allow dialects to provide OperatorTypes used in
/// scheduling problems.
class OperatorTypesInterface
    : public mlir::DialectInterface::Base<OperatorTypesInterface> {
public:
  OperatorTypesInterface(Dialect *dialect) : Base(dialect) {}

  /// A dialect should populate the problem with operator types its operations
  /// can implement. For each input operation in the block, the dialect should
  /// set the linked operator type to indicate which input operations can be
  /// supported by its operations. If unsupported input operations are
  /// encountered, failure should be returned.
  ///
  /// For example, a dialect might add a combinational operator type, and link
  /// simple operations from the arithmetic dialect to this operator type.
  virtual mlir::LogicalResult populateOperatorTypes(Problem &problem,
                                                    Block *block) const {
    return success();
  }
};

} // namespace scheduling
} // namespace circt

#endif // CIRCT_SCHEDULING_INTERFACES_H
