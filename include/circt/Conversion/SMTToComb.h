//===- AffineToStaticLogic.h ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_SMTTOCOMB_H_
#define CIRCT_CONVERSION_SMTTOCOMB_H_

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {
std::unique_ptr<mlir::Pass> createSMTToComb();
} // namespace circt

#endif // CIRCT_CONVERSION_SMTTOCOMG_H_
