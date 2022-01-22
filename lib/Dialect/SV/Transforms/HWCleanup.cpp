//===- HWCleanup.cpp - HW Cleanup Pass ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs various cleanups and canonicalization
// transformations for hw.module bodies.  This is intended to be used early in
// the HW/SV pipeline to expose optimization opportunities that require global
// analysis.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace circt;

//===----------------------------------------------------------------------===//
// Helper utilities
//===----------------------------------------------------------------------===//

namespace {

/// Check the equivalence of operations by doing a deep comparison of operands
/// and attributes, but does not compare the content of any regions attached to
/// each op.
struct AlwaysLikeOpInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return mlir::OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/mlir::OperationEquivalence::directHashValue,
        /*hashResults=*/mlir::OperationEquivalence::ignoreHashValue,
        mlir::OperationEquivalence::IgnoreLocations);
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    // Trivially the same.
    if (lhs == rhs)
      return true;
    // Filter out tombstones and empty ops.
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    // Compare attributes.
    if (lhs->getName() != rhs->getName() ||
        lhs->getAttrDictionary() != rhs->getAttrDictionary() ||
        lhs->getNumOperands() != rhs->getNumOperands())
      return false;
    // Compare operands.
    for (auto operandPair : llvm::zip(lhs->getOperands(), rhs->getOperands())) {
      Value lhsOperand = std::get<0>(operandPair);
      Value rhsOperand = std::get<1>(operandPair);
      if (lhsOperand != rhsOperand)
        return false;
    }
    // The two AlwaysOps are similar enough to be combined.
    return true;
  }
};

} // end anonymous namespace

// Merge two regions together. These regions must only have a one block.
static void mergeRegions(Region *region1, Region *region2) {
  assert(region1->getBlocks().size() <= 1 && region2->getBlocks().size() <= 1 &&
         "Can only merge regions with a single block");
  if (region1->empty()) {
    // If both regions are empty, move on to the next pair of regions
    if (region2->empty())
      return;
    // If the first region has no block, move the second region's block over.
    region1->getBlocks().splice(region1->end(), region2->getBlocks());
    return;
  }

  // If the second region is not empty, splice its block into the start of the
  // first region.
  if (!region2->empty()) {
    auto &block1 = region1->front();
    auto &block2 = region2->front();
    block1.getOperations().splice(block1.begin(), block2.getOperations());
  }
}

static void peelConjunction(Value cond,
                            llvm::SmallSetVector<Value, 4> &conjunctions,
                            llvm::SmallSetVector<Operation *, 4> &operations) {
  auto andOp = dyn_cast_or_null<comb::AndOp>(cond.getDefiningOp());
  if (!andOp) {
    conjunctions.insert(cond);
    return;
  }

  operations.insert(andOp);
  llvm::for_each(andOp.getOperands(), [&](Value cond) {
    peelConjunction(cond, conjunctions, operations);
  });
}

static std::tuple<llvm::SmallSetVector<Value, 4>,
                  llvm::SmallSetVector<Value, 4>, llvm::SmallVector<Value>,
                  llvm::SmallSetVector<Operation *, 4>>
splitCondjunction(Value cond1, Value cond2) {
  llvm::SmallSetVector<Value, 4> conjunction1;
  llvm::SmallSetVector<Value, 4> conjunction2;
  llvm::SmallSetVector<Operation *, 4> operations;

  peelConjunction(cond1, conjunction1, operations);
  peelConjunction(cond2, conjunction2, operations);

  SmallVector<Value> commonConjunction;
  for (auto c1 : conjunction1)
    if (conjunction2.contains(c1))
      commonConjunction.push_back(c1);

  if (commonConjunction.empty())
    return {conjunction1, conjunction2, commonConjunction, operations};

  if (commonConjunction.size() == conjunction1.size() &&
      commonConjunction.size() == conjunction2.size()) {
    return {{}, {}, commonConjunction, operations};
  }

  for (auto c : commonConjunction) {
    conjunction1.remove(c);
    conjunction2.remove(c);
  }

  return {conjunction1, conjunction2, commonConjunction, operations};
}
static void cleanUpUses(llvm::SmallSetVector<Operation *, 4> &operations) {
  for (Operation *op : operations)
    if (op->use_empty())
      op->erase();
}

//===----------------------------------------------------------------------===//

// HWCleanupPass
//===----------------------------------------------------------------------===//

namespace {
struct HWCleanupPass : public sv::HWCleanupBase<HWCleanupPass> {
  void runOnOperation() override;

  void runOnRegionsInOp(Operation &op);
  void runOnGraphRegion(Region &region);
  void runOnProceduralRegion(Region &region);

private:
  /// Inline all regions from the second operation into the first and delete the
  /// second operation.
  void mergeOperationsIntoFrom(Operation *op1, Operation *op2) {
    assert(op1 != op2 && "Cannot merge an op into itself");
    for (size_t i = 0, e = op1->getNumRegions(); i != e; ++i)
      mergeRegions(&op1->getRegion(i), &op2->getRegion(i));

    op2->erase();
    anythingChanged = true;
  }

  sv::IfOp mergeIfOpConditions(sv::IfOp op1, sv::IfOp op2);

  sv::IfOp mergeIfOps(sv::IfOp op1, sv::IfOp op2) {
    assert(op1 != op2 && "Cannot merge an op into itself");

    // If conditions of op1 and op2 are equal, we can just merge them.
    if (op1.cond() == op2.cond()) {
      anythingChanged = true;
      mergeOperationsIntoFrom(op1, op2);
      return op1;
    }

    // TOOD: Handle even when they have else blocks
    if (op1.hasElse() || op2.hasElse())
      return op1;

    return mergeIfOpConditions(op1, op2);
  }

  bool anythingChanged;
};
} // end anonymous namespace

void HWCleanupPass::runOnOperation() {
  // Keeps track if anything changed during this pass, used to determine if
  // the analyses were preserved.
  anythingChanged = false;
  runOnGraphRegion(getOperation().getBody());

  // If we did not change anything in the graph mark all analysis as
  // preserved.
  if (!anythingChanged)
    markAllAnalysesPreserved();
}

/// Recursively process all of the regions in the specified op, dispatching to
/// graph or procedural processing as appropriate.
void HWCleanupPass::runOnRegionsInOp(Operation &op) {
  if (op.hasTrait<sv::ProceduralRegion>()) {
    for (auto &region : op.getRegions())
      runOnProceduralRegion(region);
  } else {
    for (auto &region : op.getRegions())
      runOnGraphRegion(region);
  }
}

/// Run simplifications on the specified graph region.
void HWCleanupPass::runOnGraphRegion(Region &region) {
  if (region.getBlocks().size() != 1)
    return;
  Block &body = region.front();

  // A set of operations in the current block which are mergable. Any
  // operation in this set is a candidate for another similar operation to
  // merge in to.
  DenseSet<Operation *, AlwaysLikeOpInfo> alwaysFFOpsSeen;
  llvm::SmallDenseMap<Attribute, Operation *, 4> ifdefOps;
  sv::InitialOp initialOpSeen;
  sv::AlwaysCombOp alwaysCombOpSeen;

  for (Operation &op : llvm::make_early_inc_range(body)) {
    // Merge alwaysff and always operations by hashing them to check to see if
    // we've already encountered one.  If so, merge them and reprocess the body.
    if (isa<sv::AlwaysOp, sv::AlwaysFFOp>(op)) {
      // Merge identical alwaysff's together and delete the old operation.
      auto itAndInserted = alwaysFFOpsSeen.insert(&op);
      if (itAndInserted.second)
        continue;
      auto *existingAlways = *itAndInserted.first;
      mergeOperationsIntoFrom(&op, existingAlways);

      *itAndInserted.first = &op;
      continue;
    }

    // Merge graph ifdefs anywhere in the module.
    if (auto ifdefOp = dyn_cast<sv::IfDefOp>(op)) {
      auto *&entry = ifdefOps[ifdefOp.condAttr()];
      if (entry)
        mergeOperationsIntoFrom(ifdefOp, entry);

      entry = ifdefOp;
      continue;
    }

    // Merge initial ops anywhere in the module.
    if (auto initialOp = dyn_cast<sv::InitialOp>(op)) {
      if (initialOpSeen)
        mergeOperationsIntoFrom(initialOp, initialOpSeen);
      initialOpSeen = initialOp;
      continue;
    }

    // Merge always_comb ops anywhere in the module.
    if (auto alwaysComb = dyn_cast<sv::AlwaysCombOp>(op)) {
      if (alwaysCombOpSeen)
        mergeOperationsIntoFrom(alwaysComb, alwaysCombOpSeen);
      alwaysCombOpSeen = alwaysComb;
      continue;
    }
  }

  for (Operation &op : llvm::make_early_inc_range(body)) {
    // Recursively process any regions in the op.
    if (op.getNumRegions() != 0)
      runOnRegionsInOp(op);
  }
}

/// Run simplifications on the specified procedural region.
void HWCleanupPass::runOnProceduralRegion(Region &region) {
  if (region.getBlocks().size() != 1)
    return;
  Block &body = region.front();

  Operation *lastSideEffectingOp = nullptr;
  for (auto it = body.begin(), end = body.end(); it != end; it++) {
    Operation &op = *it;
    // Merge procedural ifdefs with neighbors in the procedural region.
    if (auto ifdef = dyn_cast<sv::IfDefProceduralOp>(op)) {
      if (auto prevIfDef =
              dyn_cast_or_null<sv::IfDefProceduralOp>(lastSideEffectingOp)) {
        if (ifdef.cond() == prevIfDef.cond()) {
          // We know that there are no side effective operations between the
          // two, so merge the first one into this one.
          mergeOperationsIntoFrom(ifdef, prevIfDef);
        }
      }
    }

    // Merge 'if' operations with the same condition.
    if (auto ifop = dyn_cast<sv::IfOp>(op)) {
      if (auto prevIf = dyn_cast_or_null<sv::IfOp>(lastSideEffectingOp)) {
        auto newOp = mergeIfOps(ifop, prevIf);
        if (newOp != ifop) {
          lastSideEffectingOp = newOp;
          it = newOp->getIterator();
          continue;
        }
      }
    }

    // Keep track of the last side effecting operation we've seen.
    if (!mlir::MemoryEffectOpInterface::hasNoEffect(&op))
      lastSideEffectingOp = &op;
  }

  for (Operation &op : llvm::make_early_inc_range(body)) {
    // Recursively process any regions in the op.
    if (op.getNumRegions() != 0)
      runOnRegionsInOp(op);
  }
}

sv::IfOp HWCleanupPass::mergeIfOpConditions(sv::IfOp op1, sv::IfOp op2) {
  // op2:
  //   if a1 & a2 &... & an & c1 & c2 .. & cn {e1}
  // op1:
  //   if b1 & b2 &... & bn & c1 & c2 .. & cn {e2}
  // ====>
  //   if c1 & c2 .. & cn {
  //     if a1 & a2 & ... & an {e1}
  //     if b1 & b2 & ... & bn {e2}
  //   }

  // op1.cond() = conjunction1 /\ commonConjunction
  // op2.cond() = conjunction2 /\ commonConjunction
  auto [conjunction1, conjunction2, commonConjunction, mightBeDeleted] =
      splitCondjunction(op1.cond(), op2.cond());

  // If there is nothing in common, we cannot merge.
  if (commonConjunction.empty())
    return op1;

  // If both conjunction1 and conjunction2 are empty, it means the conditions
  // are actually equivalent.
  if (conjunction1.empty() && conjunction2.empty()) {
    mergeOperationsIntoFrom(op1, op2);
    return op1;
  }

  OpBuilder builder(op1.getContext());
  auto i1Type = op1.cond().getType();

  auto generateCondValue =
      [&](Location loc, llvm::SmallSetVector<Value, 4> &conjunction) -> Value {
    if (conjunction.empty())
      return builder.create<hw::ConstantOp>(loc, i1Type, 1);

    return builder.createOrFold<comb::AndOp>(loc, i1Type,
                                             conjunction.takeVector());
  };
  if (conjunction2.empty()) {
    // Move op1 to the end of op2's then block.
    builder.setInsertionPointToEnd(op2.getThenBlock());
    auto newCond1 = generateCondValue(op1.cond().getLoc(), conjunction1);
    op1.setOperand(newCond1);
    // op1 might contain ops defined between op2 and op1, we have to move op2
    // to the position of op1 to ensure that the dominance doesn't break.
    op2->moveBefore(op1);
    op1->moveBefore(op2.getThenBlock(), op2.getThenBlock()->end());
    cleanUpUses(mightBeDeleted);
    return op2;
  }

  if (conjunction1.empty()) {
    // Move op2 to the start of op1's then block.
    builder.setInsertionPointToStart(op1.getThenBlock());
    auto newCond2 = generateCondValue(op2.cond().getLoc(), conjunction2);
    op2.setOperand(newCond2);
    op2->moveAfter(op1.getThenBlock(), op1.getThenBlock()->begin());
    cleanUpUses(mightBeDeleted);
    return op1;
  }

  builder.setInsertionPoint(op1);
  auto newCond1 = generateCondValue(op1.getLoc(), conjunction1);
  auto newCond2 = generateCondValue(op2.getLoc(), conjunction2);
  auto cond =
      builder.createOrFold<comb::AndOp>(op1.getLoc(), commonConjunction);
  auto newIf = builder.create<sv::IfOp>(op2.getLoc(), cond, [&]() {});

  op2->moveBefore(newIf.getThenBlock(), newIf.getThenBlock()->begin());
  op1->moveAfter(op2);
  op1.setOperand(newCond1);
  op2.setOperand(newCond2);
  cleanUpUses(mightBeDeleted);

  return newIf;
}

std::unique_ptr<Pass> circt::sv::createHWCleanupPass() {
  return std::make_unique<HWCleanupPass>();
}
