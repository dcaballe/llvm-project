//===- OperationCache.cpp - Operation deduplication cache -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OperationCache.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/Support/DebugLog.h"

using namespace mlir;

bool BlockScopedConstantLikeCache::isCacheable(Operation *op) const {
  return op && op->hasTrait<OpTrait::ConstantLike>();
}

OperationCache::CacheLookupResult
BlockScopedConstantLikeCache::lookupOrInsertIntoCache(
    Operation *op, Block *scopeBlock, Block::iterator insertionPoint) {
  if (!isCacheable(op))
    return {nullptr, false};

  // Operation needs to be scoped to its insertion block.
  if (!scopeBlock) {
    LDBG() << "[BlockScopedConstantLikeCache]: Can't lookup or insert op "
              "with no scope: "
           << *op << "\n";
    return {nullptr, false};
  }

  // Look up the scoped operation.
  ScopedCacheOp key = {scopeBlock, op};
  Operation *&cachedOp = constantOpCache[key];

  if (cachedOp) {
    assert(cachedOp->getBlock() == scopeBlock &&
           "cached op was moved to a different block and was not invalidated");
    assert(OperationEquivalence::isEquivalentTo(
               cachedOp, op, OperationEquivalence::IgnoreLocations) &&
           "cached op was modified and was not invalidated");

    // A found op is only reusable if it dominates the caller's insertion point
    // within the block. If it doesn't dominate, decline to reuse it.
    if (!cachedOp->isBeforeInBlock(insertionPoint)) {
      LDBG() << "[BlockScopedConstantLikeCache]: Equivalent op found but not "
                "reusable at the insertion point: "
             << *cachedOp << "\n";
      return {nullptr, false};
    }

    LDBG() << "[BlockScopedConstantLikeCache]: Op found in cache: " << *cachedOp
           << "\n";
    return {cachedOp, false};
  }

  // No cached operation found so add it to the cache. The caller will insert
  // the op into the block following the cache insertion policy.
  cachedOp = op;

  // This cache policy hoists new constant-like ops toward the beginning of the
  // block to maximize reuse but the caller will use the result at
  // `insertionPoint` so the inserted op must dominate it. Honor the hoist
  // point only when it does not fall after `insertionPoint`. Otherwise insert
  // right at `insertionPoint`.
  Block::iterator hoistPoint = getInsertionPoint(scopeBlock);
  bool hoistDominates =
      insertionPoint == scopeBlock->end() ||
      (hoistPoint != scopeBlock->end() &&
       hoistPoint->isBeforeInBlock(insertionPoint));
  Block::iterator actualInsertPoint = hoistPoint;

  if (!hoistDominates) {
    // The cache policy's insertion point does not dominate the caller's
    // insertion point, so it cannot be honored here. Fall back to the caller's
    // insertion point.
    LDBG() << "[BlockScopedConstantLikeCache]: insertion policy could not be "
              "honored; its point does not dominate the requested insertion "
              "point, falling back to it\n";
    actualInsertPoint = insertionPoint;
  }

  LDBG() << "[BlockScopedConstantLikeCache]: Op added to cache: " << *op
         << "\n";

  return {op, /*inserted=*/true, scopeBlock, actualInsertPoint};
}

Block::iterator
BlockScopedConstantLikeCache::getInsertionPoint(Block *block) const {
  // Fast path: we've already inserted at least one cached op into `block`, so
  // the next op should land immediately after the last one we recorded
  // (preserving chronological order of insertions). Defensive: verify the
  // recorded op is still in this block; otherwise fall back to the slow path.
  // This guards against out-of-band erasures/moves by code that isn't aware of
  // the cache.
  auto cacheIt = cacheInsertionPoints.find(block);
  if (cacheIt != cacheInsertionPoints.end()) {
    Operation *lastCached = cacheIt->second;
    if (lastCached && lastCached->getBlock() == block)
      return std::next(Block::iterator(lastCached));
  }

  // Slow path: nothing recorded yet. Fall back to walking past any
  // constant-like ops already at the block's begin (these may have been
  // hoisted by a prior run or by a pass that didn't use this cache). The
  // returned iterator points at the first non-constant op.
  auto it = block->begin();
  while (it != block->end() && it->hasTrait<OpTrait::ConstantLike>())
    ++it;
  return it;
}

void BlockScopedConstantLikeCache::notifyInserted(Operation *op, Block *block) {
  assert(op && block && "null op/block passed to notifyInserted");
  assert(op->getBlock() == block &&
         "op was not actually inserted into the expected block");
  assert(constantOpCache.lookup(ScopedCacheOp{block, op}) == op &&
         "notifyInserted called for an op not recorded in the cache");

  // Advance the cache's insertion point only when the op landed immediately
  // after the current insertion point (cache insertion's policy). If the cached
  // op landed somewhere else to satisfy dominance requirements, keep the
  // current insertion point.
  Operation *&cacheInsertionPoint = cacheInsertionPoints[block];
  if (cacheInsertionPoint && cacheInsertionPoint->getBlock() == block &&
      std::next(Block::iterator(cacheInsertionPoint)) != Block::iterator(op)) {
    return;
  }

  cacheInsertionPoint = op;
}

void BlockScopedConstantLikeCache::invalidate(Operation *op, Block *scopeBlock) {
  if (!isCacheable(op))
    return;

  // If no scope block is provided, use the current block of the operation.
  if (!scopeBlock) {
    scopeBlock = op->getBlock();
    // If the operation is not scoped to a block, we can't invalidate it.
    if (!scopeBlock) {
      LDBG() << "[BlockScopedConstantLikeCache]: Can't invalidate op without "
                "scope: "
             << *op << "\n";
      return;
    }
  }

  ScopedCacheOp key = {scopeBlock, op};
  constantOpCache.erase(key);

  // If the erased op was the cache insertion point for its block, drop it
  // so it'll be recomputed on the next use.
  auto cacheIt = cacheInsertionPoints.find(scopeBlock);
  if (cacheIt != cacheInsertionPoints.end() && cacheIt->second == op)
    cacheInsertionPoints.erase(cacheIt);
}

void BlockScopedConstantLikeCache::clear() {
  LDBG() << "[BlockScopedConstantLikeCache]: Cache cleared\n";
  constantOpCache.clear();
  cacheInsertionPoints.clear();
}

bool IsolatedRegionScopedConstantLikeCache::isCacheable(Operation *op) const {
  return op && op->hasTrait<OpTrait::ConstantLike>();
}

Region *IsolatedRegionScopedConstantLikeCache::getScopeRegion(Block *block) const {
  while (Region *region = block->getParent()) {
    Operation *parentOp = region->getParentOp();
    if (!parentOp)
      return nullptr;

    // A constant cannot be reused past a region that is isolated from above, so
    // that region bounds the scope.
    if (parentOp->mightHaveTrait<OpTrait::IsIsolatedFromAbove>())
      return region;

    // A dialect can require constants to stay inside a region that is not
    // isolated from above. Honor that, otherwise hoisting past it would move a
    // constant the dialect expects to find in place.
    auto *interface = interfaces.getInterfaceFor(parentOp);
    if (LLVM_UNLIKELY(interface && interface->shouldMaterializeInto(region)))
      return region;

    // The enclosing op is still detached, so the final scope is not known yet
    // and caching against this region would be keyed on a scope that changes
    // once the op is linked. Builders reach this while running an op's body
    // builder, before the op is inserted.
    if (!parentOp->getBlock())
      return nullptr;

    block = parentOp->getBlock();
  }
  return nullptr;
}

OperationCache::CacheLookupResult
IsolatedRegionScopedConstantLikeCache::lookupOrInsertIntoCache(
    Operation *op, Block *scopeBlock, Block::iterator insertionPoint) {
  if (!isCacheable(op))
    return {nullptr, false};

  if (!scopeBlock) {
    LDBG() << "[IsolatedRegionScopedConstantLikeCache]: Can't lookup or insert op "
              "with no scope: "
           << *op << "\n";
    return {nullptr, false};
  }

  Region *scopeRegion = getScopeRegion(scopeBlock);
  if (!scopeRegion || scopeRegion->empty()) {
    LDBG() << "[IsolatedRegionScopedConstantLikeCache]: No scope region with an entry "
              "block for op: "
           << *op << "\n";
    return {nullptr, false};
  }
  Block *entryBlock = &scopeRegion->front();

  // Look up the scoped operation.
  ScopedCacheOp key = {scopeRegion, op};
  Operation *&cachedOp = constantOpCache[key];

  if (cachedOp) {
    assert(getScopeRegion(cachedOp->getBlock()) == scopeRegion &&
           "cached op was moved out of its scope and was not invalidated");
    assert(OperationEquivalence::isEquivalentTo(
               cachedOp, op, OperationEquivalence::IgnoreLocations) &&
           "cached op was modified and was not invalidated");

    // A cached op sitting in the same block as the requested insertion point
    // only dominates it when it comes first. Anywhere else in the scope it is in
    // the entry block, which dominates the whole region.
    if (cachedOp->getBlock() == scopeBlock &&
        !cachedOp->isBeforeInBlock(insertionPoint)) {
      LDBG() << "[IsolatedRegionScopedConstantLikeCache]: Equivalent op found but not "
                "reusable at the insertion point: "
             << *cachedOp << "\n";
      return {nullptr, false};
    }

    LDBG() << "[IsolatedRegionScopedConstantLikeCache]: Op found in cache: "
           << *cachedOp << "\n";
    return {cachedOp, false};
  }

  // No cached operation found so add it to the cache. The caller will insert
  // the op into the entry block following the cache insertion policy.
  cachedOp = op;

  // Hoist toward the beginning of the scope's entry block to maximize reuse.
  // When the caller is building into that same block, the hoist point still has
  // to dominate its insertion point, so clamp to it when it does not.
  Block::iterator hoistPoint = getInsertionPoint(entryBlock);
  Block::iterator actualInsertPoint = hoistPoint;
  if (entryBlock == scopeBlock) {
    bool hoistDominates = insertionPoint == entryBlock->end() ||
                          (hoistPoint != entryBlock->end() &&
                           hoistPoint->isBeforeInBlock(insertionPoint));
    if (!hoistDominates) {
      LDBG() << "[IsolatedRegionScopedConstantLikeCache]: insertion policy could not "
                "be honored; its point does not dominate the requested "
                "insertion point, falling back to it\n";
      actualInsertPoint = insertionPoint;
    }
  }

  LDBG() << "[IsolatedRegionScopedConstantLikeCache]: Op added to cache: " << *op
         << "\n";

  return {op, /*inserted=*/true, entryBlock, actualInsertPoint};
}

Block::iterator
IsolatedRegionScopedConstantLikeCache::getInsertionPoint(Block *entryBlock) const {
  auto cacheIt = cacheInsertionPoints.find(entryBlock);
  if (cacheIt != cacheInsertionPoints.end()) {
    Operation *lastCached = cacheIt->second;
    if (lastCached && lastCached->getBlock() == entryBlock)
      return std::next(Block::iterator(lastCached));
  }

  auto it = entryBlock->begin();
  while (it != entryBlock->end() && it->hasTrait<OpTrait::ConstantLike>())
    ++it;
  return it;
}

void IsolatedRegionScopedConstantLikeCache::notifyInserted(Operation *op,
                                                     Block *block) {
  assert(op && block && "null op/block passed to notifyInserted");
  assert(op->getBlock() == block &&
         "op was not actually inserted into the expected block");
  assert(constantOpCache.lookup(ScopedCacheOp{getScopeRegion(block), op}) ==
             op &&
         "notifyInserted called for an op not recorded in the cache");

  Operation *&cacheInsertionPoint = cacheInsertionPoints[block];
  if (cacheInsertionPoint && cacheInsertionPoint->getBlock() == block &&
      std::next(Block::iterator(cacheInsertionPoint)) != Block::iterator(op)) {
    return;
  }

  cacheInsertionPoint = op;
}

void IsolatedRegionScopedConstantLikeCache::invalidate(Operation *op,
                                                 Block *scopeBlock) {
  if (!isCacheable(op))
    return;

  if (!scopeBlock) {
    scopeBlock = op->getBlock();
    if (!scopeBlock) {
      LDBG() << "[IsolatedRegionScopedConstantLikeCache]: Can't invalidate op without "
                "scope: "
             << *op << "\n";
      return;
    }
  }

  Region *scopeRegion = getScopeRegion(scopeBlock);
  if (!scopeRegion)
    return;

  constantOpCache.erase(ScopedCacheOp{scopeRegion, op});

  // If the erased op was the cache insertion point for its entry block, drop it
  // so it'll be recomputed on the next use.
  if (scopeRegion->empty())
    return;
  Block *entryBlock = &scopeRegion->front();
  auto cacheIt = cacheInsertionPoints.find(entryBlock);
  if (cacheIt != cacheInsertionPoints.end() && cacheIt->second == op)
    cacheInsertionPoints.erase(cacheIt);
}

void IsolatedRegionScopedConstantLikeCache::invalidate(Block *block) {
  for (Operation &op : *block)
    invalidate(&op, block);
  cacheInsertionPoints.erase(block);
}

void IsolatedRegionScopedConstantLikeCache::clear() {
  LDBG() << "[IsolatedRegionScopedConstantLikeCache]: Cache cleared\n";
  constantOpCache.clear();
  cacheInsertionPoints.clear();
}
