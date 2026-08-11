//===- OperationCache.h - Operation deduplication cache ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file defines the OperationCache interface for deduplicating equivalent
/// operations.
///
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_OPERATIONCACHE_H
#define MLIR_IR_OPERATIONCACHE_H

#include "mlir/IR/Block.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {

/// Abstract interface for a cache of operations. It deduplicates equivalent
/// operations and chooses their insertion point. A few contract points:
///   * Caching is best-effort: an implementation may decline caching any
///     operation, so clients must not assume a given op will be cached.
///   * Invalidate before mutation: the cache stores `Operation *` references
///     and uses them to compute equivalence. Entries must be invalidated before
///     the referenced op is mutated or erased.
class OperationCache {
public:
  virtual ~OperationCache() = default;

  /// Result of a cache lookup operation.
  struct CacheLookupResult {
    /// Returned operation, either the input op or the cached op. Null if the
    /// op was not cached.
    Operation *op = nullptr;
    /// Set to true if `op` was newly inserted into the cache by this lookup.
    bool inserted = false;
    /// Block for the caller to insert a newly cached `op` into. Valid only when
    /// `insertedInCache()` is true. It may differ from the block passed to
    /// `lookupOrInsertIntoCache` when the cache scope spans several blocks.
    Block *insertionBlock = nullptr;
    /// Insertion point within `insertionBlock` for the caller to insert a newly
    /// cached `op`. Valid only when `insertedInCache()` is true. The cache
    /// guarantees this point dominates the insertion point passed to
    /// `lookupOrInsertIntoCache`.
    Block::iterator insertionPoint = {};

    /// Returns true if an equivalent operation was found in the cache.
    bool foundInCache() const { return op && !inserted; }
    /// Returns true if `op` was newly inserted into the cache by this lookup.
    bool insertedInCache() const { return op && inserted; }
  };

  /// Returns true if `op` can be cached by this cache.
  virtual bool isCacheable(Operation *op) const = 0;

  /// Looks up an operation in the cache that is equivalent to `op` and can be
  /// reused at `insertionPoint` within `scopeBlock`, or inserts `op` into the
  /// cache if no reusable equivalent is found. Whether a found op can be reused
  /// (e.g., whether it dominates `insertionPoint`) is decided by the
  /// implementation. If cached, the caller is responsible for inserting `op`
  /// into the IR at the result's `insertionPoint`, which the cache guarantees
  /// dominates `insertionPoint`.
  virtual CacheLookupResult
  lookupOrInsertIntoCache(Operation *op, Block *scopeBlock,
                          Block::iterator insertionPoint) = 0;

  /// Notifies the cache that `op` was inserted into `block`, so the cache can
  /// update any placement bookkeeping it maintains.
  virtual void notifyInserted(Operation *op, Block *block) = 0;

  /// Invalidates `op` in the cache, if present. If `scopeBlock` is null, the
  /// op's current block is used as scope.
  virtual void invalidate(Operation *op, Block *scopeBlock = nullptr) = 0;

  /// Invalidates all cached operations in `block`.
  virtual void invalidate(Block *block) = 0;

  /// Invalidates all cached operations.
  virtual void clear() = 0;
};

namespace detail {
/// Hashing and comparison for a cache key pairing a scope with an operation.
/// Operations are compared by equivalence ignoring locations, so equivalent ops
/// in the same scope share a single entry.
template <typename ScopeT>
struct ScopedOpCacheInfo {
  using ScopedOp = std::pair<ScopeT, Operation *>;

  static unsigned getHashValue(const ScopedOp &scopedOp) {
    unsigned scopeHash =
        llvm::DenseMapInfo<ScopeT>::getHashValue(scopedOp.first);
    unsigned opHash = OperationEquivalence::computeHash(
        scopedOp.second,
        /*hashOperands=*/OperationEquivalence::directHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
    return llvm::detail::combineHashValue(scopeHash, opHash);
  }

  static bool isEqual(const ScopedOp &lhs, const ScopedOp &rhs) {
    if (lhs == rhs)
      return true;
    return lhs.first == rhs.first &&
           OperationEquivalence::isEquivalentTo(
               lhs.second, rhs.second, OperationEquivalence::IgnoreLocations);
  }
};
} // namespace detail

/// A block-scoped cache of constant-like operations that implements the
/// following policy:
///   * Cacheability: only ops with the `ConstantLike` trait are cached.
///   * Scope: block-scoped. Equivalent cacheable ops created in the same block
///     via `createOrFold` are deduplicated. No caching happens across blocks.
///   * Insertion: newly cached ops are inserted at the beginning of the block
///     following creation order (best-effort).
class BlockScopedConstantLikeCache : public OperationCache {
public:
  bool isCacheable(Operation *op) const override;

  CacheLookupResult
  lookupOrInsertIntoCache(Operation *op, Block *scopeBlock,
                          Block::iterator insertionPoint) override;

  Block::iterator getInsertionPoint(Block *block) const;

  void notifyInserted(Operation *op, Block *block) override;

  void invalidate(Operation *op, Block *scopeBlock = nullptr) override;

  void invalidate(Block *block) override {
    for (Operation &op : *block)
      invalidate(&op, block);
    cacheInsertionPoints.erase(block);
  }

  void clear() override;

private:
  /// Key for block-scoped constant op cache.
  using ScopedCacheOpInfo = detail::ScopedOpCacheInfo<Block *>;
  using ScopedCacheOp = ScopedCacheOpInfo::ScopedOp;

  // TODO: Consider a different data structure for a more efficient full block
  // invalidation?
  using ScopedCacheMapTy =
      llvm::DenseMap<ScopedCacheOp, Operation *, ScopedCacheOpInfo>;

  /// Block-scoped cache for constant-like ops.
  ScopedCacheMapTy constantOpCache;

  /// Cache insertion points per block. A null insertion point triggers
  /// insertion point computation by walking past any pre-existing constant-like
  /// ops at the beginning of the block.
  llvm::DenseMap<Block *, Operation *> cacheInsertionPoints;
};

/// A cache of constant-like operations scoped to the closest enclosing region
/// that is isolated from above. That is usually a function body, but it is also
/// any such region nested inside one, for example a `gpu.module` body. It
/// implements the following policy:
///   * Cacheability: only ops with the `ConstantLike` trait are cached.
///   * Scope: the closest enclosing region that is isolated from above, or a
///     closer one that a dialect pins constants to through
///     `DialectFoldInterface::shouldMaterializeInto`. Equivalent cacheable ops
///     created anywhere in that region are deduplicated, across its blocks and
///     its nested regions.
///   * Insertion: newly cached ops are inserted at the beginning of that
///     region's entry block, following creation order (best-effort).
///
/// Hoisting to the entry block is safe because constant-like ops take no
/// operands, so there are no operand dominance constraints to preserve, and the
/// entry block dominates the rest of the region.
class IsolatedRegionScopedConstantLikeCache : public OperationCache {
public:
  explicit IsolatedRegionScopedConstantLikeCache(MLIRContext *ctx)
      : interfaces(ctx) {}

  bool isCacheable(Operation *op) const override;

  CacheLookupResult
  lookupOrInsertIntoCache(Operation *op, Block *scopeBlock,
                          Block::iterator insertionPoint) override;

  void notifyInserted(Operation *op, Block *block) override;

  void invalidate(Operation *op, Block *scopeBlock = nullptr) override;

  void invalidate(Block *block) override;

  void clear() override;

  /// Returns the region `block` is scoped to, that is the closest enclosing
  /// region that is isolated from above or that a dialect pins constants to.
  /// Returns null when there is none or when the walk reaches an op that is not
  /// linked into the IR yet, since the scope of such an op still depends on
  /// where it ends up.
  Region *getScopeRegion(Block *block) const;

private:
  Block::iterator getInsertionPoint(Block *entryBlock) const;

  using ScopedCacheOpInfo = detail::ScopedOpCacheInfo<Region *>;
  using ScopedCacheOp = ScopedCacheOpInfo::ScopedOp;
  using ScopedCacheMapTy =
      llvm::DenseMap<ScopedCacheOp, Operation *, ScopedCacheOpInfo>;

  /// Region-scoped cache for constant-like ops.
  ScopedCacheMapTy constantOpCache;

  /// Cache insertion points per scope entry block. A missing entry triggers
  /// insertion point computation by walking past any pre-existing constant-like
  /// ops at the beginning of the entry block.
  llvm::DenseMap<Block *, Operation *> cacheInsertionPoints;

  /// Dialect fold interfaces, queried to let a dialect pin constants to a
  /// region that is not isolated from above.
  DialectInterfaceCollection<DialectFoldInterface> interfaces;
};

} // namespace mlir

#endif // MLIR_IR_OPERATIONCACHE_H
