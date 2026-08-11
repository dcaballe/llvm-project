//===- OperationCacheTest.cpp - OperationCache unit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/OperationCache.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "gtest/gtest.h"

#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestOps.h"

using namespace mlir;

namespace {

/// Creates a `test.constant` (ConstantLike) op at the builder's insertion point
/// holding the given i64 `value`.
static Operation *makeConstant(OpBuilder &b, Location loc, int64_t value) {
  return test::TestOpConstant::create(b, loc, b.getI64Type(),
                                      b.getI64IntegerAttr(value))
      .getOperation();
}

/// Builds `module { test.isolated_one_region_op { ^entry, ^second } }` and
/// hands back the blocks of interest. The cache declines to cache blocks with
/// no enclosing region isolated from above, so tests need one to exercise it.
struct NestedIsolatedIR {
  NestedIsolatedIR(MLIRContext &ctx, OpBuilder &b, Location loc)
      : module(ModuleOp::create(b, loc)) {
    moduleBlock = module->getBody();
    b.setInsertionPointToEnd(moduleBlock);
    isolated =
        test::IsolatedOneRegionOp::create(b, loc, /*results=*/TypeRange{},
                                          /*operands=*/ValueRange{});
    entry = b.createBlock(&isolated.getMyRegion());
    second = b.createBlock(&isolated.getMyRegion());
  }

  OwningOpRef<ModuleOp> module;
  test::IsolatedOneRegionOp isolated;
  Block *moduleBlock = nullptr;
  Block *entry = nullptr;
  Block *second = nullptr;
};

TEST(OperationCacheTest, IsCacheable) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  // Cacheability only depends on the op's traits, so no enclosing scope is
  // needed here.
  Block block;
  b.setInsertionPointToEnd(&block);
  IsolatedRegionScopedConstantLikeCache cache(&ctx);

  Operation *cst = makeConstant(b, loc, 0);
  EXPECT_TRUE(cache.isCacheable(cst));

  // A non-constant-like op is not cacheable.
  Operation *cast = UnrealizedConversionCastOp::create(
                        b, loc, TypeRange{b.getI64Type()}, ValueRange{})
                        .getOperation();
  EXPECT_FALSE(cache.isCacheable(cast));
  EXPECT_FALSE(cache.isCacheable(nullptr));
}

TEST(OperationCacheTest, InvalidateOp) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  NestedIsolatedIR ir(ctx, b, loc);
  IsolatedRegionScopedConstantLikeCache cache(&ctx);

  b.setInsertionPointToEnd(ir.entry);
  Operation *c0 = makeConstant(b, loc, 0);
  EXPECT_TRUE(cache.lookupOrInsertIntoCache(c0, ir.entry, ir.entry->end())
                  .insertedInCache());

  cache.invalidate(c0, ir.entry);

  // After invalidation an equivalent lookup misses again.
  Operation *c0b = makeConstant(b, loc, 0);
  EXPECT_TRUE(cache.lookupOrInsertIntoCache(c0b, ir.entry, ir.entry->end())
                  .insertedInCache());
}

TEST(OperationCacheTest, InvalidateBlockAndClear) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  NestedIsolatedIR ir(ctx, b, loc);
  IsolatedRegionScopedConstantLikeCache cache(&ctx);

  b.setInsertionPointToEnd(ir.entry);
  Operation *c0 = makeConstant(b, loc, 0);
  Operation *c1 = makeConstant(b, loc, 1);
  cache.lookupOrInsertIntoCache(c0, ir.entry, ir.entry->end());
  cache.lookupOrInsertIntoCache(c1, ir.entry, ir.entry->end());

  cache.invalidate(ir.entry);
  EXPECT_TRUE(cache.lookupOrInsertIntoCache(c0, ir.entry, ir.entry->end())
                  .insertedInCache());
  EXPECT_TRUE(cache.lookupOrInsertIntoCache(c1, ir.entry, ir.entry->end())
                  .insertedInCache());

  cache.clear();
  Operation *c0b = makeConstant(b, loc, 0);
  EXPECT_TRUE(cache.lookupOrInsertIntoCache(c0b, ir.entry, ir.entry->end())
                  .insertedInCache());
}

TEST(OperationCacheTest, EraseInvalidatesViaRewriter) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  IRRewriter rewriter(&ctx);
  Location loc = rewriter.getUnknownLoc();
  NestedIsolatedIR ir(ctx, rewriter, loc);
  rewriter.setInsertionPointToEnd(ir.entry);

  // The cache is opt-in, so install one to exercise the rewriter's
  // invalidation hooks.
  IsolatedRegionScopedConstantLikeCache cacheStorage(&ctx);
  rewriter.setOperationCache(&cacheStorage);
  OperationCache *cache = rewriter.getOperationCache();
  ASSERT_NE(cache, nullptr);

  Operation *c0 = makeConstant(rewriter, loc, 0);
  EXPECT_TRUE(cache->lookupOrInsertIntoCache(c0, ir.entry, ir.entry->end())
                  .insertedInCache());

  // Erasing through the rewriter must drop the cached entry; otherwise the next
  // lookup would return a dangling op.
  rewriter.eraseOp(c0);

  Operation *c0b = makeConstant(rewriter, loc, 0);
  EXPECT_TRUE(cache->lookupOrInsertIntoCache(c0b, ir.entry, ir.entry->end())
                  .insertedInCache());
}

TEST(OperationCacheTest, InPlaceModifyInvalidatesViaRewriter) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  IRRewriter rewriter(&ctx);
  Location loc = rewriter.getUnknownLoc();
  NestedIsolatedIR ir(ctx, rewriter, loc);
  rewriter.setInsertionPointToEnd(ir.entry);

  // The cache is opt-in, so install one to exercise the rewriter's
  // invalidation hooks.
  IsolatedRegionScopedConstantLikeCache cacheStorage(&ctx);
  rewriter.setOperationCache(&cacheStorage);
  OperationCache *cache = rewriter.getOperationCache();
  ASSERT_NE(cache, nullptr);

  Operation *c0 = makeConstant(rewriter, loc, 0);
  EXPECT_TRUE(cache->lookupOrInsertIntoCache(c0, ir.entry, ir.entry->end())
                  .insertedInCache());

  // `startOpModification` invalidates the entry under the *old* key, while it
  // still matches. Without it, the old-value entry would be orphaned and become
  // dangling after the op is later erased.
  rewriter.modifyOpInPlace(
      c0, [&]() { c0->setAttr("value", rewriter.getI64IntegerAttr(1)); });
  rewriter.eraseOp(c0);

  Operation *cOldValue = makeConstant(rewriter, loc, 0);
  EXPECT_TRUE(
      cache->lookupOrInsertIntoCache(cOldValue, ir.entry, ir.entry->end())
          .insertedInCache());
}

//===----------------------------------------------------------------------===//
// Fold-materialized constant caching (createOrFold / tryFold integration).
//===----------------------------------------------------------------------===//

/// Creates an `arith.constant` of type i32 holding `value` at the builder's
/// insertion point.
static Value makeArithI32(OpBuilder &b, Location loc, int64_t value) {
  return arith::ConstantIntOp::create(b, loc, b.getI32Type(), value);
}

/// Constants produced by folding (not created directly) are routed through the
/// cache, so equal folded constants collapse to a single op.
TEST(OperationCacheTest, FoldMaterializedConstantsAreDeduplicated) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, test::TestDialect>();
  IsolatedRegionScopedConstantLikeCache cache(&ctx);
  OpBuilder b(&ctx, /*listener=*/nullptr, &cache);
  Location loc = b.getUnknownLoc();
  NestedIsolatedIR ir(ctx, b, loc);
  b.setInsertionPointToEnd(ir.entry);

  Value c1 = makeArithI32(b, loc, 1);

  // Each createOrFold folds addi(1, 1) into a materialized constant 2.
  Value r0 = b.createOrFold<arith::AddIOp>(loc, c1, c1);
  Value r1 = b.createOrFold<arith::AddIOp>(loc, c1, c1);
  EXPECT_EQ(r0, r1);

  int numTwos = 0;
  for (Operation &op : *ir.entry) {
    auto cst = dyn_cast<arith::ConstantOp>(&op);
    if (!cst)
      continue;
    auto attr = dyn_cast<IntegerAttr>(cst.getValue());
    if (attr && attr.getInt() == 2)
      ++numTwos;
  }
  EXPECT_EQ(numTwos, 1);
}

/// A folded constant reuses a constant that was previously created directly
/// through createOrFold, unifying the two creation paths.
TEST(OperationCacheTest, FoldedConstantReusesDirectlyCachedConstant) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, test::TestDialect>();
  IsolatedRegionScopedConstantLikeCache cache(&ctx);
  OpBuilder b(&ctx, /*listener=*/nullptr, &cache);
  Location loc = b.getUnknownLoc();
  NestedIsolatedIR ir(ctx, b, loc);
  b.setInsertionPointToEnd(ir.entry);

  Value direct = b.createOrFold<arith::ConstantIntOp>(loc, b.getI32Type(), 2);

  Value c1 = makeArithI32(b, loc, 1);
  Value folded = b.createOrFold<arith::AddIOp>(loc, c1, c1);
  EXPECT_EQ(folded, direct);
}

/// tryFold reports the constant backing each result, whether it was newly
/// materialized or reused from the cache.
TEST(OperationCacheTest, TryFoldReportsBackingConstantForReuseAndInsert) {
  MLIRContext ctx;
  ctx.loadDialect<arith::ArithDialect, test::TestDialect>();
  IsolatedRegionScopedConstantLikeCache cache(&ctx);
  OpBuilder b(&ctx, /*listener=*/nullptr, &cache);
  Location loc = b.getUnknownLoc();
  NestedIsolatedIR ir(ctx, b, loc);
  b.setInsertionPointToEnd(ir.entry);
  Value c1 = makeArithI32(b, loc, 1);

  // First fold materializes and reports a new constant.
  Operation *add0 = arith::AddIOp::create(b, loc, c1, c1).getOperation();
  SmallVector<Value> res0;
  SmallVector<Operation *> backing0;
  ASSERT_TRUE(succeeded(b.tryFold(add0, res0, &backing0)));
  ASSERT_EQ(backing0.size(), 1u);
  add0->erase();

  // Second fold reuses the cached constant and still reports it as backing.
  Operation *add1 = arith::AddIOp::create(b, loc, c1, c1).getOperation();
  SmallVector<Value> res1;
  SmallVector<Operation *> backing1;
  ASSERT_TRUE(succeeded(b.tryFold(add1, res1, &backing1)));
  ASSERT_EQ(backing1.size(), 1u);
  add1->erase();

  EXPECT_EQ(backing0[0], backing1[0]);
  EXPECT_EQ(res0[0], res1[0]);
}

//===----------------------------------------------------------------------===//
// Region scoping.
//===----------------------------------------------------------------------===//

/// The scope stops at the closest region isolated from above, so a block inside
/// a nested isolated op is scoped to that op's region and not to the enclosing
/// module.
TEST(OperationCacheTest, ScopeStopsAtClosestIsolatedRegion) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  NestedIsolatedIR ir(ctx, b, loc);

  IsolatedRegionScopedConstantLikeCache cache(&ctx);
  EXPECT_EQ(cache.getScopeRegion(ir.entry), &ir.isolated.getMyRegion());
  EXPECT_EQ(cache.getScopeRegion(ir.second), &ir.isolated.getMyRegion());
  EXPECT_EQ(cache.getScopeRegion(ir.moduleBlock), &ir.module->getBodyRegion());
}

/// A dialect can pin constants to a region that is not isolated from above
/// through `DialectFoldInterface::shouldMaterializeInto`. The test dialect pins
/// to `test.one_region_op`, so the scope stops there instead of hoisting out to
/// the enclosing isolated region.
TEST(OperationCacheTest, ScopeStopsAtPinnedRegion) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  NestedIsolatedIR ir(ctx, b, loc);

  b.setInsertionPointToEnd(ir.entry);
  auto pinned = test::OneRegionOp::create(b, loc, /*resultTypes=*/TypeRange{},
                                          /*operands=*/ValueRange{});
  Block *pinnedBody = b.createBlock(&pinned->getRegion(0));

  IsolatedRegionScopedConstantLikeCache cache(&ctx);
  EXPECT_EQ(cache.getScopeRegion(pinnedBody), &pinned->getRegion(0));

  // A constant created in the pinned region stays there rather than being
  // hoisted to the entry block of the enclosing isolated region.
  b.setInsertionPointToEnd(pinnedBody);
  Operation *cst = makeConstant(b, loc, 0);
  auto result = cache.lookupOrInsertIntoCache(cst, pinnedBody, pinnedBody->end());
  ASSERT_TRUE(result.insertedInCache());
  EXPECT_EQ(result.insertionBlock, pinnedBody);
}

/// Equivalent ops created in different blocks of one scope are deduplicated, and
/// a newly cached op is reported for the scope's entry block.
TEST(OperationCacheTest, DedupAcrossBlocksWithinScope) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  NestedIsolatedIR ir(ctx, b, loc);
  IsolatedRegionScopedConstantLikeCache cache(&ctx);

  b.setInsertionPointToEnd(ir.second);
  Operation *c0 = makeConstant(b, loc, 0);
  auto r0 = cache.lookupOrInsertIntoCache(c0, ir.second, ir.second->end());
  ASSERT_TRUE(r0.insertedInCache());
  EXPECT_EQ(r0.insertionBlock, ir.entry);

  // An equivalent op requested from the entry block reuses it, since the entry
  // block is where the cached op lives.
  b.setInsertionPointToEnd(ir.entry);
  Operation *c0Entry = makeConstant(b, loc, 0);
  auto rEntry =
      cache.lookupOrInsertIntoCache(c0Entry, ir.entry, ir.entry->end());
  EXPECT_TRUE(rEntry.foundInCache());
  EXPECT_EQ(rEntry.op, c0);
}

/// Scopes do not leak across an isolated boundary, so the same constant is
/// cached once per scope.
TEST(OperationCacheTest, NoDedupAcrossIsolatedBoundary) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  NestedIsolatedIR ir(ctx, b, loc);
  IsolatedRegionScopedConstantLikeCache cache(&ctx);

  b.setInsertionPointToEnd(ir.second);
  Operation *inner = makeConstant(b, loc, 0);
  ASSERT_TRUE(cache.lookupOrInsertIntoCache(inner, ir.second, ir.second->end())
                  .insertedInCache());

  b.setInsertionPointToEnd(ir.moduleBlock);
  Operation *outer = makeConstant(b, loc, 0);
  auto rOuter =
      cache.lookupOrInsertIntoCache(outer, ir.moduleBlock, ir.moduleBlock->end());
  EXPECT_TRUE(rOuter.insertedInCache());
  EXPECT_EQ(rOuter.insertionBlock, ir.moduleBlock);
}

/// A block whose enclosing op is still detached has no known scope yet, so the
/// cache declines rather than keying on a scope that changes once it is linked.
TEST(OperationCacheTest, DeclinesWhileEnclosingOpIsDetached) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();

  // A region op that is not linked into any block yet. It must be an op the
  // test dialect does not pin constants to, otherwise the pin decides the scope
  // before the detached check is reached.
  auto detached = test::TwoRegionOp::create(b, loc, /*resultTypes=*/TypeRange{},
                                            /*operands=*/ValueRange{});
  Block *body = b.createBlock(&detached->getRegion(0));

  IsolatedRegionScopedConstantLikeCache cache(&ctx);
  EXPECT_EQ(cache.getScopeRegion(body), nullptr);

  b.setInsertionPointToEnd(body);
  Operation *cst = makeConstant(b, loc, 0);
  auto r = cache.lookupOrInsertIntoCache(cst, body, body->end());
  EXPECT_FALSE(r.insertedInCache());
  EXPECT_FALSE(r.foundInCache());

  detached->erase();
}

//===----------------------------------------------------------------------===//
// Block scoping.
//===----------------------------------------------------------------------===//

TEST(OperationCacheTest, DedupWithinBlockAndScoping) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  Block block, otherBlock;
  BlockScopedConstantLikeCache cache;

  b.setInsertionPointToEnd(&block);
  Operation *c0a = makeConstant(b, loc, 0);
  Operation *c0b = makeConstant(b, loc, 0); // Equivalent to c0a.
  Operation *c1 = makeConstant(b, loc, 1);  // Different value.

  auto r0a = cache.lookupOrInsertIntoCache(c0a, &block, block.end());
  EXPECT_TRUE(r0a.insertedInCache());
  EXPECT_EQ(r0a.op, c0a);

  // Equivalent op in the same block is deduplicated to the first one.
  auto r0b = cache.lookupOrInsertIntoCache(c0b, &block, block.end());
  EXPECT_TRUE(r0b.foundInCache());
  EXPECT_EQ(r0b.op, c0a);

  // Different value is not deduplicated.
  EXPECT_TRUE(cache.lookupOrInsertIntoCache(c1, &block, block.end()).insertedInCache());

  // The same value in a different block is not deduplicated (block-scoped).
  b.setInsertionPointToEnd(&otherBlock);
  Operation *c0other = makeConstant(b, loc, 0);
  EXPECT_TRUE(cache.lookupOrInsertIntoCache(c0other, &otherBlock, otherBlock.end()).insertedInCache());
}

TEST(OperationCacheTest, InsertionPointHoistsConstants) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  Block block;
  BlockScopedConstantLikeCache cache;

  // Empty block: the insertion point is the block begin.
  EXPECT_EQ(cache.getInsertionPoint(&block), block.begin());

  // After a cached op is inserted and announced, subsequent cached ops land
  // immediately after it (chronological order at the top of the block).
  b.setInsertionPointToStart(&block);
  Operation *c0 = makeConstant(b, loc, 0);
  cache.lookupOrInsertIntoCache(c0, &block, block.end());
  cache.notifyInserted(c0, &block);
  EXPECT_EQ(cache.getInsertionPoint(&block), std::next(Block::iterator(c0)));
}

TEST(OperationCacheTest, InsertionPointDominatesRequestedPoint) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  Block block;
  b.setInsertionPointToEnd(&block);
  BlockScopedConstantLikeCache cache;

  // Cache an op at the block top so the policy's hoist point is *after* it,
  // then cache a new op with the insertion point set *at* that first op (i.e.
  // above the hoist point).
  Operation *c0 = makeConstant(b, loc, 0);
  cache.lookupOrInsertIntoCache(c0, &block, block.end());
  cache.notifyInserted(c0, &block);
  Operation *c1 = makeConstant(b, loc, 1);

  auto r = cache.lookupOrInsertIntoCache(c1, &block, Block::iterator(c0));
  ASSERT_TRUE(r.insertedInCache());

  // A block-scoped cache always reports the block it was scoped to.
  EXPECT_EQ(r.insertionBlock, &block);

  // The policy alone would return std::next(c0); it must be clamped to the
  // requested insertion point so the op dominates it.
  EXPECT_EQ(r.insertionPoint, Block::iterator(c0));
  EXPECT_NE(r.insertionPoint, std::next(Block::iterator(c0)));
}

TEST(OperationCacheTest, ClampedInsertionKeepsCacheInsertionPoint) {
  MLIRContext ctx;
  ctx.loadDialect<test::TestDialect>();
  OpBuilder b(&ctx);
  Location loc = b.getUnknownLoc();
  Block block;
  BlockScopedConstantLikeCache cache;

  // Two normally-hoisted ops: the cache's insertion point advances past the
  // last one (`c1`), so the next op lands right after it.
  b.setInsertionPointToStart(&block);
  Operation *c0 = makeConstant(b, loc, 0);
  cache.lookupOrInsertIntoCache(c0, &block, block.end());
  cache.notifyInserted(c0, &block);
  b.setInsertionPointToEnd(&block);
  Operation *c1 = makeConstant(b, loc, 1);
  cache.lookupOrInsertIntoCache(c1, &block, block.end());
  cache.notifyInserted(c1, &block);
  EXPECT_EQ(cache.getInsertionPoint(&block), std::next(Block::iterator(c1)));

  // A clamped op lands before the cache's insertion point. That point stays at
  // `c1` so later ops keep hoisting after `c1`, not after the clamped op.
  b.setInsertionPointToStart(&block);
  Operation *clamped = makeConstant(b, loc, 2);
  cache.lookupOrInsertIntoCache(clamped, &block, block.end());
  cache.notifyInserted(clamped, &block);
  EXPECT_EQ(cache.getInsertionPoint(&block), std::next(Block::iterator(c1)));
  EXPECT_NE(cache.getInsertionPoint(&block), std::next(Block::iterator(clamped)));
}

} // namespace
