//===- VectorTransforms.cpp - Conversion within the Vector dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilitites for the
// vector.mask operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/Transforms/VectorMasking.h"
//#include "mlir/Dialect/Affine/IR/AffineOps.h"
//#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
//#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
//#include "mlir/Dialect/Linalg/IR/Linalg.h"
//#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
//#include "mlir/Dialect/SCF/IR/SCF.h"
//#include "mlir/Dialect/Utils/IndexingUtils.h"
//#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
//#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
//#include "mlir/IR/BuiltinTypes.h"
//#include "mlir/IR/ImplicitLocOpBuilder.h"
//#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
//#include "mlir/Interfaces/VectorInterfaces.h"
//#include "llvm/ADT/DenseSet.h"
//#include "llvm/ADT/MapVector.h"
//#include "llvm/ADT/STLExtras.h"
//#include "llvm/Support/CommandLine.h"
//#include "llvm/Support/Debug.h"
//#include "llvm/Support/raw_ostream.h"
//#include <type_traits>
#include "PassDetail.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

//#define DEBUG_TYPE "vector-masking"

using namespace mlir;
using namespace mlir::vector;

namespace {

struct MaskTransferReadOpPattern : public OpRewritePattern<TransferReadOp> {
public:
  explicit MaskTransferReadOpPattern(MLIRContext *context)
      : mlir::OpRewritePattern<TransferReadOp>(context) {}

  LogicalResult matchAndRewrite(TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    auto parentMaskOp = dyn_cast<MaskOp>(readOp->getParentOp());
    // Unmasked operation.
    if (!parentMaskOp)
      return failure();
    // Operation was already lowered.
    if (readOp.getMask())
      return failure();

    Value padding = parentMaskOp.hasPassthrough()
                        ? parentMaskOp.getPassthrough()
                        : readOp.getPadding();
    rewriter.replaceOpWithNewOp<TransferReadOp>(
        readOp, readOp.getVectorType(), readOp.getSource(), readOp.getIndices(),
        readOp.getPermutationMap(), readOp.getPadding(), parentMaskOp.getMask(),
        readOp.getInBounds().value_or(ArrayAttr()));
    return success();
  }
};

struct MaskTransferWriteOpPattern : public OpRewritePattern<TransferWriteOp> {
public:
  explicit MaskTransferWriteOpPattern(MLIRContext *context)
      : mlir::OpRewritePattern<TransferWriteOp>(context) {}

  LogicalResult matchAndRewrite(TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    auto parentMaskOp = dyn_cast<MaskOp>(writeOp->getParentOp());
    // Unmasked operation.
    if (!parentMaskOp)
      return failure();
    // Operation was already lowered.
    if (writeOp.getMask())
      return failure();

    rewriter.replaceOpWithNewOp<TransferWriteOp>(
        writeOp, writeOp.getResult().getType(), writeOp.getVector(),
        writeOp.getSource(), writeOp.getIndices(), writeOp.getPermutationMap(),
        parentMaskOp.getMask(), writeOp.getInBounds().value_or(ArrayAttr()));
    return success();
  }
};

struct RemoveMaskOpPattern : public OpRewritePattern<MaskOp> {
public:
  explicit RemoveMaskOpPattern(MLIRContext *context)
      : mlir::OpRewritePattern<MaskOp>(context) {}

  LogicalResult matchAndRewrite(MaskOp maskOp,
                                PatternRewriter &rewriter) const override {

    llvm::outs() << maskOp << "\n";
    // Move all the operations within vector.mask (except vector.yield) before
    // vector.mask.
    auto& dstOps = maskOp->getBlock()->getOperations();
    auto dstPtr = Block::iterator(maskOp);
    auto& srcOps = maskOp.getRegion().front().getOperations();
    auto yieldOpIt = std::prev(srcOps.end());
    dstOps.splice(dstPtr, srcOps, srcOps.begin(), yieldOpIt);

    // Replace 'vector.mask' with 'vector.yield' operands, now outside of
    // 'vector.mask'.
    rewriter.replaceOp(maskOp, yieldOpIt->getOperands());
    return success();
  }
};

struct LowerVectorMaskPass
    : public LowerVectorMaskPassBase<LowerVectorMaskPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    GreedyRewriteConfig rewriterConfig;
    rewriterConfig.maxIterations = 1;
    rewriterConfig.useTopDownTraversal = true;

    // Two-step lowering:
    //   1. Lower the masked operations nested in a vector.mask operation.
    {
      RewritePatternSet patterns(context);
      populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
      (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns),
                                         rewriterConfig);
    }
    //   2. Remove the vector.mask operation.
    {
      RewritePatternSet patterns(context);
      populateVectorMaskOpRemovalPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns),
                                         rewriterConfig);
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
};

} // namespace

void mlir::vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
    RewritePatternSet &patterns) {
  patterns.add<MaskTransferReadOpPattern, MaskTransferWriteOpPattern>(
      patterns.getContext());
}

void mlir::vector::populateVectorMaskOpRemovalPatterns(
    RewritePatternSet &patterns) {
  patterns.add<RemoveMaskOpPattern>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::vector::createLowerVectorMaskPass() {
  return std::make_unique<LowerVectorMaskPass>();
}
