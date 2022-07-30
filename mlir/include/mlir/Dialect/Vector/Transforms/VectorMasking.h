//===- VectorTransforms.h - Vector transformations as patterns --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORMASKING_H
#define MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORMASKING_H

namespace mlir {
class RewritePatternSet;

namespace vector {

// TODO.
void populateVectorMaskLoweringPatternsForSideEffectingOps(
    RewritePatternSet &patterns);

void populateVectorMaskOpRemovalPatterns(RewritePatternSet &patterns);

}
}

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_VECTORMASKING_H
