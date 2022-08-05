//===- MaskingInterfaces.cpp - Unrollable vector operations ====-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/MaskingInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace vector;

bool vector::detail::isMaskedDefaultImplementation(Operation *op) {
  if (!op)
    return false;
  return isa<vector::MaskOp>(op->getParentOp());
}


//===----------------------------------------------------------------------===//
// Masking Interfaces
//===----------------------------------------------------------------------===//

/// Include the definitions of the masking interfaces.
#include "mlir/Interfaces/MaskingInterfaces.cpp.inc"
