//===- MaskingInterfaces.h - Masking operations interfaces ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the operation interfaces for Masking operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_MASKINGINTERFACES_H_
#define MLIR_INTERFACES_MASKINGINTERFACES_H_

//#include "mlir/IR/AffineMap.h"
//#include "mlir/IR/BlockAndValueMapping.h"
//#include "mlir/IR/BuiltinTypes.h"
//#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OpDefinition.h"
//#include "mlir/Interfaces/InferTypeOpInterface.h"
//#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace vector {
namespace detail {
bool isMaskedDefaultImplementation(Operation *op);
} // namespace detail
} // namespace vector
} // namespace mlir

/// Include the generated interface declarations.
#include "mlir/Interfaces/MaskingInterfaces.h.inc"

#endif // MLIR_INTERFACES_MASKINGINTERFACES_H_
