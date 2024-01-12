//===- BuiltinTypeInterfaces.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
/// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinTypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// ShapedType
//===----------------------------------------------------------------------===//

constexpr int64_t ShapedType::kDynamic;

int64_t ShapedType::getNumElements(ArrayRef<int64_t> shape) {
  int64_t num = 1;
  for (int64_t dim : shape) {
    num *= dim;
    assert(num >= 0 && "integer overflow in element count computation");
  }
  return num;
}

//===----------------------------------------------------------------------===//
// VectorBaseType
//===----------------------------------------------------------------------===//

VectorBaseType VectorBaseType::get(ArrayRef<int64_t> shape, Type elementType,
                                   ArrayRef<int64_t> scalableBases) {
  if (any_of(scalableBases,
             [](int64_t scalableBase) { return scalableBase != 0; })) {
    return ScalableVectorType::get(shape, elementType, scalableBases);
  }

  return FixedVectorType::get(shape, elementType);
}

VectorBaseType
VectorBaseType::getChecked(function_ref<::mlir::InFlightDiagnostic()> emitError,
                           ArrayRef<int64_t> shape, Type elementType,
                           ArrayRef<int64_t> scalableBases) {
  if (any_of(scalableBases,
             [](int64_t scalableBase) { return scalableBase != 0; })) {
    return ScalableVectorType::getChecked(emitError, shape, elementType,
                                          scalableBases);
  }

  return FixedVectorType::getChecked(emitError, shape, elementType);
}

VectorBaseType VectorBaseType::get(ArrayRef<int64_t> shape, Type elementType,
                                   ArrayRef<bool> scalableDims) {
  if (any_of(scalableDims,
             [](bool scalableDim) { return scalableDim == true; })) {
    return ScalableVectorType::get(shape, elementType, scalableDims);
  }

  return FixedVectorType::get(shape, elementType);
}

VectorBaseType
VectorBaseType::getChecked(function_ref<::mlir::InFlightDiagnostic()> emitError,
                           ArrayRef<int64_t> shape, Type elementType,
                           ArrayRef<bool> scalableDims) {
  if (any_of(scalableDims,
             [](bool scalableDim) { return scalableDim == true; })) {
    return ScalableVectorType::getChecked(emitError, shape, elementType,
                                          scalableDims);
  }

  return FixedVectorType::getChecked(emitError, shape, elementType);
}

VectorBaseType VectorBaseType::scaleElementBitwidth(unsigned scale) {
  if (!scale)
    return VectorBaseType();
  if (auto et = llvm::dyn_cast<IntegerType>(getElementType()))
    if (auto scaledEt = et.scaleElementBitwidth(scale))
      return VectorBaseType::get(getShape(), scaledEt, getScalableBases());
  if (auto et = llvm::dyn_cast<FloatType>(getElementType()))
    if (auto scaledEt = et.scaleElementBitwidth(scale))
      return VectorBaseType::get(getShape(), scaledEt, getScalableBases());
  return VectorBaseType();
}

bool VectorBaseType::isValidElementType(Type t) {
  return ::llvm::isa<::mlir::IntegerType, ::mlir::IndexType, ::mlir::FloatType>(
      t);
}
