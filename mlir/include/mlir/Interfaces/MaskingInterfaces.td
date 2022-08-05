//===- MaskingInterfaces.td - Masking Interfaces Decls === -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for vector masking related interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_MASKINGINTERFACES
#define MLIR_INTERFACES_MASKINGINTERFACES

include "mlir/IR/OpBase.td"

// The 'VectorMaskingContractionOpInterface' provides access to the
// 'ContractionOpInterface'.
def MaskableOpInterface : OpInterface<"MaskableOpInterface"> {
  let description = [{
   TODO
  }];
  let cppNamespace = "::mlir::vector";
  let methods = [
    InterfaceMethod<
    /*desc=*/"Return true if the operation is masked, either by a "
             "'vector.maskOp' or by a mask in the op specification",
    /*retTy=*/"bool",
    /*methodName=*/"isMasked",
    /*args=*/(ins),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
      return ::mlir::vector::detail::isMaskedDefaultImplementation(
        $_op.getOperation());
    }]>,
    InterfaceMethod<
    /*desc=*/"TODO",
    /*retTy=*/"AffineMap",
    /*methodName=*/"getPermutationMapForMask",
    /*args=*/(ins),
    /*methodBody=*/"",
    /*defaultImplementation=*/[{
      return AffineMap();
    }]>,
  ];
}

#endif // MLIR_INTERFACES_MASKINGINTERFACES
