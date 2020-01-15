// RUN: mlir-opt -convert-std-to-llvm -split-input-file -convert-std-to-llvm-use-bare-ptr-memref-call-conv=1 %s | FileCheck %s --check-prefix=BAREPTR

// BAREPTR-LABEL: func @check_noalias
// BAREPTR-SAME: [[ARG:%.*]]: !llvm<"float*"> {llvm.noalias = true}
func @check_noalias(%static : memref<2xf32> {llvm.noalias = true}) {
    return
}

// WIP: Move tests with static shapes from convert-memref-ops.mlir here.
