//// RUN: mlir-opt %s -test-linalg-transform-patterns=test-linalg-to-vector-patterns -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-linalg-transform-patterns=test-linalg-to-vector-patterns -lower-vector-mask -split-input-file | FileCheck %s

func.func @masked_add(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
                      %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
       ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
       outs(%arg2: tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %3 = arith.addf %a, %b : f32
    linalg.yield %3 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

//func.func @masked_add(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
//                      %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
//  %0 = linalg.generic {
//    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
//                     affine_map<(d0, d1) -> (d0, d1)>,
//                     affine_map<(d0, d1) -> (d0, d1)>],
//         iterator_types = ["parallel", "parallel"]}
//         ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
//         outs(%arg2: tensor<?x?xf32>) {
//    ^bb0(%a: f32, %b: f32, %c: f32):
//      %3 = arith.addf %a, %b : f32
//      linalg.yield %3 : f32
//    } -> tensor<?x?xf32>
//  return %0 : tensor<?x?xf32>
//}


//func.func @masked_matmul(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
//                         %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
//  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<?x?xf32>, tensor<?x?xf32>)
//                     outs(%arg2: tensor<?x?xf32>)
//    -> tensor<?x?xf32>
//  return %0 : tensor<?x?xf32>
//}

