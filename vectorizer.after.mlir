module {
  func.func @vectorize_dynamic_identity(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim : vector<4xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %2 = vector.mask %0 { vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %3 = vector.mask %0 { vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %4 = arith.addf %1, %2 : vector<4xf32>
    %5 = vector.mask %0 { vector.transfer_write %4, %arg2[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>
    return %5 : tensor<?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [4] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @vectorize_dynamic_identity_with_constant(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim : vector<4xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %2 = vector.mask %0 { vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %3 = vector.mask %0 { vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %4 = arith.addf %1, %2 : vector<4xf32>
    %5 = vector.mask %0 { vector.transfer_write %4, %arg2[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>
    return %5 : tensor<?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.structured.match ops{["arith.constant"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [%1] : !transform.any_op, !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @vectorize_dynamic_identity_with_param(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim : vector<4xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %2 = vector.mask %0 { vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %3 = vector.mask %0 { vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %4 = arith.addf %1, %2 : vector<4xf32>
    %5 = vector.mask %0 { vector.transfer_write %4, %arg2[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>
    return %5 : tensor<?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.param.constant 4 : i64 -> !transform.param<i64>
      transform.structured.vectorize %0 vector_sizes [%1] : !transform.any_op, !transform.param<i64>
      transform.yield 
    }
  }
}

// -----
#map = affine_map<(d0) -> (0)>
module {
  func.func @vectorize_dynamic_1d_broadcast(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0], %cst {permutation_map = #map} : tensor<?xf32>, vector<4xf32>
    %1 = vector.create_mask %dim : vector<4xi1>
    %2 = vector.mask %1 { vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %3 = vector.mask %1 { vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %4 = arith.addf %0, %2 : vector<4xf32>
    %5 = vector.mask %1 { vector.transfer_write %4, %arg2[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>
    return %5 : tensor<?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [4] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
#map = affine_map<(d0, d1) -> (d0)>
module {
  func.func @dynamic_generic_with_reduction_and_broadcast(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim, %dim_0 : vector<4x4xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x4xf32> } : vector<4x4xi1> -> vector<4x4xf32>
    %2 = vector.create_mask %dim : vector<4xi1>
    %3 = vector.mask %2 { vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true], permutation_map = #map} : tensor<?x?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %4 = vector.mask %0 { vector.multi_reduction <add>, %1, %3 [1] : vector<4x4xf32> to vector<4xf32> } : vector<4x4xi1> -> vector<4xf32>
    %5 = vector.mask %2 { vector.transfer_write %4, %arg1[%c0, %c0] {in_bounds = [true], permutation_map = #map} : vector<4xf32>, tensor<?x?xf32> } : vector<4xi1> -> tensor<?x?xf32>
    return %5 : tensor<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [4, 4] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
#map = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @vectorize_dynamic_2d_transpose(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %c0 = arith.constant 0 : index
    %dim_0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim_0, %dim : vector<8x4xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = #map} : tensor<?x?xf32>, vector<4x8xf32> } : vector<8x4xi1> -> vector<4x8xf32>
    %2 = vector.create_mask %dim, %dim_0 : vector<4x8xi1>
    %3 = vector.mask %2 { vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
    %4 = vector.mask %2 { vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
    %5 = arith.addf %1, %3 : vector<4x8xf32>
    %6 = vector.mask %2 { vector.transfer_write %5, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xf32>, tensor<?x?xf32> } : vector<4x8xi1> -> tensor<?x?xf32>
    return %6 : tensor<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [4, 8] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
#map = affine_map<(d0, d1) -> (0, d1)>
module {
  func.func @vectorize_dynamic_generic_2d_broadcast(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim_0 : vector<8xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = #map} : tensor<?x?xf32>, vector<4x8xf32> } : vector<8xi1> -> vector<4x8xf32>
    %2 = vector.create_mask %dim, %dim_0 : vector<4x8xi1>
    %3 = vector.mask %2 { vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
    %4 = vector.mask %2 { vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
    %5 = arith.addf %1, %3 : vector<4x8xf32>
    %6 = vector.mask %2 { vector.transfer_write %5, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<4x8xf32>, tensor<?x?xf32> } : vector<4x8xi1> -> tensor<?x?xf32>
    return %6 : tensor<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [4, 8] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @vectorize_dynamic_reduction(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim, %dim_0 : vector<4x8xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
    %2 = vector.create_mask %dim : vector<4xi1>
    %3 = vector.mask %2 { vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %4 = vector.mask %0 { vector.multi_reduction <add>, %1, %3 [1] : vector<4x8xf32> to vector<4xf32> } : vector<4x8xi1> -> vector<4xf32>
    %5 = vector.mask %2 { vector.transfer_write %4, %arg1[%c0] {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>
    return %5 : tensor<?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [4, 8] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
#map = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @vectorize_dynamic_transpose_reduction(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
    %c2 = arith.constant 2 : index
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim, %dim_0, %dim_1 : vector<4x8x16xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : tensor<?x?x?xf32>, vector<4x8x16xf32> } : vector<4x8x16xi1> -> vector<4x8x16xf32>
    %2 = vector.create_mask %dim_1, %dim_0 : vector<16x8xi1>
    %3 = vector.mask %2 { vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = #map} : tensor<?x?xf32>, vector<8x16xf32> } : vector<16x8xi1> -> vector<8x16xf32>
    %4 = vector.mask %0 { vector.multi_reduction <add>, %1, %3 [0] : vector<4x8x16xf32> to vector<8x16xf32> } : vector<4x8x16xi1> -> vector<8x16xf32>
    %5 = vector.mask %2 { vector.transfer_write %4, %arg1[%c0, %c0] {in_bounds = [true, true], permutation_map = #map} : vector<8x16xf32>, tensor<?x?xf32> } : vector<16x8xi1> -> tensor<?x?xf32>
    return %5 : tensor<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [4, 8, 16] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
#map = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @vectorize_dynamic_transpose_reduction_with_params(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
    %c2 = arith.constant 2 : index
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim, %dim_0, %dim_1 : vector<4x8x16xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : tensor<?x?x?xf32>, vector<4x8x16xf32> } : vector<4x8x16xi1> -> vector<4x8x16xf32>
    %2 = vector.create_mask %dim_1, %dim_0 : vector<16x8xi1>
    %3 = vector.mask %2 { vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = #map} : tensor<?x?xf32>, vector<8x16xf32> } : vector<16x8xi1> -> vector<8x16xf32>
    %4 = vector.mask %0 { vector.multi_reduction <add>, %1, %3 [0] : vector<4x8x16xf32> to vector<8x16xf32> } : vector<4x8x16xi1> -> vector<8x16xf32>
    %5 = vector.mask %2 { vector.transfer_write %4, %arg1[%c0, %c0] {in_bounds = [true, true], permutation_map = #map} : vector<8x16xf32>, tensor<?x?xf32> } : vector<16x8xi1> -> tensor<?x?xf32>
    return %5 : tensor<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.param.constant 4 : i64 -> !transform.param<i64>
      %2 = transform.param.constant 16 : i64 -> !transform.param<i64>
      transform.structured.vectorize %0 vector_sizes [%1, 8, %2] : !transform.any_op, !transform.param<i64>, !transform.param<i64>
      transform.yield 
    }
  }
}

// -----
module {
  func.func @vectorize_partial_dynamic_identity(%arg0: tensor<8x?xf32>, %arg1: tensor<8x?xf32>, %arg2: tensor<8x?xf32>) -> tensor<8x?xf32> {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<8x?xf32>
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %c8, %dim : vector<8x32xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
    %2 = vector.mask %0 { vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
    %3 = vector.mask %0 { vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
    %4 = arith.addf %1, %2 : vector<8x32xf32>
    %5 = vector.mask %0 { vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x32xf32>, tensor<8x?xf32> } : vector<8x32xi1> -> tensor<8x?xf32>
    return %5 : tensor<8x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [8, 32] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @do_not_generate_masks(%arg0: tensor<8x32xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<8x32xf32>) -> tensor<8x32xf32> {
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst : tensor<8x32xf32>, vector<8x32xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0], %cst : tensor<8x32xf32>, vector<8x32xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0], %cst : tensor<8x32xf32>, vector<8x32xf32>
    %3 = arith.addf %0, %1 : vector<8x32xf32>
    %4 = vector.transfer_write %3, %arg2[%c0, %c0] : vector<8x32xf32>, tensor<8x32xf32>
    return %4 : tensor<8x32xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [8, 32] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @vectorize_static_shape_with_mask(%arg0: tensor<8x30xf32>, %arg1: tensor<8x30xf32>, %arg2: tensor<8x30xf32>) -> tensor<8x30xf32> {
    %c8 = arith.constant 8 : index
    %c30 = arith.constant 30 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %c8, %c30 : vector<8x32xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x30xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
    %2 = vector.mask %0 { vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x30xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
    %3 = vector.mask %0 { vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<8x30xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
    %4 = arith.addf %1, %2 : vector<8x32xf32>
    %5 = vector.mask %0 { vector.transfer_write %4, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x32xf32>, tensor<8x30xf32> } : vector<8x32xi1> -> tensor<8x30xf32>
    return %5 : tensor<8x30xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [8, 32] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @vectorize_dynamic_fill(%arg0: tensor<?x?xf32>, %arg1: f32) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim, %dim_0 : vector<8x16xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<8x16xf32> } : vector<8x16xi1> -> vector<8x16xf32>
    %2 = vector.broadcast %arg1 : f32 to vector<8x16xf32>
    %3 = vector.mask %0 { vector.transfer_write %2, %arg0[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, tensor<?x?xf32> } : vector<8x16xi1> -> tensor<?x?xf32>
    return %3 : tensor<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [8, 16] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
#map = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @test_masked_vectorize_linalg_transpose(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %c0 = arith.constant 0 : index
    %dim_0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim_0, %dim : vector<4x2xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = #map} : tensor<?x?xf32>, vector<2x4xf32> } : vector<4x2xi1> -> vector<2x4xf32>
    %2 = vector.create_mask %dim, %dim_0 : vector<2x4xi1>
    %3 = vector.mask %2 { vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<2x4xf32> } : vector<2x4xi1> -> vector<2x4xf32>
    %4 = vector.mask %2 { vector.transfer_write %1, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xf32>, tensor<?x?xf32> } : vector<2x4xi1> -> tensor<?x?xf32>
    return %4 : tensor<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.transpose"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [2, 4] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_masked_vectorize_linalg_copy(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim, %dim_0 : vector<2x4xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x4xf32> } : vector<2x4xi1> -> vector<2x4xf32>
    %2 = vector.mask %0 { vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x4xf32> } : vector<2x4xi1> -> vector<2x4xf32>
    vector.mask %0 { vector.transfer_write %1, %arg1[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xf32>, memref<?x?xf32> } : vector<2x4xi1>
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.copy"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [2, 4] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_masked_vectorize_pad(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index) -> tensor<2x4xf32> {
    %cst = arith.constant 4.243000e+01 : f32
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0_0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = vector.create_mask %dim, %dim_1 : vector<2x4xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0_0, %c0_0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<2x4xf32> } : vector<2x4xi1> -> vector<2x4xf32>
    %2 = tensor.empty() : tensor<2x4xf32>
    %3 = vector.transfer_write %1, %2[%c0_0, %c0_0] {in_bounds = [true, true]} : vector<2x4xf32>, tensor<2x4xf32>
    return %3 : tensor<2x4xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pad"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [2, 4] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
#map = affine_map<()[s0, s1] -> (s0 + s1)>
module {
  func.func @test_masked_vectorize_dynamic_pad(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index) -> tensor<?x?xf32> {
    %cst = arith.constant 4.243000e+01 : f32
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0_0 : tensor<?x?xf32>
    %0 = affine.apply #map()[%arg1, %dim]
    %c1 = arith.constant 1 : index
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %1 = affine.apply #map()[%arg2, %dim_1]
    %dim_2 = tensor.dim %arg0, %c0_0 : tensor<?x?xf32>
    %dim_3 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %2 = vector.create_mask %dim_2, %dim_3 : vector<2x4xi1>
    %3 = vector.mask %2 { vector.transfer_read %arg0[%c0_0, %c0_0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<2x4xf32> } : vector<2x4xi1> -> vector<2x4xf32>
    %4 = tensor.empty(%0, %1) : tensor<?x?xf32>
    %dim_4 = tensor.dim %4, %c0_0 : tensor<?x?xf32>
    %dim_5 = tensor.dim %4, %c1 : tensor<?x?xf32>
    %5 = vector.create_mask %dim_4, %dim_5 : vector<2x4xi1>
    %6 = vector.mask %5 { vector.transfer_write %3, %4[%c0_0, %c0_0] {in_bounds = [true, true]} : vector<2x4xf32>, tensor<?x?xf32> } : vector<2x4xi1> -> tensor<?x?xf32>
    return %6 : tensor<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pad"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [2, 4] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_masked_vectorize_non_zero_low_pad_unit_res_dim(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index) -> tensor<1x4xf32> {
    %cst = arith.constant 4.243000e+01 : f32
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0_0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = vector.create_mask %dim, %dim_1 : vector<1x4xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0_0, %c0_0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<1x4xf32> } : vector<1x4xi1> -> vector<1x4xf32>
    %2 = tensor.empty() : tensor<1x4xf32>
    %3 = vector.transfer_write %1, %2[%c0_0, %c0_0] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>
    return %3 : tensor<1x4xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.pad"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [1, 4] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_pack(%arg0: tensor<32x8x16xf32>, %arg1: tensor<4x1x32x16x2xf32>) -> tensor<4x1x32x16x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : tensor<32x8x16xf32>, vector<32x8x16xf32>
    %1 = vector.shape_cast %0 : vector<32x8x16xf32> to vector<32x4x2x1x16xf32>
    %2 = vector.transpose %1, [1, 3, 0, 4, 2] : vector<32x4x2x1x16xf32> to vector<4x1x32x16x2xf32>
    %3 = tensor.empty() : tensor<4x1x32x16x2xf32>
    %4 = vector.transfer_write %2, %3[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true]} : vector<4x1x32x16x2xf32>, tensor<4x1x32x16x2xf32>
    return %4 : tensor<4x1x32x16x2xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [4, 1, 32] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_padded_pack(%arg0: tensor<32x7x15xf32>, %arg1: tensor<32x4x1x16x2xf32>) -> tensor<32x4x1x16x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c7 = arith.constant 7 : index
    %c15 = arith.constant 15 : index
    %0 = vector.create_mask %c32, %c7, %c15 : vector<32x8x16xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : tensor<32x7x15xf32>, vector<32x8x16xf32> } : vector<32x8x16xi1> -> vector<32x8x16xf32>
    %2 = vector.shape_cast %1 : vector<32x8x16xf32> to vector<32x4x2x1x16xf32>
    %3 = vector.transpose %2, [0, 1, 3, 4, 2] : vector<32x4x2x1x16xf32> to vector<32x4x1x16x2xf32>
    %4 = tensor.empty() : tensor<32x4x1x16x2xf32>
    %5 = vector.transfer_write %3, %4[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true]} : vector<32x4x1x16x2xf32>, tensor<32x4x1x16x2xf32>
    return %5 : tensor<32x4x1x16x2xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [32, 4, 1] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_dynamic_pack(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x16x2xf32>) -> tensor<?x?x16x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?x?x16x2xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?x16x2xf32>
    %dim_1 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_2 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = vector.create_mask %dim_1, %dim_2 : vector<8x16xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<8x16xf32> } : vector<8x16xi1> -> vector<8x16xf32>
    %2 = vector.shape_cast %1 : vector<8x16xf32> to vector<4x2x1x16xf32>
    %3 = vector.transpose %2, [0, 2, 3, 1] : vector<4x2x1x16xf32> to vector<4x1x16x2xf32>
    %4 = tensor.empty(%dim, %dim_0) : tensor<?x?x16x2xf32>
    %dim_3 = tensor.dim %4, %c0 : tensor<?x?x16x2xf32>
    %dim_4 = tensor.dim %4, %c1 : tensor<?x?x16x2xf32>
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %5 = vector.create_mask %dim_3, %dim_4, %c16, %c2 : vector<4x1x16x2xi1>
    %6 = vector.mask %5 { vector.transfer_write %3, %4[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<4x1x16x2xf32>, tensor<?x?x16x2xf32> } : vector<4x1x16x2xi1> -> tensor<?x?x16x2xf32>
    return %6 : tensor<?x?x16x2xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [4, 1] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
#map = affine_map<(d0, d1) -> (d0, 0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1, d0)>
module {
  func.func @matmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim, %dim_1 : vector<8x4xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map} : memref<?x?xf32>, vector<8x16x4xf32> } : vector<8x4xi1> -> vector<8x16x4xf32>
    %2 = vector.create_mask %dim_1, %dim_0 : vector<4x16xi1>
    %3 = vector.mask %2 { vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map1} : memref<?x?xf32>, vector<8x16x4xf32> } : vector<4x16xi1> -> vector<8x16x4xf32>
    %4 = vector.create_mask %dim, %dim_0 : vector<8x16xi1>
    %5 = vector.mask %4 { vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?xf32>, vector<8x16xf32> } : vector<8x16xi1> -> vector<8x16xf32>
    %6 = arith.mulf %1, %3 : vector<8x16x4xf32>
    %7 = vector.create_mask %dim, %dim_0, %dim_1 : vector<8x16x4xi1>
    %8 = vector.mask %7 { vector.multi_reduction <add>, %6, %5 [2] : vector<8x16x4xf32> to vector<8x16xf32> } : vector<8x16x4xi1> -> vector<8x16xf32>
    vector.mask %4 { vector.transfer_write %8, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x16xf32>, memref<?x?xf32> } : vector<8x16xi1>
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [8, 16, 4] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
#map = affine_map<(d0, d1, d2, d3) -> (d0, 0, d1, d2, 0, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (0, d0, d1, 0, d2, d3)>
module {
  func.func @mmt4d(%arg0: memref<16x16x8x1xf32>, %arg1: memref<16x16x8x1xf32>, %arg2: memref<16x16x8x8xf32>) {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst {permutation_map = #map} : memref<16x16x8x1xf32>, vector<16x16x16x8x8x1xf32>
    %1 = vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %cst {permutation_map = #map1} : memref<16x16x8x1xf32>, vector<16x16x16x8x8x1xf32>
    %2 = vector.transfer_read %arg2[%c0, %c0, %c0, %c0], %cst : memref<16x16x8x8xf32>, vector<16x16x8x8xf32>
    %3 = arith.mulf %0, %1 : vector<16x16x16x8x8x1xf32>
    %4 = vector.multi_reduction <add>, %3, %2 [2, 5] : vector<16x16x16x8x8x1xf32> to vector<16x16x8x8xf32>
    vector.transfer_write %4, %arg2[%c0, %c0, %c0, %c0] : vector<16x16x8x8xf32>, memref<16x16x8x8xf32>
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.mmt4d"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 : !transform.any_op
      transform.yield 
    }
  }
}

// -----
#map = affine_map<(d0, d1) -> (d0, 0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1, d0)>
module {
  func.func @matmul_scalable(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c0 = arith.constant 0 : index
    %dim = memref.dim %arg0, %c0 : memref<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = memref.dim %arg1, %c1 : memref<?x?xf32>
    %dim_1 = memref.dim %arg0, %c1 : memref<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim, %dim_1 : vector<8x4xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map} : memref<?x?xf32>, vector<8x[16]x4xf32> } : vector<8x4xi1> -> vector<8x[16]x4xf32>
    %2 = vector.create_mask %dim_1, %dim_0 : vector<4x[16]xi1>
    %3 = vector.mask %2 { vector.transfer_read %arg1[%c0, %c0], %cst {in_bounds = [true, true, true], permutation_map = #map1} : memref<?x?xf32>, vector<8x[16]x4xf32> } : vector<4x[16]xi1> -> vector<8x[16]x4xf32>
    %4 = vector.create_mask %dim, %dim_0 : vector<8x[16]xi1>
    %5 = vector.mask %4 { vector.transfer_read %arg2[%c0, %c0], %cst {in_bounds = [true, true]} : memref<?x?xf32>, vector<8x[16]xf32> } : vector<8x[16]xi1> -> vector<8x[16]xf32>
    %6 = arith.mulf %1, %3 : vector<8x[16]x4xf32>
    %7 = vector.create_mask %dim, %dim_0, %dim_1 : vector<8x[16]x4xi1>
    %8 = vector.mask %7 { vector.multi_reduction <add>, %6, %5 [2] : vector<8x[16]x4xf32> to vector<8x[16]xf32> } : vector<8x[16]x4xi1> -> vector<8x[16]xf32>
    vector.mask %4 { vector.transfer_write %8, %arg2[%c0, %c0] {in_bounds = [true, true]} : vector<8x[16]xf32>, memref<?x?xf32> } : vector<8x[16]xi1>
    return
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [8, [16], 4] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_dynamic_shapes_unpack(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x16x2xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %dim_1 = tensor.dim %arg1, %c0 : tensor<?x?x16x2xf32>
    %dim_2 = tensor.dim %arg1, %c1 : tensor<?x?x16x2xf32>
    %c16 = arith.constant 16 : index
    %c2 = arith.constant 2 : index
    %0 = vector.create_mask %dim_1, %dim_2, %c16, %c2 : vector<2x1x16x2xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg1[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<?x?x16x2xf32>, vector<2x1x16x2xf32> } : vector<2x1x16x2xi1> -> vector<2x1x16x2xf32>
    %2 = vector.transpose %1, [0, 3, 1, 2] : vector<2x1x16x2xf32> to vector<2x2x1x16xf32>
    %3 = vector.shape_cast %2 : vector<2x2x1x16xf32> to vector<4x16xf32>
    %4 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %dim_3 = tensor.dim %4, %c0 : tensor<?x?xf32>
    %dim_4 = tensor.dim %4, %c1 : tensor<?x?xf32>
    %5 = vector.create_mask %dim_3, %dim_4 : vector<4x16xi1>
    %6 = vector.mask %5 { vector.transfer_write %3, %4[%c0, %c0] {in_bounds = [true, true]} : vector<4x16xf32>, tensor<?x?xf32> } : vector<4x16xi1> -> tensor<?x?xf32>
    return %6 : tensor<?x?xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [4, 16] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_unpack(%arg0: tensor<8x8x32x16xf32>, %arg1: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %0 = vector.create_mask %c8, %c8, %c32, %c16 : vector<16x8x32x16xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<8x8x32x16xf32>, vector<16x8x32x16xf32> } : vector<16x8x32x16xi1> -> vector<16x8x32x16xf32>
    %2 = vector.transpose %1, [0, 2, 1, 3] : vector<16x8x32x16xf32> to vector<16x32x8x16xf32>
    %3 = vector.shape_cast %2 : vector<16x32x8x16xf32> to vector<512x128xf32>
    %4 = tensor.empty() : tensor<256x128xf32>
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %5 = vector.create_mask %c256, %c128 : vector<512x128xi1>
    %6 = vector.mask %5 { vector.transfer_write %3, %4[%c0, %c0] {in_bounds = [true, true]} : vector<512x128xf32>, tensor<256x128xf32> } : vector<512x128xi1> -> tensor<256x128xf32>
    return %6 : tensor<256x128xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [512, 128] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_unpack_no_masks(%arg0: tensor<8x8x32x16xf32>, %arg1: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<8x8x32x16xf32>, vector<8x8x32x16xf32>
    %1 = vector.transpose %0, [0, 2, 1, 3] : vector<8x8x32x16xf32> to vector<8x32x8x16xf32>
    %2 = vector.shape_cast %1 : vector<8x32x8x16xf32> to vector<256x128xf32>
    %3 = tensor.empty() : tensor<256x128xf32>
    %4 = vector.transfer_write %2, %3[%c0, %c0] {in_bounds = [true, true]} : vector<256x128xf32>, tensor<256x128xf32>
    return %4 : tensor<256x128xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [256, 128] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_unpack_with_outer_perm(%arg0: tensor<8x8x32x16xf32>, %arg1: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<8x8x32x16xf32>, vector<8x8x32x16xf32>
    %1 = vector.transpose %0, [1, 2, 0, 3] : vector<8x8x32x16xf32> to vector<8x32x8x16xf32>
    %2 = vector.shape_cast %1 : vector<8x32x8x16xf32> to vector<256x128xf32>
    %3 = tensor.empty() : tensor<256x128xf32>
    %4 = vector.transfer_write %2, %3[%c0, %c0] {in_bounds = [true, true]} : vector<256x128xf32>, tensor<256x128xf32>
    return %4 : tensor<256x128xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [256, 128] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_pack_no_vector_sizes(%arg0: tensor<64x4xf32>, %arg1: tensor<2x4x16x2xf32>) -> tensor<2x4x16x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<64x4xf32>, vector<64x4xf32>
    %1 = vector.shape_cast %0 : vector<64x4xf32> to vector<4x16x2x2xf32>
    %2 = vector.transpose %1, [2, 0, 1, 3] : vector<4x16x2x2xf32> to vector<2x4x16x2xf32>
    %3 = tensor.empty() : tensor<2x4x16x2xf32>
    %4 = vector.transfer_write %2, %3[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<2x4x16x2xf32>, tensor<2x4x16x2xf32>
    return %4 : tensor<2x4x16x2xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_padded_pack_no_vector_sizes(%arg0: tensor<32x7x15xf32>, %arg1: tensor<32x4x1x16x2xf32>) -> tensor<32x4x1x16x2xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, false, false]} : tensor<32x7x15xf32>, vector<32x8x16xf32>
    %1 = vector.shape_cast %0 : vector<32x8x16xf32> to vector<32x4x2x1x16xf32>
    %2 = vector.transpose %1, [0, 1, 3, 4, 2] : vector<32x4x2x1x16xf32> to vector<32x4x1x16x2xf32>
    %3 = tensor.empty() : tensor<32x4x1x16x2xf32>
    %4 = vector.transfer_write %2, %3[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true, true]} : vector<32x4x1x16x2xf32>, tensor<32x4x1x16x2xf32>
    return %4 : tensor<32x4x1x16x2xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_unpack_no_vector_sizes(%arg0: tensor<8x8x32x16xf32>, %arg1: tensor<256x128xf32>) -> tensor<256x128xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<8x8x32x16xf32>, vector<8x8x32x16xf32>
    %1 = vector.transpose %0, [0, 2, 1, 3] : vector<8x8x32x16xf32> to vector<8x32x8x16xf32>
    %2 = vector.shape_cast %1 : vector<8x32x8x16xf32> to vector<256x128xf32>
    %3 = tensor.empty() : tensor<256x128xf32>
    %4 = vector.transfer_write %2, %3[%c0, %c0] {in_bounds = [true, true]} : vector<256x128xf32>, tensor<256x128xf32>
    return %4 : tensor<256x128xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_unpack_no_vector_sizes_slice_output(%arg0: tensor<8x4x16x16xf32>, %arg1: tensor<64x127xf32>) -> tensor<64x127xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : tensor<8x4x16x16xf32>, vector<8x4x16x16xf32>
    %1 = vector.transpose %0, [1, 2, 0, 3] : vector<8x4x16x16xf32> to vector<4x16x8x16xf32>
    %2 = vector.shape_cast %1 : vector<4x16x8x16xf32> to vector<64x128xf32>
    %3 = tensor.empty() : tensor<64x127xf32>
    %4 = vector.transfer_write %2, %3[%c0, %c0] {in_bounds = [true, false]} : vector<64x128xf32>, tensor<64x127xf32>
    return %4 : tensor<64x127xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func @test_vectorize_unpack_no_vector_sizes_permute(%arg0: tensor<4x7x4xf32>, %arg1: tensor<7x16xf32>) -> tensor<7x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : tensor<4x7x4xf32>, vector<4x7x4xf32>
    %1 = vector.transpose %0, [1, 0, 2] : vector<4x7x4xf32> to vector<7x4x4xf32>
    %2 = vector.shape_cast %1 : vector<7x4x4xf32> to vector<7x16xf32>
    %3 = tensor.empty() : tensor<7x16xf32>
    %4 = vector.transfer_write %2, %3[%c0, %c0] {in_bounds = [true, true]} : vector<7x16xf32>, tensor<7x16xf32>
    return %4 : tensor<7x16xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func private @insert_slice_static_sizes(%arg0: tensor<?x3x?x1xi32>) -> tensor<5x3xi32> {
    %c2 = arith.constant 2 : index
    %0 = tensor.empty() : tensor<5x3xi32>
    %extracted_slice = tensor.extract_slice %arg0[0, %c2, 0, 0] [1, 1, 5, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<5x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %1 = vector.create_mask %c5, %c1 : vector<8x1xi1>
    %c0 = arith.constant 0 : index
    %2 = vector.mask %1 { vector.transfer_read %extracted_slice[%c0, %c0], %c0_i32 : tensor<5x1xi32>, vector<8x1xi32> } : vector<8x1xi1> -> vector<8x1xi32>
    %3 = vector.mask %1 { vector.transfer_write %2, %0[%c0, %c2] : vector<8x1xi32>, tensor<5x3xi32> } : vector<8x1xi1> -> tensor<5x3xi32>
    return %3 : tensor<5x3xi32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [8, 1] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func private @insert_slice_dynamic_src_dim(%arg0: tensor<?x3x?x1xi32>, %arg1: index) -> tensor<5x3xi32> {
    %c2 = arith.constant 2 : index
    %0 = tensor.empty() : tensor<5x3xi32>
    %extracted_slice = tensor.extract_slice %arg0[0, %c2, 0, 0] [1, 1, %arg1, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<?x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %1 = vector.create_mask %arg1, %c1 : vector<8x1xi1>
    %c0 = arith.constant 0 : index
    %2 = vector.mask %1 { vector.transfer_read %extracted_slice[%c0, %c0], %c0_i32 : tensor<?x1xi32>, vector<8x1xi32> } : vector<8x1xi1> -> vector<8x1xi32>
    %3 = vector.mask %1 { vector.transfer_write %2, %0[%c0, %c2] : vector<8x1xi32>, tensor<5x3xi32> } : vector<8x1xi1> -> tensor<5x3xi32>
    return %3 : tensor<5x3xi32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [8, 1] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func private @insert_slice_dynamic_dest_dim(%arg0: tensor<?x3x?x1xi32>, %arg1: index) -> tensor<?x3xi32> {
    %c2 = arith.constant 2 : index
    %0 = tensor.empty(%arg1) : tensor<?x3xi32>
    %extracted_slice = tensor.extract_slice %arg0[0, %c2, 0, 0] [1, 1, 5, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<5x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %1 = vector.create_mask %c5, %c1 : vector<8x1xi1>
    %c0 = arith.constant 0 : index
    %2 = vector.mask %1 { vector.transfer_read %extracted_slice[%c0, %c0], %c0_i32 : tensor<5x1xi32>, vector<8x1xi32> } : vector<8x1xi1> -> vector<8x1xi32>
    %3 = vector.mask %1 { vector.transfer_write %2, %0[%c0, %c2] : vector<8x1xi32>, tensor<?x3xi32> } : vector<8x1xi1> -> tensor<?x3xi32>
    return %3 : tensor<?x3xi32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [8, 1] : !transform.any_op
      transform.yield 
    }
  }
}

// -----
module {
  func.func private @insert_slice_dynamic_source_and_dest_dim(%arg0: tensor<?x3x?x1xi32>, %arg1: index) -> tensor<?x3xi32> {
    %c2 = arith.constant 2 : index
    %0 = tensor.empty(%arg1) : tensor<?x3xi32>
    %extracted_slice = tensor.extract_slice %arg0[0, %c2, 0, 0] [1, 1, %arg1, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<?x1xi32>
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %1 = vector.create_mask %arg1, %c1 : vector<8x1xi1>
    %c0 = arith.constant 0 : index
    %2 = vector.mask %1 { vector.transfer_read %extracted_slice[%c0, %c0], %c0_i32 : tensor<?x1xi32>, vector<8x1xi32> } : vector<8x1xi1> -> vector<8x1xi32>
    %3 = vector.mask %1 { vector.transfer_write %2, %0[%c0, %c2] : vector<8x1xi32>, tensor<?x3xi32> } : vector<8x1xi1> -> tensor<?x3xi32>
    return %3 : tensor<?x3xi32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.structured.vectorize %0 vector_sizes [8, 1] : !transform.any_op
      transform.yield 
    }
  }
}

