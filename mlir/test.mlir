module {
  func.func @vectorize_dynamic_reduction(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %c0_1 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.create_mask %dim, %dim_0 : vector<4x8xi1>
    %1 = vector.mask %0 { vector.transfer_read %arg0[%c0_1, %c0_1], %cst {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %2 = vector.create_mask %dim : vector<4xi1>
    %3 = vector.mask %2 { vector.transfer_read %arg1[%c0_1], %cst_2 {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
    %4 = vector.mask %0 { vector.multi_reduction <add>, %1, %3 [1] : vector<4x8xf32> to vector<4xf32> } : vector<4x8xi1> -> vector<4xf32>
    %c0_3 = arith.constant 0 : index
    %5 = vector.mask %2 { vector.transfer_write %4, %arg1[%c0_3] {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>
    return %5 : tensor<?xf32>
  }
  transform.sequence failures(propagate) {
  ^bb0(%arg0: !pdl.operation):
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg0
    transform.structured.masked_vectorize %0 vector_sizes [4, 8]
  }
}


