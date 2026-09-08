// RUN: triton-opt "--triton-to-linalg=compile-on-910-95=false" %s | FileCheck %s --check-prefix=A2A3
// RUN: triton-opt "--triton-to-linalg=compile-on-910-95=false" --mlir-print-op-generic %s | FileCheck %s --check-prefix=SEGMENTS
// RUN: triton-opt "--triton-to-linalg=compile-on-910-95=true" %s | FileCheck %s --check-prefix=A5

// SEGMENTS-LABEL: "builtin.module"
// SEGMENTS: operandSegmentSizes = array<i32: 1, 1, 0>

// isinf needs no extra buffer on any target: its 6-stage RAW chain runs in
// count mode with the output memref as in-place scratch and vabs in place of
// a vand sign-mask (see lib/a3/simd/isinf.cce and references/operator-
// patterns.md). Both A2/A3 and A5 emit a plain CustomOp with no extra_buffers.

// A2A3-LABEL: func.func @test_isinf_128
// A2A3: hivm.hir.custom
// A2A3-NOT: extra_buffers_
// A2A3-SAME: symbol = "isinf_fp32"

// A2A3-LABEL: func.func @test_isinf_300
// A2A3: hivm.hir.custom
// A2A3-NOT: extra_buffers_
// A2A3-SAME: symbol = "isinf_fp32"

// A2A3-LABEL: func.func @test_isinf_16
// A2A3: hivm.hir.custom
// A2A3-NOT: extra_buffers_
// A2A3-SAME: symbol = "isinf_fp32"

// A5-LABEL: func.func @test_isinf_128
// A5: hivm.hir.custom
// A5-NOT: extra_buffers_
// A5-SAME: symbol = "isinf_fp32"

// Arithmetic-only A2/A3 entry points can use operation-sized rows. Operators
// with normal-mode compares process fixed 64-lane tiles, so their workspace
// size is independent of the tensor extent.

// A2A3-LABEL: func.func @test_tan_16
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [512]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "tan_fp32"

// A2A3-LABEL: func.func @test_tan_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [512]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "tan_fp32"

// A2A3-LABEL: func.func @test_atan2_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [640]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "atan2_fp32"

// A2A3-LABEL: func.func @test_fast_pow_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [512]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "fast_pow_fp32"

// A2A3-LABEL: func.func @test_float2ull_rd_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [320]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2ull_rd_fp32"

// A2A3-LABEL: func.func @test_float2ull_rn_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [320]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2ull_rn_fp32"

// A2A3-LABEL: func.func @test_float2ull_ru_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [320]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2ull_ru_fp32"

// A2A3-LABEL: func.func @test_float2ull_rz_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [320]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "float2ull_rz_fp32"

// A2A3-LABEL: func.func @test_tgamma_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [576]
// A2A3-SAME: extra_buffers_types = [f32]
// A2A3-SAME: symbol = "tgamma_fp32"

// A2A3-LABEL: func.func @test_copysign_2048
// A2A3: hivm.hir.custom
// A2A3-SAME: extra_buffers_sizes = [2048]
// A2A3-SAME: extra_buffers_types = [i32]
// A2A3-SAME: symbol = "copysign_fp32"

// A5-LABEL: func.func @test_tan_16
// A5: hivm.hir.custom
// A5-NOT: extra_buffers_
// A5-SAME: symbol = "tan_fp32"

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func @test_isinf_128(%arg0: tensor<128xf32>) -> tensor<128xi1> {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_isinf_fp32"
    } : (tensor<128xf32>) -> tensor<128xi1>
    tt.return %0 : tensor<128xi1>
  }

  tt.func @test_isinf_300(%arg0: tensor<300xf32>) -> tensor<300xi1> {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_isinf_fp32"
    } : (tensor<300xf32>) -> tensor<300xi1>
    tt.return %0 : tensor<300xi1>
  }

  tt.func @test_isinf_16(%arg0: tensor<16xf32>) -> tensor<16xi1> {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_isinf_fp32"
    } : (tensor<16xf32>) -> tensor<16xi1>
    tt.return %0 : tensor<16xi1>
  }

  tt.func @test_tan_16(%arg0: tensor<16xf32>) -> tensor<16xf32> {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_tan_fp32"
    } : (tensor<16xf32>) -> tensor<16xf32>
    tt.return %0 : tensor<16xf32>
  }

  tt.func @test_tan_2048(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_tan_fp32"
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    tt.return %0 : tensor<2048xf32>
  }

  tt.func @test_atan2_2048(%arg0: tensor<2048xf32>,
                            %arg1: tensor<2048xf32>) -> tensor<2048xf32> {
    %0 = tt.extern_elementwise %arg0, %arg1 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_atan2_fp32"
    } : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    tt.return %0 : tensor<2048xf32>
  }

  tt.func @test_fast_pow_2048(%arg0: tensor<2048xf32>,
                               %arg1: tensor<2048xf32>) -> tensor<2048xf32> {
    %0 = tt.extern_elementwise %arg0, %arg1 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_fast_pow_fp32"
    } : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    tt.return %0 : tensor<2048xf32>
  }

  tt.func @test_float2ull_rd_2048(%arg0: tensor<2048xf32>)
      -> tensor<2048xi64> {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true,
      symbol = "__hmf_float2ull_rd_fp32"
    } : (tensor<2048xf32>) -> tensor<2048xi64>
    tt.return %0 : tensor<2048xi64>
  }

  tt.func @test_float2ull_rn_2048(%arg0: tensor<2048xf32>)
      -> tensor<2048xi64> {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true,
      symbol = "__hmf_float2ull_rn_fp32"
    } : (tensor<2048xf32>) -> tensor<2048xi64>
    tt.return %0 : tensor<2048xi64>
  }

  tt.func @test_float2ull_ru_2048(%arg0: tensor<2048xf32>)
      -> tensor<2048xi64> {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true,
      symbol = "__hmf_float2ull_ru_fp32"
    } : (tensor<2048xf32>) -> tensor<2048xi64>
    tt.return %0 : tensor<2048xi64>
  }

  tt.func @test_float2ull_rz_2048(%arg0: tensor<2048xf32>)
      -> tensor<2048xi64> {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true,
      symbol = "__hmf_float2ull_rz_fp32"
    } : (tensor<2048xf32>) -> tensor<2048xi64>
    tt.return %0 : tensor<2048xi64>
  }

  tt.func @test_tgamma_2048(%arg0: tensor<2048xf32>) -> tensor<2048xf32> {
    %0 = tt.extern_elementwise %arg0 {
      libname = "", libpath = "", pure = true, symbol = "__hmf_tgamma_fp32"
    } : (tensor<2048xf32>) -> tensor<2048xf32>
    tt.return %0 : tensor<2048xf32>
  }

  tt.func @test_copysign_2048(%arg0: tensor<2048xf32>,
                               %arg1: tensor<2048xf32>) -> tensor<2048xf32> {
    %0 = tt.extern_elementwise %arg0, %arg1 {
      libname = "", libpath = "", pure = true,
      symbol = "__hmf_copysign_fp32"
    } : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    tt.return %0 : tensor<2048xf32>
  }
}
