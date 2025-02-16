// RUN: circt-opt %s --lower-scf-to-calyx -canonicalize -split-input-file | FileCheck %s

// CHECK:      module  {
// CHECK-NEXT:   calyx.program "main" {
// CHECK-LABEL:    calyx.component @main(%in0: i32, %in1: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
// CHECK-DAG:        %true = hw.constant true
// CHECK-DAG:        %std_sub_0.left, %std_sub_0.right, %std_sub_0.out = calyx.std_sub @std_sub_0 : i32, i32, i32
// CHECK-DAG:        %std_lsh_0.left, %std_lsh_0.right, %std_lsh_0.out = calyx.std_lsh @std_lsh_0 : i32, i32, i32
// CHECK-DAG:        %std_add_0.left, %std_add_0.right, %std_add_0.out = calyx.std_add @std_add_0 : i32, i32, i32
// CHECK-DAG:        %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:       calyx.wires  {
// CHECK-NEXT:         calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:         calyx.group @ret_assign_0  {
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.in = %std_sub_0.out : i32
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:           calyx.assign %std_sub_0.left = %std_lsh_0.out : i32
// CHECK-NEXT:           calyx.assign %std_lsh_0.left = %std_add_0.out : i32
// CHECK-NEXT:           calyx.assign %std_add_0.left = %in0 : i32
// CHECK-NEXT:           calyx.assign %std_add_0.right = %in1 : i32
// CHECK-NEXT:           calyx.assign %std_lsh_0.right = %in0 : i32
// CHECK-NEXT:           calyx.assign %std_sub_0.right = %std_add_0.out : i32
// CHECK-NEXT:           calyx.group_done %ret_arg0_reg.done : i1
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.control  {
// CHECK-NEXT:         calyx.seq  {
// CHECK-NEXT:           calyx.enable @ret_assign_0
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     } {toplevel}
// CHECK-NEXT:   }
// CHECK-NEXT: }
module {
  func @main(%a0 : i32, %a1 : i32) -> i32 {
    %0 = arith.addi %a0, %a1 : i32
    %1 = arith.shli %0, %a0 : i32
    %2 = arith.subi %1, %0 : i32
    return %2 : i32
  }
}

// -----

// Test multiple return values.

// CHECK:      module  {
// CHECK-NEXT:   calyx.program "main"  {
// CHECK-LABEL:    calyx.component @main(%in0: i32, %in1: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %out1: i32, %done: i1 {done}) {
// CHECK-DAG:        %true = hw.constant true
// CHECK-DAG:        %ret_arg1_reg.in, %ret_arg1_reg.write_en, %ret_arg1_reg.clk, %ret_arg1_reg.reset, %ret_arg1_reg.out, %ret_arg1_reg.done = calyx.register @ret_arg1_reg : i32, i1, i1, i1, i32, i1
// CHECK-DAG:        %ret_arg0_reg.in, %ret_arg0_reg.write_en, %ret_arg0_reg.clk, %ret_arg0_reg.reset, %ret_arg0_reg.out, %ret_arg0_reg.done = calyx.register @ret_arg0_reg : i32, i1, i1, i1, i32, i1
// CHECK-NEXT:       calyx.wires  {
// CHECK-NEXT:         calyx.assign %out1 = %ret_arg1_reg.out : i32
// CHECK-NEXT:         calyx.assign %out0 = %ret_arg0_reg.out : i32
// CHECK-NEXT:         calyx.group @ret_assign_0  {
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.in = %in0 : i32
// CHECK-NEXT:           calyx.assign %ret_arg0_reg.write_en = %true : i1
// CHECK-NEXT:           calyx.assign %ret_arg1_reg.in = %in1 : i32
// CHECK-NEXT:           calyx.assign %ret_arg1_reg.write_en = %true : i1
// CHECK-NEXT:           %0 = comb.and %ret_arg0_reg.done, %ret_arg1_reg.done : i1
// CHECK-NEXT:           calyx.group_done %0 ? %true : i1
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       calyx.control  {
// CHECK-NEXT:         calyx.seq  {
// CHECK-NEXT:           calyx.enable @ret_assign_0
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     } {toplevel}
// CHECK-NEXT:   }
// CHECK-NEXT: }
module {
  func @main(%a0 : i32, %a1 : i32) -> (i32, i32) {
    return %a0, %a1 : i32, i32
  }
}

// -----

module {
  func @main(%a0 : i32, %a1 : i32) -> i32 {
// CHECK:       calyx.group @bb0_0  {
// CHECK-DAG:    calyx.assign %mult_reg_0_reg.in = %std_mult_pipe_0.out : i32
// CHECK-DAG:    calyx.assign %mult_reg_0_reg.write_en = %std_mult_pipe_0.done : i1
// CHECK-DAG:    calyx.assign %std_mult_pipe_0.left = %in0 : i32
// CHECK-DAG:    calyx.assign %std_mult_pipe_0.right = %in1 : i32
// CHECK-DAG:    calyx.assign %std_mult_pipe_0.go = %true : i1
// CHECK-DAG:    calyx.group_done %mult_reg_0_reg.done : i1
// CHECK-NEXT:  }
    %0 = arith.muli %a0, %a1 : i32
    return %0 : i32
  }
}
