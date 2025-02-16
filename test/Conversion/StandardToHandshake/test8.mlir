// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @simple_loop(
// CHECK-SAME:                                %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]] = br %[[VAL_0]] : none
// CHECK:           %[[VAL_2:.*]], %[[VAL_3:.*]] = control_merge %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]]:2 = fork [2] %[[VAL_2]] : none
// CHECK:           sink %[[VAL_3]] : index
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_4]]#0 {value = 42 : index} : index
// CHECK:           %[[VAL_6:.*]] = br %[[VAL_4]]#1 : none
// CHECK:           %[[VAL_7:.*]] = br %[[VAL_5]] : index
// CHECK:           %[[VAL_8:.*]], %[[VAL_9:.*]] = control_merge %[[VAL_6]] : none
// CHECK:           %[[VAL_10:.*]] = buffer [1] %[[VAL_11:.*]] {initValues = [0], sequential = true} : i1
// CHECK:           %[[VAL_12:.*]]:2 = fork [2] %[[VAL_10]] : i1
// CHECK:           %[[VAL_13:.*]] = mux %[[VAL_12]]#1 {{\[}}%[[VAL_8]], %[[VAL_14:.*]]] : i1, none
// CHECK:           %[[VAL_15:.*]] = mux %[[VAL_9]] {{\[}}%[[VAL_7]]] : index, index
// CHECK:           %[[VAL_16:.*]] = mux %[[VAL_12]]#0 {{\[}}%[[VAL_15]], %[[VAL_17:.*]]] : i1, index
// CHECK:           %[[VAL_18:.*]]:3 = fork [3] %[[VAL_16]] : index
// CHECK:           %[[VAL_11]] = merge %[[VAL_19:.*]]#0 : i1
// CHECK:           %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_18]]#0, %[[VAL_18]]#1 : index
// CHECK:           %[[VAL_19]]:3 = fork [3] %[[VAL_20]] : i1
// CHECK:           %[[VAL_21:.*]], %[[VAL_22:.*]] = cond_br %[[VAL_19]]#2, %[[VAL_18]]#2 : index
// CHECK:           sink %[[VAL_22]] : index
// CHECK:           %[[VAL_23:.*]], %[[VAL_24:.*]] = cond_br %[[VAL_19]]#1, %[[VAL_13]] : none
// CHECK:           %[[VAL_25:.*]] = merge %[[VAL_21]] : index
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = control_merge %[[VAL_23]] : none
// CHECK:           %[[VAL_28:.*]]:3 = fork [3] %[[VAL_26]] : none
// CHECK:           sink %[[VAL_27]] : index
// CHECK:           %[[VAL_29:.*]] = constant %[[VAL_28]]#1 {value = 52 : index} : index
// CHECK:           sink %[[VAL_29]] : index
// CHECK:           %[[VAL_30:.*]] = constant %[[VAL_28]]#0 {value = 62 : index} : index
// CHECK:           sink %[[VAL_30]] : index
// CHECK:           %[[VAL_17]] = br %[[VAL_25]] : index
// CHECK:           %[[VAL_14]] = br %[[VAL_28]]#2 : none
// CHECK:           %[[VAL_31:.*]], %[[VAL_32:.*]] = control_merge %[[VAL_24]] : none
// CHECK:           sink %[[VAL_32]] : index
// CHECK:           return %[[VAL_31]] : none
// CHECK:         }
func @simple_loop() {
^bb0:
  br ^bb1
^bb1:	// pred: ^bb0
  %c42 = arith.constant 42 : index
  br ^bb2
^bb2:	// 2 preds: ^bb1, ^bb3
  %1 = arith.cmpi slt, %c42, %c42 : index
  cond_br %1, ^bb3, ^bb4
^bb3:	// pred: ^bb2
  %c52 = arith.constant 52 : index
  %c62 = arith.constant 62 : index
  br ^bb2
^bb4:	// pred: ^bb2
  return
}
