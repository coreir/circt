// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @nested_ifs(
// CHECK-SAME:                               %[[VAL_0:.*]]: none, ...) -> none attributes {argNames = ["inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_1:.*]]:4 = fork [4] %[[VAL_0]] : none
// CHECK:           %[[VAL_2:.*]] = constant %[[VAL_1]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_3:.*]]:2 = fork [2] %[[VAL_2]] : index
// CHECK:           %[[VAL_4:.*]] = constant %[[VAL_1]]#1 {value = -1 : index} : index
// CHECK:           %[[VAL_5:.*]] = arith.muli %[[VAL_3]]#0, %[[VAL_4]] : index
// CHECK:           %[[VAL_6:.*]]:2 = fork [2] %[[VAL_5]] : index
// CHECK:           %[[VAL_7:.*]] = constant %[[VAL_1]]#0 {value = 20 : index} : index
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_6]]#1, %[[VAL_7]] : index
// CHECK:           %[[VAL_9:.*]] = arith.cmpi sge, %[[VAL_8]], %[[VAL_3]]#1 : index
// CHECK:           %[[VAL_10:.*]]:2 = fork [2] %[[VAL_9]] : i1
// CHECK:           %[[VAL_11:.*]], %[[VAL_12:.*]] = cond_br %[[VAL_10]]#1, %[[VAL_1]]#3 : none
// CHECK:           %[[VAL_13:.*]], %[[VAL_14:.*]] = cond_br %[[VAL_10]]#0, %[[VAL_6]]#0 : index
// CHECK:           %[[VAL_15:.*]] = merge %[[VAL_13]] : index
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = control_merge %[[VAL_11]] : none
// CHECK:           %[[VAL_18:.*]]:3 = fork [3] %[[VAL_16]] : none
// CHECK:           sink %[[VAL_17]] : index
// CHECK:           %[[VAL_19:.*]] = constant %[[VAL_18]]#1 {value = 0 : index} : index
// CHECK:           %[[VAL_20:.*]] = constant %[[VAL_18]]#0 {value = -10 : index} : index
// CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_15]], %[[VAL_20]] : index
// CHECK:           %[[VAL_22:.*]] = arith.cmpi sge, %[[VAL_21]], %[[VAL_19]] : index
// CHECK:           %[[VAL_23:.*]], %[[VAL_24:.*]] = cond_br %[[VAL_22]], %[[VAL_18]]#2 : none
// CHECK:           %[[VAL_25:.*]], %[[VAL_26:.*]] = control_merge %[[VAL_23]] : none
// CHECK:           sink %[[VAL_26]] : index
// CHECK:           %[[VAL_27:.*]] = br %[[VAL_25]] : none
// CHECK:           %[[VAL_28:.*]], %[[VAL_29:.*]] = control_merge %[[VAL_27]], %[[VAL_24]] : none
// CHECK:           sink %[[VAL_29]] : index
// CHECK:           %[[VAL_30:.*]] = br %[[VAL_28]] : none
// CHECK:           %[[VAL_31:.*]] = merge %[[VAL_14]] : index
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = control_merge %[[VAL_12]] : none
// CHECK:           %[[VAL_34:.*]]:3 = fork [3] %[[VAL_32]] : none
// CHECK:           sink %[[VAL_33]] : index
// CHECK:           %[[VAL_35:.*]] = constant %[[VAL_34]]#1 {value = 0 : index} : index
// CHECK:           %[[VAL_36:.*]] = constant %[[VAL_34]]#0 {value = -10 : index} : index
// CHECK:           %[[VAL_37:.*]] = arith.addi %[[VAL_31]], %[[VAL_36]] : index
// CHECK:           %[[VAL_38:.*]] = arith.cmpi sge, %[[VAL_37]], %[[VAL_35]] : index
// CHECK:           %[[VAL_39:.*]], %[[VAL_40:.*]] = cond_br %[[VAL_38]], %[[VAL_34]]#2 : none
// CHECK:           %[[VAL_41:.*]], %[[VAL_42:.*]] = control_merge %[[VAL_39]] : none
// CHECK:           sink %[[VAL_42]] : index
// CHECK:           %[[VAL_43:.*]] = br %[[VAL_41]] : none
// CHECK:           %[[VAL_44:.*]], %[[VAL_45:.*]] = control_merge %[[VAL_43]], %[[VAL_40]] : none
// CHECK:           sink %[[VAL_45]] : index
// CHECK:           %[[VAL_46:.*]] = br %[[VAL_44]] : none
// CHECK:           %[[VAL_47:.*]], %[[VAL_48:.*]] = control_merge %[[VAL_46]], %[[VAL_30]] : none
// CHECK:           sink %[[VAL_48]] : index
// CHECK:           return %[[VAL_47]] : none
// CHECK:         }
  func @nested_ifs() {
    %c0 = arith.constant 0 : index
    %c-1 = arith.constant -1 : index
    %1 = arith.muli %c0, %c-1 : index
    %c20 = arith.constant 20 : index
    %2 = arith.addi %1, %c20 : index
    %3 = arith.cmpi sge, %2, %c0 : index
    cond_br %3, ^bb1, ^bb4
  ^bb1: // pred: ^bb0
    %c0_0 = arith.constant 0 : index
    %c-10 = arith.constant -10 : index
    %4 = arith.addi %1, %c-10 : index
    %5 = arith.cmpi sge, %4, %c0_0 : index
    cond_br %5, ^bb2, ^bb3
  ^bb2: // pred: ^bb1
    br ^bb3
  ^bb3: // 2 preds: ^bb1, ^bb2
    br ^bb7
  ^bb4: // pred: ^bb0
    %c0_1 = arith.constant 0 : index
    %c-10_2 = arith.constant -10 : index
    %6 = arith.addi %1, %c-10_2 : index
    %7 = arith.cmpi sge, %6, %c0_1 : index
    cond_br %7, ^bb5, ^bb6
  ^bb5: // pred: ^bb4
    br ^bb6
  ^bb6: // 2 preds: ^bb4, ^bb5
    br ^bb7
  ^bb7: // 2 preds: ^bb3, ^bb6
    return
  }
