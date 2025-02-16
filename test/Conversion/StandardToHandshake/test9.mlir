// NOTE: Assertions have been autogenerated by utils/update_mlir_test_checks.py
// RUN: circt-opt -lower-std-to-handshake %s | FileCheck %s
// CHECK-LABEL:   handshake.func @affine_dma_wait(
// CHECK-SAME:                                    %[[VAL_0:.*]]: index,
// CHECK-SAME:                                    %[[VAL_1:.*]]: none, ...) -> none attributes {argNames = ["in0", "inCtrl"], resNames = ["outCtrl"]} {
// CHECK:           %[[VAL_2:.*]] = merge %[[VAL_0]] : index
// CHECK:           %[[VAL_3:.*]]:5 = fork [5] %[[VAL_1]] : none
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_5:.*]] = constant %[[VAL_3]]#3 {value = 64 : index} : index
// CHECK:           %[[VAL_6:.*]] = constant %[[VAL_3]]#2 {value = 0 : index} : index
// CHECK:           %[[VAL_7:.*]] = constant %[[VAL_3]]#1 {value = 10 : index} : index
// CHECK:           %[[VAL_8:.*]] = constant %[[VAL_3]]#0 {value = 1 : index} : index
// CHECK:           %[[VAL_9:.*]] = br %[[VAL_2]] : index
// CHECK:           %[[VAL_10:.*]] = br %[[VAL_3]]#4 : none
// CHECK:           %[[VAL_11:.*]] = br %[[VAL_4]] : memref<1xi32>
// CHECK:           %[[VAL_12:.*]] = br %[[VAL_5]] : index
// CHECK:           %[[VAL_13:.*]] = br %[[VAL_6]] : index
// CHECK:           %[[VAL_14:.*]] = br %[[VAL_7]] : index
// CHECK:           %[[VAL_15:.*]] = br %[[VAL_8]] : index
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = control_merge %[[VAL_10]] : none
// CHECK:           %[[VAL_18:.*]]:6 = fork [6] %[[VAL_17]] : index
// CHECK:           %[[VAL_19:.*]] = buffer [1] %[[VAL_20:.*]] {initValues = [0], sequential = true} : i1
// CHECK:           %[[VAL_21:.*]]:7 = fork [7] %[[VAL_19]] : i1
// CHECK:           %[[VAL_22:.*]] = mux %[[VAL_21]]#6 {{\[}}%[[VAL_16]], %[[VAL_23:.*]]] : i1, none
// CHECK:           %[[VAL_24:.*]] = mux %[[VAL_18]]#5 {{\[}}%[[VAL_14]]] : index, index
// CHECK:           %[[VAL_25:.*]] = mux %[[VAL_21]]#5 {{\[}}%[[VAL_24]], %[[VAL_26:.*]]] : i1, index
// CHECK:           %[[VAL_27:.*]]:2 = fork [2] %[[VAL_25]] : index
// CHECK:           %[[VAL_28:.*]] = mux %[[VAL_18]]#4 {{\[}}%[[VAL_9]]] : index, index
// CHECK:           %[[VAL_29:.*]] = mux %[[VAL_21]]#4 {{\[}}%[[VAL_28]], %[[VAL_30:.*]]] : i1, index
// CHECK:           %[[VAL_31:.*]] = mux %[[VAL_18]]#3 {{\[}}%[[VAL_11]]] : index, memref<1xi32>
// CHECK:           %[[VAL_32:.*]] = mux %[[VAL_21]]#3 {{\[}}%[[VAL_31]], %[[VAL_33:.*]]] : i1, memref<1xi32>
// CHECK:           %[[VAL_34:.*]] = mux %[[VAL_18]]#2 {{\[}}%[[VAL_12]]] : index, index
// CHECK:           %[[VAL_35:.*]] = mux %[[VAL_21]]#2 {{\[}}%[[VAL_34]], %[[VAL_36:.*]]] : i1, index
// CHECK:           %[[VAL_37:.*]] = mux %[[VAL_18]]#1 {{\[}}%[[VAL_15]]] : index, index
// CHECK:           %[[VAL_38:.*]] = mux %[[VAL_21]]#1 {{\[}}%[[VAL_37]], %[[VAL_39:.*]]] : i1, index
// CHECK:           %[[VAL_40:.*]] = mux %[[VAL_18]]#0 {{\[}}%[[VAL_13]]] : index, index
// CHECK:           %[[VAL_41:.*]] = mux %[[VAL_21]]#0 {{\[}}%[[VAL_40]], %[[VAL_42:.*]]] : i1, index
// CHECK:           %[[VAL_43:.*]]:2 = fork [2] %[[VAL_41]] : index
// CHECK:           %[[VAL_20]] = merge %[[VAL_44:.*]]#0 : i1
// CHECK:           %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_43]]#0, %[[VAL_27]]#0 : index
// CHECK:           %[[VAL_44]]:8 = fork [8] %[[VAL_45]] : i1
// CHECK:           %[[VAL_46:.*]], %[[VAL_47:.*]] = cond_br %[[VAL_44]]#7, %[[VAL_27]]#1 : index
// CHECK:           sink %[[VAL_47]] : index
// CHECK:           %[[VAL_48:.*]], %[[VAL_49:.*]] = cond_br %[[VAL_44]]#6, %[[VAL_29]] : index
// CHECK:           sink %[[VAL_49]] : index
// CHECK:           %[[VAL_50:.*]], %[[VAL_51:.*]] = cond_br %[[VAL_44]]#5, %[[VAL_32]] : memref<1xi32>
// CHECK:           sink %[[VAL_51]] : memref<1xi32>
// CHECK:           %[[VAL_52:.*]], %[[VAL_53:.*]] = cond_br %[[VAL_44]]#4, %[[VAL_35]] : index
// CHECK:           sink %[[VAL_53]] : index
// CHECK:           %[[VAL_54:.*]], %[[VAL_55:.*]] = cond_br %[[VAL_44]]#3, %[[VAL_38]] : index
// CHECK:           sink %[[VAL_55]] : index
// CHECK:           %[[VAL_56:.*]], %[[VAL_57:.*]] = cond_br %[[VAL_44]]#2, %[[VAL_22]] : none
// CHECK:           %[[VAL_58:.*]], %[[VAL_59:.*]] = cond_br %[[VAL_44]]#1, %[[VAL_43]]#1 : index
// CHECK:           sink %[[VAL_59]] : index
// CHECK:           %[[VAL_60:.*]] = merge %[[VAL_58]] : index
// CHECK:           %[[VAL_61:.*]]:2 = fork [2] %[[VAL_60]] : index
// CHECK:           %[[VAL_62:.*]] = merge %[[VAL_48]] : index
// CHECK:           %[[VAL_63:.*]]:2 = fork [2] %[[VAL_62]] : index
// CHECK:           %[[VAL_64:.*]] = merge %[[VAL_50]] : memref<1xi32>
// CHECK:           %[[VAL_65:.*]]:2 = fork [2] %[[VAL_64]] : memref<1xi32>
// CHECK:           %[[VAL_66:.*]] = merge %[[VAL_52]] : index
// CHECK:           %[[VAL_67:.*]]:2 = fork [2] %[[VAL_66]] : index
// CHECK:           %[[VAL_68:.*]] = merge %[[VAL_54]] : index
// CHECK:           %[[VAL_69:.*]]:2 = fork [2] %[[VAL_68]] : index
// CHECK:           %[[VAL_70:.*]] = merge %[[VAL_46]] : index
// CHECK:           %[[VAL_71:.*]], %[[VAL_72:.*]] = control_merge %[[VAL_56]] : none
// CHECK:           %[[VAL_73:.*]]:2 = fork [2] %[[VAL_71]] : none
// CHECK:           sink %[[VAL_72]] : index
// CHECK:           %[[VAL_74:.*]] = arith.addi %[[VAL_61]]#1, %[[VAL_63]]#1 : index
// CHECK:           %[[VAL_75:.*]] = constant %[[VAL_73]]#0 {value = 17 : index} : index
// CHECK:           %[[VAL_76:.*]] = arith.addi %[[VAL_74]], %[[VAL_75]] : index
// CHECK:           memref.dma_wait %[[VAL_65]]#1{{\[}}%[[VAL_76]]], %[[VAL_67]]#1 : memref<1xi32>
// CHECK:           %[[VAL_77:.*]] = arith.addi %[[VAL_61]]#0, %[[VAL_69]]#1 : index
// CHECK:           %[[VAL_30]] = br %[[VAL_63]]#0 : index
// CHECK:           %[[VAL_33]] = br %[[VAL_65]]#0 : memref<1xi32>
// CHECK:           %[[VAL_36]] = br %[[VAL_67]]#0 : index
// CHECK:           %[[VAL_39]] = br %[[VAL_69]]#0 : index
// CHECK:           %[[VAL_26]] = br %[[VAL_70]] : index
// CHECK:           %[[VAL_23]] = br %[[VAL_73]]#1 : none
// CHECK:           %[[VAL_42]] = br %[[VAL_77]] : index
// CHECK:           %[[VAL_78:.*]], %[[VAL_79:.*]] = control_merge %[[VAL_57]] : none
// CHECK:           sink %[[VAL_79]] : index
// CHECK:           return %[[VAL_78]] : none
// CHECK:         }
func @affine_dma_wait(%arg0: index) {
  %0 = memref.alloc() : memref<1xi32>
  %c64 = arith.constant 64 : index
  %c0 = arith.constant 0 : index
  %c10 = arith.constant 10 : index
  %c1 = arith.constant 1 : index
  br ^bb1(%c0 : index)
^bb1(%1: index):      // 2 preds: ^bb0, ^bb2
  %2 = arith.cmpi slt, %1, %c10 : index
  cond_br %2, ^bb2, ^bb3
^bb2: // pred: ^bb1
  %3 = arith.addi %1, %arg0 : index
  %c17 = arith.constant 17 : index
  %4 = arith.addi %3, %c17 : index
  memref.dma_wait %0[%4], %c64 : memref<1xi32>
  %5 = arith.addi %1, %c1 : index
  br ^bb1(%5 : index)
^bb3: // pred: ^bb1
  return
}
