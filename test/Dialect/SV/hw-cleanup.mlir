// RUN: circt-opt -hw-cleanup='aggressive-if-op-merge=true' %s | FileCheck %s

//CHECK-LABEL: hw.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
//CHECK-NEXT:   sv.initial {
//CHECK-NEXT:     sv.fwrite "Middle\0A"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:     sv.fwrite "A1"
//CHECK-NEXT:     sv.fwrite "A2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.alwaysff(posedge %arg1)  {
//CHECK-NEXT:     sv.fwrite "B1"
//CHECK-NEXT:     sv.fwrite "B2"
//CHECK-NEXT:   }
//CHECK-NEXT:   hw.output
//CHECK-NEXT: }

hw.module @alwaysff_basic(%arg0: i1, %arg1: i1) {
  sv.alwaysff(posedge %arg0) {
    sv.fwrite "A1"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite "B1"
  }
  sv.initial {
    sv.fwrite "Middle\n"
  }
  sv.alwaysff(posedge %arg0) {
    sv.fwrite "A2"
  }
  sv.alwaysff(posedge %arg1) {
    sv.fwrite "B2"
  }
  hw.output
}

// CHECK-LABEL: hw.module @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite "A1"
// CHECK-NEXT:     sv.fwrite "A2"
// CHECK-NEXT:   }(asyncreset : negedge %arg1)  {
// CHECK-NEXT:     sv.fwrite "B1"
// CHECK-NEXT:     sv.fwrite "B2"
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output
// CHECK-NEXT: }

hw.module @alwaysff_basic_reset(%arg0: i1, %arg1: i1) {
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A1"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A2"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B2"
  }
  hw.output
}


// CHECK-LABEL: hw.module @alwaysff_different_reset(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite "A1"
// CHECK-NEXT:     sv.fwrite "A2"
// CHECK-NEXT:   }(asyncreset : negedge %arg1)  {
// CHECK-NEXT:     sv.fwrite "B1"
// CHECK-NEXT:     sv.fwrite "B2"
// CHECK-NEXT:   }
// CHECK-NEXT:   sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:     sv.fwrite "C1"
// CHECK-NEXT:     sv.fwrite "C2"
// CHECK-NEXT:   }(asyncreset : posedge %arg1)  {
// CHECK-NEXT:     sv.fwrite "D1"
// CHECK-NEXT:     sv.fwrite "D2"
// CHECK-NEXT:   }
// CHECK-NEXT:   hw.output
// CHECK-NEXT: }

hw.module @alwaysff_different_reset(%arg0: i1, %arg1: i1) {
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A1"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "C1"
  } ( asyncreset : posedge %arg1) {
    sv.fwrite "D1"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "A2"
  } ( asyncreset : negedge %arg1) {
    sv.fwrite "B2"
  }
  sv.alwaysff (posedge %arg0) {
    sv.fwrite "C2"
  } ( asyncreset : posedge %arg1) {
    sv.fwrite "D2"
  }
  hw.output
}

//CHECK-LABEL: hw.module @alwaysff_ifdef(%arg0: i1) {
//CHECK-NEXT:  sv.ifdef "FOO" {
//CHECK-NEXT:     sv.alwaysff(posedge %arg0)  {
//CHECK-NEXT:       sv.fwrite "A1"
//CHECK-NEXT:       sv.fwrite "B1"
//CHECK-NEXT:     }
//CHECK-NEXT:   }
//CHECK-NEXT:   hw.output
//CHECK-NEXT: }

hw.module @alwaysff_ifdef(%arg0: i1) {
  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "A1"
    }
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "B1"
    }
  }
  hw.output
}

// CHECK-LABEL: hw.module @ifdef_merge(%arg0: i1) {
// CHECK-NEXT:    sv.ifdef "FOO"  {
// CHECK-NEXT:      sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:        sv.fwrite "A1"
// CHECK-NEXT:        sv.fwrite "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
hw.module @ifdef_merge(%arg0: i1) {
  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "A1"
    }
  }
  sv.ifdef "FOO" {
    sv.alwaysff(posedge %arg0) {
      sv.fwrite "B1"
    }
  }
  hw.output
}

// CHECK-LABEL: hw.module @ifdef_proc_merge(%arg0: i1) {
// CHECK-NEXT:    sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:      %true = hw.constant true
// CHECK-NEXT:      %0 = comb.xor %arg0, %true : i1
// CHECK-NEXT:      sv.ifdef.procedural "FOO"  {
// CHECK-NEXT:        sv.fwrite "A1"
// CHECK-NEXT:        sv.fwrite "%x"(%0) : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      sv.ifdef.procedural "BAR"  {
// CHECK-NEXT:        sv.fwrite "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
hw.module @ifdef_proc_merge(%arg0: i1) {
  sv.alwaysff(posedge %arg0) {
    sv.ifdef.procedural "FOO" {
      sv.fwrite "A1"
    }
    %true = hw.constant true
    %0 = comb.xor %arg0, %true : i1
    sv.ifdef.procedural "FOO" {
       sv.fwrite "%x"(%0) : i1
    }
     sv.ifdef.procedural "BAR" {
       sv.fwrite "B1"
    }
  }
  hw.output
}

// CHECK-LABEL: hw.module @if_merge(%arg0: i1, %arg1: i1) {
// CHECK-NEXT:    sv.alwaysff(posedge %arg0)  {
// CHECK-NEXT:      %true = hw.constant true
// CHECK-NEXT:      %0 = comb.xor %arg1, %true : i1
// CHECK-NEXT:      sv.if %arg1  {
// CHECK-NEXT:        sv.fwrite "A1"
// CHECK-NEXT:        sv.fwrite "%x"(%0) : i1
// CHECK-NEXT:      }
// CHECK-NEXT:      sv.if %0 {
// CHECK-NEXT:        sv.fwrite "B1"
// CHECK-NEXT:      }
// CHECK-NEXT:    }
hw.module @if_merge(%arg0: i1, %arg1: i1) {
  sv.alwaysff(posedge %arg0) {
    sv.if %arg1 {
      sv.fwrite "A1"
    }
    %true = hw.constant true
    %0 = comb.xor %arg1, %true : i1
    sv.if %arg1 {
      sv.fwrite "%x"(%0) : i1
    }
    sv.if %0 {
      sv.fwrite "B1"
    }
  }
  hw.output
}


// CHECK-LABEL: hw.module @initial_merge(%arg0: i1) {
// CHECK-NEXT:    sv.initial {
// CHECK-NEXT:      sv.fwrite "A1"
// CHECK-NEXT:      sv.fwrite "B1"
// CHECK-NEXT:    }
hw.module @initial_merge(%arg0: i1) {
  sv.initial {
    sv.fwrite "A1"
  }
  sv.initial {
    sv.fwrite "B1"
  }
  hw.output
}

//CHECK-LABEL: hw.module @always_basic(%arg0: i1, %arg1: i1) {
//CHECK-NEXT:   sv.initial {
//CHECK-NEXT:     sv.fwrite "Middle\0A"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.always   posedge %arg0   {
//CHECK-NEXT:     sv.fwrite "A1"
//CHECK-NEXT:     sv.fwrite "A2"
//CHECK-NEXT:   }
//CHECK-NEXT:   sv.always   posedge %arg1   {
//CHECK-NEXT:     sv.fwrite "B1"
//CHECK-NEXT:     sv.fwrite "B2"
//CHECK-NEXT:   }
//CHECK-NEXT:   hw.output
//CHECK-NEXT: }
hw.module @always_basic(%arg0: i1, %arg1: i1) {
  sv.always posedge %arg0 {
    sv.fwrite "A1"
  }
  sv.always posedge %arg1 {
    sv.fwrite "B1"
  }
  sv.initial {
    sv.fwrite "Middle\n"
  }
  sv.always posedge %arg0 {
    sv.fwrite "A2"
  }
  sv.always posedge %arg1 {
    sv.fwrite "B2"
  }
  hw.output
}


// CHECK-LABEL: hw.module @alwayscomb_basic(
hw.module @alwayscomb_basic(%a: i1, %b: i1) -> (x: i1, y: i1) {
  %w1 = sv.reg : !hw.inout<i1>
  %w2 = sv.reg : !hw.inout<i1>
  // CHECK: sv.alwayscomb {
  sv.alwayscomb {
    // CHECK-NEXT: sv.bpassign %w1, %a : i1
    sv.bpassign %w1, %a : i1
  }

  %out1 = sv.read_inout %w1 : !hw.inout<i1>

  sv.alwayscomb {
    // CHECK-NEXT: sv.bpassign %w2, %b : i1
    sv.bpassign %w2, %b : i1
  } // CHECK-NEXT: }

  %out2 = sv.read_inout %w1 : !hw.inout<i1>

  hw.output %out1, %out2 : i1, i1
}

// CHECK-LABEL: hw.module @nested_regions(
hw.module @nested_regions() {
  sv.initial {
    sv.ifdef.procedural "L1" {
      sv.ifdef.procedural "L2" {
        sv.ifdef.procedural "L3" {
          sv.fwrite "A"
        }
      }
    }
  }
  sv.initial {
    sv.ifdef.procedural "L1" {
      sv.ifdef.procedural "L2" {
        sv.ifdef.procedural "L3" {
          sv.fwrite "B"
        }
      }
    }
  }
  // CHECK-NEXT:  sv.initial  {
  // CHECK-NEXT:    sv.ifdef.procedural "L1"  {
  // CHECK-NEXT:      sv.ifdef.procedural "L2"  {
  // CHECK-NEXT:        sv.ifdef.procedural "L3"  {
  // CHECK-NEXT:          sv.fwrite "A"
  // CHECK-NEXT:          sv.fwrite "B"
  // CHECK-NEXT:        }
  // CHECK-NEXT:      }
  // CHECK-NEXT:    }
  // CHECK-NEXT:  }
}

// CHECK-LABEL: hw.module @if_condition_hosting(
// CHECK:      sv.initial  {
// CHECK-NEXT:   sv.if %a  {
// CHECK-NEXT:     sv.if %c  {
// CHECK-NEXT:       sv.if %b  {
// CHECK-NEXT:         sv.fwrite "A_B_C"
// CHECK-NEXT:         sv.if %d  {
// CHECK-NEXT:           sv.fwrite "A_B_C_D"
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       sv.if %d  {
// CHECK-NEXT:         sv.fwrite "A_C_D"
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     sv.if %d  {
// CHECK-NEXT:       sv.fwrite "A_D"
// CHECK-NEXT:       %0 = comb.and %b, %c : i1
// CHECK-NEXT:       sv.if %0  {
// CHECK-NEXT:         sv.fwrite "A_B_C_D again"
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
hw.module @if_condition_hosting(%a: i1, %b: i1, %c:i1, %d:i1) {
  %0 = comb.and %a, %b, %c, %d: i1
  %1 = comb.and %a, %b, %c: i1
  %2 = comb.and %a, %c, %d: i1
  %3 = comb.and %a, %d, %d: i1
  sv.initial {
    sv.if %0 {
      sv.fwrite "A_B_C_D"
    }
    sv.if %1 {
      sv.fwrite "A_B_C"
    }
    sv.if %2 {
      sv.fwrite "A_C_D"
    }
    sv.if %3 {
      sv.fwrite "A_D"
    }
    sv.if %0 {
      sv.fwrite "A_B_C_D again"
    }
  }

}