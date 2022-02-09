hw.module @simple_comb(%a: i16, %b: i16, %c: i16) -> (y: i16, z: i16) {
    %0 = smt.add %a, %b : i16
    hw.output %0, %0 : i16, i16
}
