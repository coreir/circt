# REQUIRES: bindings_python
# RUN: %PYTHON% %s 2> %t | FileCheck %s
# RUN: cat %t | FileCheck --check-prefix=ERR %s

import circt
from circt.dialects import hw, msft

import mlir.ir as ir
import mlir.passmanager
import sys

with ir.Context() as ctx, ir.Location.unknown():
  circt.register_dialects(ctx)
  i32 = ir.IntegerType.get_signless(32)
  i1 = ir.IntegerType.get_signless(1)

  m = ir.Module.create()
  with ir.InsertionPoint(m.body):
    extmod = msft.MSFTModuleExternOp(name='MyExternMod',
                                     input_ports=[],
                                     output_ports=[])

    entity_extern = msft.EntityExternOp.create("tag", "extra details")

    op = msft.MSFTModuleOp(name='MyWidget', input_ports=[], output_ports=[])
    with ir.InsertionPoint(op.add_entry_block()):
      msft.OutputOp([])

    top = msft.MSFTModuleOp(name='top', input_ports=[], output_ports=[])
    with ir.InsertionPoint(top.add_entry_block()):
      msft.OutputOp([])

    msft_mod = msft.MSFTModuleOp(name='msft_mod',
                                 input_ports=[],
                                 output_ports=[],
                                 parameters=ir.DictAttr.get(
                                     {"WIDTH": ir.IntegerAttr.get(i32, 8)}))
    with ir.InsertionPoint(msft_mod.add_entry_block()):
      msft.OutputOp([])

  with ir.InsertionPoint.at_block_terminator(op.body.blocks[0]):
    ext_inst = extmod.create("ext1")

  with ir.InsertionPoint.at_block_terminator(top.body.blocks[0]):
    path = op.create("inst1")
    minst = msft_mod.create("minst")

  # CHECK: #msft.physloc<M20K, 2, 6, 1, "foo_subpath">
  physAttr = msft.PhysLocationAttr.get(msft.M20K,
                                       x=2,
                                       y=6,
                                       num=1,
                                       sub_path="foo_subpath")
  print(physAttr)

  # CHECK: #msft.physloc<FF, 0, 0, 0>
  regAttr = msft.PhysLocationAttr.get(msft.FF, x=0, y=0, num=0)
  print(regAttr)

  path = ir.ArrayAttr.get([
      hw.InnerRefAttr.get(ir.StringAttr.get("top"), ir.StringAttr.get("inst1")),
      hw.InnerRefAttr.get(ir.StringAttr.get("MyWidget"),
                          ir.StringAttr.get("ext1"))
  ])
  print(path)
  # CHECK-NEXT: [#hw.innerNameRef<@top::@inst1>, #hw.innerNameRef<@MyWidget::@ext1>]

  resolved_inst = msft.get_instance(top.operation,
                                    ir.Attribute.parse("@inst1::@ext1"))
  assert (resolved_inst == ext_inst.operation)

  not_found_inst = msft.get_instance(top.operation,
                                     ir.Attribute.parse("@inst_none::@ext1"))
  assert (not_found_inst is None)

  # CHECK: msft.module @MyWidget {} ()
  # CHECK:   msft.output
  # CHECK: msft.module @msft_mod {WIDTH = 8 : i32} ()
  m.operation.print()

  db = msft.PlacementDB(top.operation)

  assert db.get_instance_at(physAttr) is None
  place_rc = db.add_placement(physAttr, path, "foo_subpath", resolved_inst)
  assert place_rc
  located_inst = db.get_instance_at(physAttr)
  assert located_inst is not None
  assert located_inst[0] == path
  assert located_inst[1] == "foo_subpath"
  assert located_inst[2] == resolved_inst

  place_rc = db.add_placement(physAttr, path, "foo_subpath", resolved_inst)
  assert not place_rc
  # ERR: error: 'msft.instance' op Could not apply placement #msft.physloc<M20K, 2, 6, 1, "foo_subpath">. Position already occupied by msft.instance @ext1 @MyExternMod

  physAttr2 = msft.PhysLocationAttr.get(msft.M20K, x=40, y=40, num=1)
  devdb = msft.PrimitiveDB()
  assert not devdb.is_valid_location(physAttr)
  devdb.add_primitive(physAttr)
  devdb.add_primitive(msft.PhysLocationAttr.get(msft.M20K, x=2, y=50, num=1))
  devdb.add_primitive(physAttr2)
  assert devdb.is_valid_location(physAttr)

  seeded_pdb = msft.PlacementDB(top.operation, devdb)

  print(seeded_pdb.get_nearest_free_in_column(msft.M20K, 2, 49))
  # CHECK: #msft.physloc<M20K, 2, 50, 1>
  print(seeded_pdb.get_nearest_free_in_column(msft.M20K, 2, 4))
  # CHECK: #msft.physloc<M20K, 2, 6, 1>

  rc = seeded_pdb.add_placement(physAttr, path, "foo_subpath", resolved_inst)
  assert rc
  external_path = ir.ArrayAttr.get(
      [ir.FlatSymbolRefAttr.get(entity_extern.sym_name.value)])
  rc = seeded_pdb.add_placement(physAttr2, external_path, "", entity_extern)
  assert rc
  with ir.InsertionPoint(m.body):
    global_ref = hw.GlobalRefOp(ir.StringAttr.get("foo"), path)
    global_ref.attributes["loc:foo_subpath"] = physAttr
    global_ref = hw.GlobalRefOp(ir.StringAttr.get("bar"), external_path)
    global_ref.attributes["loc"] = physAttr2

  nearest = seeded_pdb.get_nearest_free_in_column(msft.M20K, 2, 4)
  assert isinstance(nearest, msft.PhysLocationAttr)
  print(nearest)

  # CHECK: #msft.physloc<M20K, 2, 50, 1>


  def print_placement(loc, placement):
    assert isinstance(loc, msft.PhysLocationAttr)
    if placement:
      path = placement[0]
      print(f"{loc}, {path}")
    else:
      print(f"{loc}")

  print("=== Placements:")
  seeded_pdb.walk_placements(print_placement)
  # CHECK-LABEL: === Placements:
  # CHECK: #msft.physloc<M20K, 40, 40, 1>, [@tag]
  # CHECK: #msft.physloc<M20K, 2, 6, 1, "foo_subpath">, [#hw.innerNameRef<@top::@inst1>, #hw.innerNameRef<@MyWidget::@ext1>]
  # CHECK: #msft.physloc<M20K, 2, 50, 1>

  print("=== Placements (col 2):")
  seeded_pdb.walk_placements(print_placement, bounds=(2, 2, None, None))
  # CHECK-LABEL: === Placements (col 2):
  # CHECK: #msft.physloc<M20K, 2, 6, 1, "foo_subpath">, [#hw.innerNameRef<@top::@inst1>, #hw.innerNameRef<@MyWidget::@ext1>]
  # CHECK: #msft.physloc<M20K, 2, 50, 1>

  print("=== Placements (col 2, row > 10):")
  seeded_pdb.walk_placements(print_placement, bounds=(2, 2, 10, None))
  # CHECK-LABEL: === Placements (col 2, row > 10):
  # CHECK: #msft.physloc<M20K, 2, 50, 1>

  print("=== Placements (col 6):")
  seeded_pdb.walk_placements(print_placement, bounds=(6, 6, None, None))
  # CHECK-LABEL: === Placements (col 6):

  devdb = msft.PrimitiveDB()
  devdb.add_primitive(msft.PhysLocationAttr.get(msft.M20K, x=0, y=0, num=0))
  devdb.add_primitive(msft.PhysLocationAttr.get(msft.M20K, x=1, y=0, num=1))
  devdb.add_primitive(msft.PhysLocationAttr.get(msft.M20K, x=0, y=1, num=0))
  devdb.add_primitive(msft.PhysLocationAttr.get(msft.M20K, x=0, y=0, num=1))
  devdb.add_primitive(msft.PhysLocationAttr.get(msft.M20K, x=1, y=0, num=0))
  devdb.add_primitive(msft.PhysLocationAttr.get(msft.M20K, x=1, y=1, num=1))
  devdb.add_primitive(msft.PhysLocationAttr.get(msft.M20K, x=0, y=1, num=1))
  devdb.add_primitive(msft.PhysLocationAttr.get(msft.M20K, x=1, y=1, num=0))
  pdb = msft.PlacementDB(top.operation, devdb)

  print("=== Placements ASC, ASC:")
  walk_order = msft.WalkOrder(columns=msft.Direction.ASC,
                              rows=msft.Direction.ASC)
  pdb.walk_placements(print_placement, walk_order=walk_order)
  # CHECK-LABEL: === Placements ASC, ASC:
  # CHECK: #msft.physloc<M20K, 0, 0
  # CHECK: #msft.physloc<M20K, 0, 0
  # CHECK: #msft.physloc<M20K, 0, 1
  # CHECK: #msft.physloc<M20K, 0, 1
  # CHECK: #msft.physloc<M20K, 1, 0
  # CHECK: #msft.physloc<M20K, 1, 0
  # CHECK: #msft.physloc<M20K, 1, 1
  # CHECK: #msft.physloc<M20K, 1, 1

  print("=== Placements DESC, DESC:")
  walk_order = msft.WalkOrder(columns=msft.Direction.DESC,
                              rows=msft.Direction.DESC)
  pdb.walk_placements(print_placement, walk_order=walk_order)
  # CHECK-LABEL: === Placements DESC, DESC:
  # CHECK: #msft.physloc<M20K, 1, 1
  # CHECK: #msft.physloc<M20K, 1, 1
  # CHECK: #msft.physloc<M20K, 1, 0
  # CHECK: #msft.physloc<M20K, 1, 0
  # CHECK: #msft.physloc<M20K, 0, 1
  # CHECK: #msft.physloc<M20K, 0, 1
  # CHECK: #msft.physloc<M20K, 0, 0
  # CHECK: #msft.physloc<M20K, 0, 0

  print("=== Placements ASC, DESC:")
  walk_order = msft.WalkOrder(columns=msft.Direction.ASC,
                              rows=msft.Direction.DESC)
  pdb.walk_placements(print_placement, walk_order=walk_order)
  # CHECK-LABEL: === Placements ASC, DESC:
  # CHECK: #msft.physloc<M20K, 0, 1
  # CHECK: #msft.physloc<M20K, 0, 1
  # CHECK: #msft.physloc<M20K, 0, 0
  # CHECK: #msft.physloc<M20K, 0, 0
  # CHECK: #msft.physloc<M20K, 1, 1
  # CHECK: #msft.physloc<M20K, 1, 1
  # CHECK: #msft.physloc<M20K, 1, 0
  # CHECK: #msft.physloc<M20K, 1, 0

  print("=== Placements None, Asc:")
  walk_order = msft.WalkOrder(rows=msft.Direction.DESC)
  pdb.walk_placements(print_placement, walk_order=walk_order)
  # CHECK-LABEL: === Placements None, Asc:
  # CHECK: #msft.physloc<M20K, {{.+}}, 1
  # CHECK: #msft.physloc<M20K, {{.+}}, 1
  # CHECK: #msft.physloc<M20K, {{.+}}, 0
  # CHECK: #msft.physloc<M20K, {{.+}}, 0
  # CHECK: #msft.physloc<M20K, {{.+}}, 1
  # CHECK: #msft.physloc<M20K, {{.+}}, 1
  # CHECK: #msft.physloc<M20K, {{.+}}, 0
  # CHECK: #msft.physloc<M20K, {{.+}}, 0

  print("=== Mutations:")
  old_location = msft.PhysLocationAttr.get(msft.M20K, x=0, y=0, num=0)
  new_location = msft.PhysLocationAttr.get(msft.M20K, x=1, y=1, num=1)

  pdb.add_placement(old_location, path, "", resolved_inst)
  assert pdb.get_instance_at(old_location)[2] == resolved_inst
  rc = pdb.remove_placement(new_location)
  assert rc == False
  rc = pdb.remove_placement(old_location)
  assert rc == True
  assert pdb.get_instance_at(old_location) is None

  rc = pdb.move_placement(old_location, new_location)
  assert rc == False
  pdb.add_placement(old_location, path, "", resolved_inst)
  pdb.add_placement(new_location, path, "", resolved_inst)
  rc = pdb.move_placement(old_location, new_location)
  assert rc == False
  pdb.remove_placement(new_location)
  rc = pdb.move_placement(old_location, new_location)
  assert rc == True
  assert pdb.get_instance_at(old_location) is None
  assert pdb.get_instance_at(new_location)[2] == resolved_inst

  print("=== Errors:", file=sys.stderr)
  # TODO: Python's sys.stderr doesn't seem to be shared with C++ errors.
  # See https://github.com/llvm/circt/issues/1983 for more info.
  sys.stderr.flush()
  # ERR-LABEL: === Errors:
  bad_loc = msft.PhysLocationAttr.get(msft.M20K, x=7, y=99, num=1)
  rc = seeded_pdb.add_placement(bad_loc, path, "foo_subpath", resolved_inst)
  assert not rc
  # ERR: error: 'msft.instance' op Could not apply placement. Invalid location

  # CHECK-LABEL: === tcl ===
  print("=== tcl ===")

  # CHECK: proc top_config { parent } {
  # CHECK:   set_location_assignment M20K_X2_Y6_N1 -to $parent|inst1|ext1|foo_subpath
  pm = mlir.passmanager.PassManager.parse("lower-msft-to-hw{tops=top}")
  pm.run(m)
  circt.export_verilog(m, sys.stdout)
