; RUN: circt-translate -import-firrtl -verify-diagnostics --split-input-file %s

; Invalid circuit should report an error

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: annotation has invalid circuit name */}}
circuit Foo: %[[{"one":null,"target":"~Fooo|Foo>bar"}]]
  module Foo:
    skip

; // -----

; Invalid circuit should report an error

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: annotation has invalid circuit name */}}
circuit Foo: %[[{"one":null,"target":"~Fooo"}]]
  module Foo:
    skip

; // -----

; Annotations JSON is not an array.

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: Expected annotations to be an array, but found something else. */}}
circuit Foo: %[{"a":"a"}]
  module Foo:
    skip

; // -----

; Annotations JSON is not an array of objects.

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: Expected annotations to be an array of objects, but found an array of something else. */}}
circuit Foo: %[[{"a":"a"},[{"b":"b"}]]]
  module Foo:
    skip

; // -----

; Annotations JSON is invalid. (There's a trailing comma.)

; expected-error @+2 {{Failed to parse JSON Annotations}}
; expected-note @+1 {{}}
circuit Foo: %[[{"a":"a"},]]]
  module Foo:
    skip

; // -----

; A custom annotation fails to parse because it is missing fields

; expected-error @+2 {{did not contain required key}}
; expected-note @+1 {{}}
circuit Foo: %[
[
  {
    "class": "sifive.enterprise.grandcentral.DataTapsAnnotation"
  }
]]
  module Foo:
    skip

; // -----

; An empty target should report an error (and not crash)


; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{"target": /* error: invalid target string */}}
circuit Foo: %[[{"target":""}]]
  module Foo:
    skip

; // -----

; A null target should report an error (and not crash)

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{"target": /* error: target must be a string type */}}
circuit Foo: %[[{"target":null}]]
  module Foo:
    skip

; // -----

; Invalid port reference should report errors
circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar[1]"}]]
  module Bar:
    input a: UInt<1>
  module Foo:
    ; expected-error @+1 {{the first token is invalid}}
    inst bar of Bar

; // -----

; Invalid sub-target annotation should report an error

circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar[a].baz"}, {"two":null,"target":"~Foo|Foo>bar[2].baz"}]]
  module Foo:
; expected-error @+2 {{expect an integer for the 1-th token of .a.baz}}
; expected-error @+1 {{the 1-th token of [2].baz is out of range in the vector}}
    wire bar: {baz: UInt<1>, qux: UInt<1>}[2]

; // -----

; Invalid sub-target annotation should report an error

circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar[1][0]"},{"two":null,"target":"~Foo|Foo>bar[1].qnx"}]]
  module Foo:
    input clock: Clock
; expected-error @+2 {{expect a string for the 2-th token of [1][0]}}
; expected-error @+1 {{the 2-th token of [1].qnx is not found in the bundle}}
    reg bar: {baz: UInt<1>, qux: UInt<1>}[2], clock

; // -----

; Invalid sub-target annotation should report an error

circuit Foo: %[[{"one":null,"target":"~Foo|Foo>bar[1].baz[0]"}]]
  module Foo:
    input clock: Clock
; expected-error @+1 {{the 3-th token of [1].baz[0] expects an aggregate type}}
    reg bar: {baz: UInt<1>, qux: UInt<1>}[2], clock

; // -----

; A target pointing at a non-existent module should error.

; expected-error @+1 {{unapplied annotations with target '~Foo|Bar' and payload '[{a}]'}}
circuit Foo: %[[{"a":null,"target":"~Foo|Bar"}]]
  module Foo:
    skip

; // -----

; A target pointing at a non-existent component should error.

; expected-error @+1 {{unapplied annotations with target '~Foo|Foo>x' and payload '[{a}]'}}
circuit Foo: %[[{"a":null,"target":"~Foo|Foo>x"}]]
  module Foo:
    skip

; // -----

; A target pointing at a non-existent instance should error.

; expected-error @+1 {{unapplied annotations with target '~Foo|Foo>baz' and payload '[{circt.nonlocal = @nla_1, class = "circt.nonlocal"}, {class = "firrtl.transforms.DontTouchAnnotation"}]'}}
circuit Foo: %[[{"a":null,"target":"~Foo|Foo/baz:Bar"}]]
  module Bar:
    skip
  module Foo:
    inst bar of Bar

; // -----
; This should work, but for now we are just checking that the annotation parses 
; into a path, not that the application is correct.  This is pending GC handling 
; nlas. 

; expected-error @+2 {{unapplied annotations with target '~NLAParse|A_companion' and payload '[{class = "sifive.enterprise.grandcentral.ViewAnnotation", id = 0 : i64, name = "A", type = "companion"}]'}}
; expected-error @+1 {{unapplied annotations with target '~NLAParse|FooBar>clock' and payload '[{class = "sifive.enterprise.grandcentral.AugmentedGroundType", id = 1 : i64, target = []}, {class = "firrtl.transforms.DontTouchAnnotation", target = []}]'}}
circuit NLAParse : %[[
  {
   "class": "sifive.enterprise.grandcentral.GrandCentralView$SerializedViewAnnotation", 
    "companion": "~NLAParse|A_companion", 
    "name": "A", 
    "parent": "~NLAParse|DUT", 
    "view": {
      "class": "sifive.enterprise.grandcentral.AugmentedBundleType", 
      "defName": "B", 
      "elements": [
        {
          "name": "C", 
          "tpe": {
            "class": "sifive.enterprise.grandcentral.AugmentedBundleType", 
            "defName": "D", 
            "elements": [
              {
                "name": "clock", 
                "tpe": {
                  "class": "sifive.enterprise.grandcentral.AugmentedGroundType", 
                  "ref": {
                    "circuit": "NLAParse", 
                    "component": [], 
                    "module": "NLAParse", 
                    "path": [
                      {
                        "_1": {
                          "value": "dut"
                        },
                        "_2": {
                          "value": "DUT"
                        }
                      },
                      {
                        "_1": {
                          "value": "foobar"
                        }, 
                        "_2": {
                          "value": "FooBar"
                        }
                      }
                    ], 
                    "ref": "clock"
                  }
                }
              }
            ]
          }
        }
      ]
    }
  }
]]
  
  module FooBar :

  module DUT :
    inst foobar of FooBar

  module NLAParse :
    inst dut of DUT
