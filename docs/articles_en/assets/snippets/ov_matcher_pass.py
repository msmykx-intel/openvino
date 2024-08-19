// ! [matcher_pass:ov_matcher_pass_py]
'''
``MatcherPass`` is used for pattern-based transformations.
To create transformation you need:

1. Create a pattern
2. Implement a callback
3. Register the pattern and Matcher

In the next example we define transformation that searches for ``Relu`` layer and inserts after it another
``Relu`` layer.
'''

from openvino.runtime.passes import MatcherPass

class PatternReplacement(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        relu = WrapType("opset13::Relu")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            new_relu = ops.relu(root.input(0).get_source_output())

            """Use new operation for additional matching
              self.register_new_node(new_relu)

              Input->Relu->Result => Input->Relu->Relu->Result
            """
            root.input(0).replace_source_output(new_relu.output(0))
            return True

        self.register_matcher(Matcher(relu, "PatternReplacement"), callback)


'''
After running this code you will see the next:

model ops :
parameter
result
relu

model ops :
parameter
result
relu
new_relu

In oder to run this script you need to export PYTHONPATH as the path to binary OpenVINO python models.
'''
from openvino.runtime.passes import Manager, GraphRewrite, BackwardGraphRewrite, Serialize
from openvino import Model, PartialShape
from openvino.runtime import opset13 as ops
from openvino.runtime.passes import ModelPass, Matcher, MatcherPass, WrapType

class PatternReplacement(MatcherPass):
    def __init__(self):
        MatcherPass.__init__(self)
        relu = WrapType("opset13::Relu")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            new_relu = ops.relu(root.input(0).get_source_output())
            new_relu.set_friendly_name('new_relu')

            """Use new operation for additional matching
              self.register_new_node(new_relu)

              Input->Relu->Result => Input->Relu->Relu->Result
            """
            root.input(0).replace_source_output(new_relu.output(0))
            return True

        self.register_matcher(Matcher(relu, "PatternReplacement"), callback)


def get_relu_model():
    # Parameter->Relu->Result
    param = ops.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = ops.relu(param.output(0))
    relu.set_friendly_name('relu')
    res = ops.result(relu.output(0), name="result")
    return Model([res], [param], "test")


def print_model_ops(model):
    print('model ops : ')
    for op in model.get_ops():
        print(op.get_friendly_name())
    print('')


manager = Manager()
manager.register_pass(PatternReplacement())


model = get_relu_model()
print_model_ops(model)
manager.run_passes(model)
print_model_ops(model)

// ! [matcher_pass:ov_matcher_pass_py]