import json
from tensorflow.python.framework import test_util
from tensorflow.python.keras import losses
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.utils import losses_utils, generic_utils
from tensorflow.python.platform import test
from tensorflow.python.util import serialization


class SerializationTests(test.TestCase):
    @test_util.run_in_graph_and_eager_modes
    def test_serialize_custom_model_compile(self):
        with generic_utils.custom_object_scope():
            @generic_utils.register_keras_serializable(package='dummy-package')
            class DummySparseCategoricalCrossentropyLoss(losses.LossFunctionWrapper):
                def __init__(
                        self,
                        from_logits=False,
                        reduction=losses_utils.ReductionV2.AUTO,
                        name="dummy_sparse_categorical_crossentropy_loss",
                ):
                    super(DummySparseCategoricalCrossentropyLoss, self).__init__(
                        losses.sparse_categorical_crossentropy,
                        name=name,
                        reduction=reduction,
                        from_logits=from_logits,
                    )

            x = input_layer.Input(shape=[3])
            y = core.Dense(10)(x)
            model = training.Model(x, y)
            model.compile(
                loss=DummySparseCategoricalCrossentropyLoss(from_logits=True)
            )
            model_round_trip = json.loads(
                json.dumps(model.loss, default=serialization.get_json_type)
            )

            self.assertEqual("dummy-package>DummySparseCategoricalCrossentropyLoss", model_round_trip["class_name"])


fx = SerializationTests()
fx.test_serialize_custom_model_compile()
