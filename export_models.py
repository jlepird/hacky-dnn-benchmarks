import torch
import torch.nn as nn
from hummingbird.ml import convert
import onnx
import numpy as np
from timeit import timeit
import onnxruntime
from tvm.driver import tvmc
from onnx_tf.backend import prepare
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


def get_random_forest_model_onnx():
    pass


_BENCHMARK_FORMAT_STRING = "#### Model: {}\tEngine: {}:\tTime (s): {}"
_OUTPUT_DIR = Path("/tmp/")


class SimpleLSTM(nn.Module):
    def __init__(self, input_feature_size=10, lstm_size=256, output_size=3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_feature_size, hidden_size=lstm_size, batch_first=True
        )
        self.dense = nn.Linear(lstm_size, output_size)

    def forward(self, X):
        lstm_out, _ = self.lstm(X)
        return self.dense(lstm_out)


def export_lstm_model(hacky_benchmark=False):
    model = SimpleLSTM()

    input_shape = (100, 10, 10)
    model_name = "LSTM"

    export_dnn_torch(model, model_name, input_shape, hacky_benchmark)
    export_dnn_onnx(model, model_name, input_shape, hacky_benchmark)
    export_dnn_tvm(model, model_name, input_shape, hacky_benchmark)
    export_dnn_tflite(model, model_name, input_shape, hacky_benchmark)


def export_transformer_model(hacky_benchmark=False):
    model = torch.hub.load(
        "huggingface/pytorch-transformers", "model", "bert-base-uncased"
    )
    input_shape = (100, 10)
    model_name = "bert"
    export_dnn_torch(model, model_name, input_shape, hacky_benchmark)
    export_dnn_onnx(model, model_name, input_shape, hacky_benchmark)
    # export_dnn_tflite(model, model_name, input_shape, hacky_benchmark) # crashes
    export_dnn_tvm(model, model_name, input_shape, hacky_benchmark)


def export_tree_model(hacky_benchmark=False):
    # Create some random data for binary classification
    num_classes = 2
    X = np.random.rand(100000, 28)
    y = np.random.randint(num_classes, size=100000)

    # Create and train a model (scikit-learn RandomForestClassifier in this case)
    skl_model = RandomForestClassifier(n_estimators=10, max_depth=10)
    skl_model.fit(X, y)

    model_torch = convert(skl_model, "pytorch")
    model_torch.save(str(_OUTPUT_DIR / "tree.pb"))

    model_onnx = convert(skl_model, "onnx", test_input=X)
    model_onnx.save(str(_OUTPUT_DIR / "tree.onnx"))

    model_tvm = convert(skl_model, "tvm", test_input=X)
    model_tvm.save(str(_OUTPUT_DIR / "tree.so"))

    if hacky_benchmark:

        def _benchmark_torch():
            model_torch.predict(X)

        def _benchmark_onnx():
            model_onnx.predict(X)

        def _benchmark_tvm():
            model_tvm.predict(X)

        for engine, benchmark_fn in {
            "torch": "_benchmark_torch",
            "onnx": "_benchmark_onnx",
            "tvm": "_benchmark_tvm",
        }.items():
            benchmark_s = timeit(benchmark_fn, globals=locals())
            print(_BENCHMARK_FORMAT_STRING.format("Tree", engine, benchmark_s))

    # export_dnn_tflite(None, "tree", input_shape=(1, 28), hacky_benchmark=hacky_benchmark)


def export_dnn_torch(model_torch, model_name, input_shape, hacky_benchmark):
    torch.save(model_torch, _OUTPUT_DIR / f"{model_name}.pt")
    if hacky_benchmark:
        with torch.no_grad():
            np.random.seed(1)
            sample_inputs = np.random.rand(*input_shape)

            def _benchmark():
                model_torch(sample_inputs)

            benchmark_s = timeit("_benchmark", globals=locals())
            print(_BENCHMARK_FORMAT_STRING.format(model_name, "Torch", benchmark_s))


def export_dnn_onnx(model_torch, model_name, input_shape, hacky_benchmark):
    filename = str(_OUTPUT_DIR / f"{model_name}.onnx")

    torch.onnx.export(
        model_torch,
        torch.ones(
            *input_shape, dtype=torch.int if model_name == "bert" else torch.float32
        ),
        filename,
        export_params=True,
        input_names=["input"],
        opset_version=12,
    )

    if hacky_benchmark:
        ort_session = onnxruntime.InferenceSession(filename)

        np.random.seed(1)
        ort_inputs = {ort_session.get_inputs()[0].name: np.random.rand(*input_shape)}

        def _benchmark():
            ort_session.run(None, ort_inputs)

        benchmark_s = timeit("_benchmark", globals=locals())
        print(_BENCHMARK_FORMAT_STRING.format(model_name, "ONNX", benchmark_s))


def export_dnn_tvm(model_torch, model_name, input_shape, hacky_benchmark):
    model = tvmc.load(str(_OUTPUT_DIR / f"{model_name}.onnx"))
    # records = tvmc.tune(model, target="llvm", enable_autoscheduler=False, early_stopping=3)
    package = tvmc.compile(
        model,
        target="llvm",
        package_path=f"/tmp/{model_name}.so",
        tuning_records=None,
    )
    if hacky_benchmark:
        np.random.seed(1)
        sample_inputs = {"input": np.random.rand(*input_shape)}

        def _benchmark():
            tvmc.run(package, device="cpu", inputs=sample_inputs)

        benchmark_s = timeit("_benchmark", globals=locals())
        print(_BENCHMARK_FORMAT_STRING.format(model_name, "TVM", benchmark_s))


def export_dnn_tflite(model_torch, model_name, input_shape, hacky_benchmark):
    onnx_model = onnx.load(str(_OUTPUT_DIR / f"{model_name}.onnx"))
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(f"/tmp/{model_name}_tf")
    converter = tf.lite.TFLiteConverter.from_saved_model(f"/tmp/{model_name}_tf")
    tflite_model = converter.convert()

    if hacky_benchmark:
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()  # Needed before execution!

        input = interpreter.get_input_details()[0]
        np.random.seed(1)
        interpreter.set_tensor(input["index"], np.float32(np.random.rand(*input_shape)))

        def _benchmark():
            interpreter.invoke()

        benchmark_s = timeit("_benchmark", globals=locals())
        print(_BENCHMARK_FORMAT_STRING.format(model_name, "TFLite", benchmark_s))


def main():
    hacky_benchmark = True
    export_lstm_model(hacky_benchmark)
    export_transformer_model(hacky_benchmark)
    export_tree_model(hacky_benchmark)


if __name__ == "__main__":
    main()
