import torch
import torch.nn as nn
from hummingbird.ml import convert
import onnx2torch
import onnx
import numpy as np
from timeit import timeit
import onnxruntime
from tvm.driver import tvmc
from onnx_tf.backend import prepare
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import logging
import tarfile
import os
from torch.utils.mobile_optimizer import optimize_for_mobile
import zipfile

_BENCHMARK_FORMAT_STRING = "#### Model: {}\tEngine: {}:\tTime (s): {}"
_OUTPUT_DIR = Path("models/")


class SimpleLSTM(nn.Module):
    def __init__(self, input_feature_size=10, lstm_size=256, output_size=3) -> None:
        super().__init__()
        torch.random.manual_seed(1)

        self.lstm = nn.LSTM(
            input_size=input_feature_size, hidden_size=lstm_size, batch_first=True
        )
        self.dense = nn.Linear(lstm_size, output_size)
        nn.init.xavier_uniform(self.dense.weight)

    def forward(self, X):
        lstm_out, _ = self.lstm(X)
        return self.dense(lstm_out)


class SimpleTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.random.manual_seed(1)

        self.transformer = nn.Transformer()

    def forward(self, X):
        return self.transformer(X, X)


def export_lstm_model(hacky_benchmark=False):
    model = SimpleLSTM()

    input_shape = (100, 10, 10)
    model_name = "LSTM"

    export_dnn_torch(model, model_name, input_shape, hacky_benchmark)
    export_dnn_onnx(model, model_name, input_shape, hacky_benchmark)
    export_dnn_tvm(model, model_name, input_shape, hacky_benchmark)
    # export_dnn_tflite(model, model_name, input_shape, hacky_benchmark)


def export_transformer_model(hacky_benchmark=False):
    model = SimpleTransformer()
    input_shape = (100, 10, 512)
    model_name = "transformer"

    export_dnn_torch(model, model_name, input_shape, hacky_benchmark)
    export_dnn_onnx(model, model_name, input_shape, hacky_benchmark)
    # export_dnn_tflite(
    #     model, model_name, input_shape, hacky_benchmark
    # )  # crashes exit code 137, out of memory
    export_dnn_tvm(model, model_name, input_shape, hacky_benchmark)


def _save_tree_model(model, output_file):
    target_file = _OUTPUT_DIR / output_file
    if (target_file).exists():
        os.remove(target_file)

    model.save(str(target_file))

    with zipfile.ZipFile(str(target_file) + ".zip") as model_archive:
        deploy_model_name = "deploy_model" + target_file.suffix
        model_archive.extract(deploy_model_name, _OUTPUT_DIR)
        os.rename(_OUTPUT_DIR / deploy_model_name, _OUTPUT_DIR / output_file)


def export_tree_model(hacky_benchmark=False):
    # Create some random data for binary classification
    num_classes = 2
    num_features = 28
    X = np.random.rand(100000, num_features).astype(np.float32)
    y = np.random.randint(num_classes, size=100000)

    # Create and train a model (scikit-learn RandomForestClassifier in this case)
    skl_model = RandomForestClassifier(n_estimators=10, max_depth=10)
    skl_model.fit(X, y)

    model_torch = convert(skl_model, "torchscript", test_input=X)
    _save_tree_model(model_torch, "tree.pb")

    model_input_shape = (100, num_features)

    export_dnn_onnx(model_torch.model, "tree", model_input_shape, hacky_benchmark)
    # TVM infinite loops trying to compile the tree model :/


def export_dnn_torch(model_torch, model_name, input_shape, hacky_benchmark):
    model_torch_script = torch.jit.script(model_torch)

    optimized_model = optimize_for_mobile(model_torch_script)
    optimized_model.save(_OUTPUT_DIR / f"{model_name}.pb")
    if hacky_benchmark:
        with torch.no_grad():
            np.random.seed(1)
            sample_inputs = np.random.rand(*input_shape)

            def _benchmark():
                optimized_model(sample_inputs)

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
        output_names=["output"],
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
    model = tvmc.load(
        str(_OUTPUT_DIR / f"{model_name}.onnx"), shape_dict={"input": input_shape}
    )
    records = None
    records = tvmc.tune(
        model,
        target="llvm",
        enable_autoscheduler=True,
    )
    package = tvmc.compile(
        model,
        target="llvm",
        package_path=str(_OUTPUT_DIR / f"{model_name}.tar"),
        tuning_records=records,
    )

    with tarfile.open(str(_OUTPUT_DIR / f"{model_name}.tar")) as model_archive:
        model_archive.extract("mod.so", _OUTPUT_DIR)
        os.rename(_OUTPUT_DIR / "mod.so", _OUTPUT_DIR / f"{model_name}.so")

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
