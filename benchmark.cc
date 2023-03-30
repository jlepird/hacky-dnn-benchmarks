
#include <string>

#include <catch2/catch_all.hpp>
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <fmt/core.h>
#include <onnxruntime_cxx_api.h>

namespace
{
    template <typename T>
    void benchmark_onnxrt(const std::string_view model_name, std::vector<float> &input_tensor_data, const std::vector<int64_t> &input_tensor_shape, std::vector<T> &output_tensor_data, const std::vector<int64_t> &output_tensor_shape)
    {
        const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        const auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_data.data(), input_tensor_data.size(), input_tensor_shape.data(), input_tensor_shape.size());
        auto output_tensor = Ort::Value::CreateTensor<T>(memory_info, output_tensor_data.data(), output_tensor_data.size(), output_tensor_shape.data(), output_tensor_shape.size());
        const char *input_names[] = {"input"};
        const char *output_names[] = {"output"};
        Ort::RunOptions run_options;
        Ort::Env env;

        const auto model_path_onnx = fmt::format("../models/{}.onnx", model_name);
        auto session = Ort::Session(env, model_path_onnx.c_str(), Ort::SessionOptions{nullptr});

        BENCHMARK("Onnxrt")
        {
            session.Run(run_options, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
        };
    };
}

TEST_CASE("Benchmarker")
{
    const auto model_name = GENERATE("tree", "LSTM", "transformer");
    CAPTURE(model_name);

    const auto batch_size = 100;
    const auto time_size = 10;

    int feature_size;
    int output_size;
    if (model_name == "LSTM")
    {
        feature_size = 10;
        output_size = 3;
    }
    else if (model_name == "transformer")
    {
        feature_size = 512;
        output_size = 512;
    }
    else if (model_name == "tree")
    {
        feature_size = 28;
        output_size = 2;
    }
    else
    {
        throw std::runtime_error(fmt::format("Unknown model {}", model_name));
    }

    if (model_name != "tree")
    {
        SECTION("TVM")
        {
            DLDevice dev{kDLCPU, 0};
            tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(fmt::format("../models/{}.so", model_name));
            tvm::runtime::Module gmod = mod_dylib.GetFunction("default")(dev);
            tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
            tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
            tvm::runtime::PackedFunc run = gmod.GetFunction("run");

            REQUIRE(run != nullptr);
            REQUIRE(set_input != nullptr);
            REQUIRE(get_output != nullptr);

            tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({batch_size, time_size, feature_size}, DLDataType{kDLFloat, 32, 1}, dev);
            set_input("x", x);
            tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({batch_size, time_size, output_size}, DLDataType{kDLFloat, 32, 1}, dev);

            BENCHMARK("TVM")
            {
                run(x, y);
            };
        }
    }

    SECTION("Torch")
    {
        torch::jit::script::Module module = torch::jit::load(fmt::format("../models/{}.pb", model_name));
        const auto get_torch_inputs = [feature_size, batch_size, time_size](std::string_view model_name)
        {
            if (model_name == "tree")
            {
                return torch::zeros({batch_size, feature_size});
            }
            return torch::zeros({batch_size, time_size, feature_size});
        };
        const auto inputs = get_torch_inputs(model_name);

        BENCHMARK("Torch")
        {
            return module.forward({inputs});
        };
    }

    SECTION("Onnxrt")
    {
        if (model_name == "tree")
        {
            auto input_tensor_data = std::vector<float>(batch_size * feature_size);
            const auto input_tensor_shape = std::vector<int64_t>{batch_size, feature_size};
            auto output_tensor_data = std::vector<int64_t>(batch_size);
            const auto output_tensor_shape = std::vector<int64_t>{batch_size};
            benchmark_onnxrt(model_name, input_tensor_data, input_tensor_shape, output_tensor_data, output_tensor_shape);
        }
        else
        {
            auto input_tensor_data = std::vector<float>(batch_size * time_size * feature_size);
            const auto input_tensor_shape = std::vector<int64_t>{batch_size, time_size, feature_size};
            auto output_tensor_data = std::vector<float>(batch_size * time_size * output_size);
            const auto output_tensor_shape = std::vector<int64_t>{batch_size, time_size, output_size};
            benchmark_onnxrt(model_name, input_tensor_data, input_tensor_shape, output_tensor_data, output_tensor_shape);
        }
    }
}