// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pvpnet.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include <iomanip>
#include <fstream>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/utils/run_python.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include <tensorflow/core/util/tensor_bundle/tensor_bundle.h>
#include "utils/tensor_view.h"
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <filesystem>



namespace open_spiel::algorithms::mpg
{

    namespace tf = tensorflow;
    using Tensor = Eigen::Tensor<float, 2, Eigen::RowMajor>;
    using TensorMap = Eigen::TensorMap<Tensor, Eigen::Aligned>;
    using TensorBool = Eigen::Tensor<bool, 2, Eigen::RowMajor>;
    using TensorMapBool = Eigen::TensorMap<TensorBool, Eigen::Aligned>;

    bool CreateGraphDefMPG(const Game& game, double learning_rate,
        double weight_decay, const std::string& path, const std::string& filename,
        std::string nn_model, int nn_width, int nn_depth, bool verbose)
    {
      return RunPython("open_spiel.python.algorithms.alpha_zero_mpg.export_model",
                       {
                           "--game", absl::StrCat("'", game.ToString(), "'"),  //
                           "--path", absl::StrCat("'", path, "'"),             //
                           "--graph_def", filename,                            //
                           "--learning_rate", absl::StrCat(learning_rate),     //
                           "--weight_decay", absl::StrCat(weight_decay),       //
                           "--nn_model", std::move(nn_model),                             //
                           "--nn_depth", absl::StrCat(nn_depth),               //
                           "--nn_width", absl::StrCat(nn_width),               //
                           absl::StrCat("--verbose=", verbose ? "true" : "false"),
                       });
    }

    std::vector<std::int64_t> AddBatchDim(const std::vector<int> &shape, int batch_dim) {
        std::vector<std::int64_t> new_shape = {batch_dim};
        std::copy(shape.begin(), shape.end(), std::back_inserter(new_shape));
        return new_shape;
    }

    std::vector<std::int64_t> AddBatchDim(const std::vector<std::int64_t> &shape, int batch_dim) {
        std::vector<std::int64_t> new_shape = {batch_dim};
        std::copy(shape.begin(), shape.end(), std::back_inserter(new_shape));
        return new_shape;
    }

    TensorViewConst<3> EnvironmentViewConst(const std::vector<float> &environment_flat,const std::vector<int> & shape)
    {
        std::array<int, 3> environment_shape_array;
        std::copy_n(shape.begin(), 3, environment_shape_array.begin());
        absl::Span environment_span(environment_flat.data(),environment_flat.size());
        return {environment_span, environment_shape_array};
    }

    void PVPNetModel::LoadSavedModel(const std::string &path, const std::string &file_name, const std::string &device)
    {
        LoadSavedModel(std::filesystem::path(path) / file_name, device);
    }

    PVPNetModel::PVPNetModel(const Game& game, const std::string& path,
                       const std::string& file_name, const open_spiel::algorithms::mpg::Device& device)
    : PVPNetModel(game,std::filesystem::path(path) / file_name, device)
    {

        // Game must have a nested observation tensor shape
        if(game.ObservationTensorShapeSpecs() != TensorShapeSpecs::kNestedList)
            SpielFatalError("ObservationTensorShapeSpecs must be kNestedList for PPVPNetModel");
        auto nested_shape= game.ObservationTensorsShapeList();
        // The firs dimension is the maximum graph size
        max_size_ = nested_shape.at(0).at(0);
        // Check that the nested shape is of the form [(n,n,?), 1] or [(n,n,?),()] where n is the maximum graph size
        SPIEL_CHECK_EQ(nested_shape.size(), 2);
        SPIEL_CHECK_EQ(nested_shape.at(0).size(), 3);
        SPIEL_CHECK_LE(nested_shape.at(1).size(), 1);
        // Extract the environment and state shapes
        environment_shape_ = nested_shape.at(0);
        state_shape_ = nested_shape.at(1);



        // Some assumptions that we can remove eventually. The value net returns
      // a single value in terms of player 0 and the game is assumed to be zero-sum,
      // so player 1 can just be -value.
      SPIEL_CHECK_EQ(game.NumPlayers(), 2);
      SPIEL_CHECK_EQ(game.GetType().utility, GameType::Utility::kZeroSum);

      LoadSavedModel(path, file_name, device.device_name());
    }

    PVPNetModel::PVPNetModel(const Game& game, const std::string &path, const open_spiel::algorithms::mpg::Device& device)
            : device_(device.device_name()),
              num_actions_(game.NumDistinctActions())
    {

        // Game must have a nested observation tensor shape
        if(game.ObservationTensorShapeSpecs() != TensorShapeSpecs::kNestedList)
            SpielFatalError("ObservationTensorShapeSpecs must be kNestedList for PPVPNetModel");
        auto nested_shape= game.ObservationTensorsShapeList();
        // The firs dimension is the maximum graph size
        max_size_ = nested_shape.at(0).at(0);
        // Check that the nested shape is of the form [(n,n,?), 1] or [(n,n,?),()] where n is the maximum graph size
        SPIEL_CHECK_EQ(nested_shape.size(), 2);
        SPIEL_CHECK_EQ(nested_shape.at(0).size(), 3);
        SPIEL_CHECK_LE(nested_shape.at(1).size(), 1);
        // Extract the environment and state shapes
        environment_shape_ = nested_shape.at(0);
        state_shape_ = nested_shape.at(1);



        // Some assumptions that we can remove eventually. The value net returns
        // a single value in terms of player 0 and the game is assumed to be zero-sum,
        // so player 1 can just be -value.
        SPIEL_CHECK_EQ(game.NumPlayers(), 2);
        SPIEL_CHECK_EQ(game.GetType().utility, GameType::Utility::kZeroSum);

        LoadSavedModel(path, device_);
    }

    std::string PVPNetModel::SaveCheckpoint(int step)
    {
      std::string full_path = absl::StrCat(path_, "/checkpoint-", step);
      tensorflow::Tensor checkpoint_path(tf::DT_STRING, tf::TensorShape());
      checkpoint_path.scalar<tensorflow::tstring>()() = full_path;
      TF_CHECK_OK(tf_session_->Run(
          {{meta_graph_def_->saver_def().filename_tensor_name(), checkpoint_path}},
          {}, {meta_graph_def_->saver_def().save_tensor_name()}, nullptr));
      // Writing a checkpoint from python writes the metagraph file, but c++
      // doesn't, so do it manually to make loading checkpoints easier.
      file::File(absl::StrCat(full_path, ".meta"), "w").Write(
          model_meta_graph_contents_);
      return full_path;
    }

void PVPNetModel::LoadCheckpoint(const std::string& path) {
  /*tf::Tensor checkpoint_path(tf::DT_STRING, tf::TensorShape());
  checkpoint_path.scalar<tensorflow::tstring>()() = path;
  TF_CHECK_OK(tf_session_->Run(
      {{meta_graph_def_->saver_def().filename_tensor_name(), checkpoint_path}},
      {}, {meta_graph_def_->saver_def().restore_op_name()}, nullptr));
    status = session->Run({}, {}, {"save/restore_all"}, nullptr);*/
     const auto fileTensorName = meta_graph_def_->saver_def().filename_tensor_name();
    const auto restoreOpName = meta_graph_def_->saver_def().restore_op_name();
    auto checkpointPathTensor = tf::Tensor(tf::DT_STRING, tf::TensorShape());
    std::cout << fileTensorName << std::endl;
    std::cout << restoreOpName << std::endl;
    checkpointPathTensor.scalar<tensorflow::tstring>()() = path;

    auto status = tf_session_->Run(
            { { fileTensorName, checkpointPathTensor }, },
            {},
            { restoreOpName },
            nullptr
    );

    // Ignore NOT_FOUND error for the missing variable
    if (!status.ok() && status.code() == tf::error::Code::NOT_FOUND) {
        std::cerr << "Warning: Ignoring NOT_FOUND error for the missing variable." << std::endl;
        std::cerr << "Warning Message: " << status.error_message() << std::endl;
        status = tf::OkStatus();
    }

    if (!status.ok())
        SpielFatalError(absl::StrCat("Error loading checkpoint: ", status.error_message()));
}

std::vector<PVPNetModel::InferenceOutputs> PVPNetModel::Inference(const std::vector<InferenceInputs>& inputs)
{
    return Inference(inputs,environment_shape_);
}

std::vector<PVPNetModel::InferenceOutputs> PVPNetModel::Inference(
    const std::vector<InferenceInputs>& inputs,const std::vector<int>& environment_shape) {
  int inference_batch_size = inputs.size();

  // The environment tensor
  tf::Tensor tf_environment_inputs(
      tf::DT_FLOAT, tf::TensorShape(AddBatchDim(environment_shape, inference_batch_size)));
  auto environment = tf_environment_inputs.tensor<float,4>();

  // The state tensor
  tf::Tensor tf_state_inputs(
      tf::DT_FLOAT, tf::TensorShape(AddBatchDim(state_shape_, inference_batch_size)));
  auto state=tf_state_inputs.matrix<float>();


  // Copy the inputs into the tensors
  for (int b = 0; b < inference_batch_size; ++b)
  {
      auto environment_view= EnvironmentViewConst(inputs[b].environment,environment_shape);
      for(int i=0; i < environment_view.shape(0); i++)
      {
          for (int j = 0; j < environment_view.shape(1); j++) for (int k = 0; k < environment_view.shape(2); k++)
            environment(b, i, j, k) = environment_view[{i, j, k}];
      }
      state(b,0) = inputs[b].state;
  }

    // Create the input specification
    InputSpecification input_specification;
    input_specification.emplace_back(input_name_map_["environment"], tf_environment_inputs);
    input_specification.emplace_back(input_name_map_["state"], tf_state_inputs);
//    input_specification.emplace_back("training", tensorflow::Tensor(false));
    // Create the output specification
    OutputSpecification output_specification;
    output_specification.emplace_back(output_name_map_["value_targets"]);
    output_specification.emplace_back(output_name_map_["policy_targets"]);

  // Run the inference
  std::vector<tensorflow::Tensor> tf_outputs;
  TF_CHECK_OK(tf_session_->Run(input_specification,
      output_specification, {}, &tf_outputs));

  // Extract the outputs
  TensorMap policy_matrix = tf_outputs[1].matrix<float>();
  TensorMap value_matrix = tf_outputs[0].matrix<float>();

  // Convert the outputs into the correct format
  std::vector<InferenceOutputs> out;
  out.reserve(inference_batch_size);
  for (int b = 0; b < inference_batch_size; ++b)
  {
    double value = value_matrix(b, 0);
    ActionsAndProbs state_policy;

    auto environment_view=EnvironmentViewConst(inputs[b].environment,environment_shape_);
    std::vector<int> legal_actions;
    //Extracts legal actions
    for(int i=0;i<num_actions_;i++) if(environment_view[{inputs[b].state,i,EnvironmentAxis::kAdjacencyAxis}] > 0.5)
        legal_actions.push_back(i);

    state_policy.reserve(legal_actions.size());
    for (Action action : legal_actions)
      state_policy.emplace_back(action, policy_matrix(b, action));

    out.push_back({value, state_policy});
  }
 // Return the predictions for the whole batch
  return out;
}

void PVPNetModel::LoadSavedModel(const std::string &model_path, const std::string &device)
{

    FreeSession();

    model_meta_graph_contents_ = file::ReadContentsFromFile(model_path, "r");
    //TF_CHECK_OK(
    //    ReadBinaryProto(tf::Env::Default(), model_path, &meta_graph_def_));


    // Loads the model and creates a session.
    model_bundle_= std::make_unique<tf::SavedModelBundle>();
    TF_CHECK_OK(tensorflow::LoadSavedModel(
            session_options_,
            run_options,
            model_path,
            {"serve"},
            model_bundle_.get()));

    // Get the meta graph definition
    meta_graph_def_ = &model_bundle_->meta_graph_def;
    // Get the signature of the model
    auto signatures = model_bundle_->GetSignatures();
    // Check that the model has the serving_default signature
    SPIEL_CHECK_TRUE(signatures.contains(kSignatureName));
    auto [input,input_mapper,output,output_mapper] = ExtractInputOutputNames(*model_bundle_, kSignatureName);
    input_names_ = std::move(input);
    output_names_ = std::move(output);
    input_name_map_ = std::move(input_mapper);
    output_name_map_ = std::move(output_mapper);
    std::set<std::string> expected_input_names = {"environment","state"};
    std::set<std::string> expected_output_names = {"value_targets","policy_targets"};
    if(input_names_ != expected_input_names)
        SpielFatalError(absl::StrCat("Input names do not match expected names. Got: ", absl::StrJoin(input_names_, ",")));
    if(output_names_ != expected_output_names)
        SpielFatalError(absl::StrCat("Output names do not match expected names. Got: ", absl::StrJoin(output_names_, ",")));

    //tf::graph::SetDefaultDevice(device, meta_graph_def_->mutable_graph_def());


    // Point the session to the graph we just loaded
    tf_session_ = model_bundle_->GetSession();

    // Load graph into session
    //TF_CHECK_OK(tf_session_->Create(meta_graph_def_->graph_def()));

    // Initialize our variables
    //TF_CHECK_OK(tf_session_->Run({}, {}, {"init_all_vars_op"}, nullptr));
}


    void PVPNetModel::Load(const std::string& path, bool checkpoint) {
  if(checkpoint)
    LoadCheckpoint(path);
  else
  {
      std::filesystem::path model_dir(path);
      LoadSavedModel(path, device_);
  }
}

    void PVPNetModel::FreeSession()
    {
        if (tf_session_ != nullptr)
            TF_CHECK_OK(tf_session_->Close());
        tf_session_ = nullptr;
    }

    PVPNetModel::~PVPNetModel()
    {
        FreeSession();
    }

PVPNetModel::LossInfo PVPNetModel::Learn(const std::vector<TrainInputs>& inputs) {
    throw std::runtime_error("Training is not supported yet");
  int training_batch_size = inputs.size();

  tensorflow::Tensor tf_environment_inputs(
      tf::DT_FLOAT, tf::TensorShape(AddBatchDim(environment_shape_,training_batch_size)));
    tensorflow::Tensor tf_state_inputs(
            tf::DT_FLOAT, tf::TensorShape(AddBatchDim(state_shape_,training_batch_size)));
  tensorflow::Tensor tf_policy_targets(
      tf::DT_FLOAT, tf::TensorShape({training_batch_size,num_actions_}));
  tensorflow::Tensor tf_value_targets(
      tf::DT_FLOAT, tf::TensorShape({training_batch_size, 1}));

  // Fill the inputs and mask
  auto environment_inputs = tf_environment_inputs.tensor<float,4>();
  auto state_inputs=tf_state_inputs.matrix<float>();
  TensorMap policy_targets = tf_policy_targets.matrix<float>();
  TensorMap value_targets = tf_value_targets.matrix<float>();

  for (int b = 0; b < training_batch_size; ++b) {
    // Zero initialize the sparse inputs.
    for (int a = 0; a < num_actions_; ++a)
      policy_targets(b, a) = 0;



    auto environment_view= EnvironmentViewConst(inputs[b].environment,environment_shape_);
    for(int i=0; i < environment_view.shape(0); i++)
    {
        for (int j = 0; j < environment_view.shape(1); j++) for (int k = 0; k < environment_view.shape(2); k++)
            environment_inputs(b, i, j, k) = environment_view[{i, j, k}];
        state_inputs(b,0) = inputs[b].state;
    }


    for (const auto& [action, prob] : inputs[b].policy)
      policy_targets(b, action) = prob;
    value_targets(b, 0) = inputs[b].value;
  }
  InputSpecification input_specification;
    input_specification.emplace_back(input_name_map_["environment"], tf_environment_inputs);
    input_specification.emplace_back(input_name_map_["state"], tf_state_inputs);
  // Run a training step and get the losses.
  std::vector<tensorflow::Tensor> tf_outputs;
  TF_CHECK_OK(tf_session_->Run({{"environment", tf_environment_inputs},
                                {"state", tf_state_inputs},
                                {"policy_targets", tf_policy_targets},
                                {"value_targets", tf_value_targets},
                                {"training", tensorflow::Tensor(true)}},
                               {"policy_loss", "value_loss", "l2_loss"},
                               {"train"}, &tf_outputs));

  return LossInfo(
      tf_outputs[0].scalar<float>()(0),
      tf_outputs[1].scalar<float>()(0),
      tf_outputs[2].scalar<float>()(0));
}

    InputOutputNamesType ExtractInputOutputNames(const tensorflow::SavedModelBundle &bundle,const std::string& signature_name)
    {
        InputOutputNamesType names;
        auto signature = bundle.GetSignatures();
        auto it = signature.find(signature_name);
        if (it == signature.end())
            SpielFatalError(absl::StrCat("Signature ", signature_name, " not found in the model."));
        else
        {
            for(const auto & input : signature.at(signature_name).inputs())
            {
                names.input_names.emplace(input.first);
                names.input_name_mapper[input.first]=input.second.name();
            }
            for(const auto & output : signature.at(signature_name).outputs())
            {
                names.output_names.emplace(output.first);
                names.output_name_mapper[output.first]=output.second.name();
            }

            return names;
        }

    }

    PVPNetModel::InferenceInputs PVPNetModel::InferenceInputs::Extract(std::vector<float> data)
    {
        InferenceInputs inputs;
        inputs.state = data.back();
        inputs.environment = std::move(data);
        inputs.environment.pop_back();
        return inputs;
    }

    Device::Device(DeviceType device_type, int device_id)
    {
        if(device_type == DeviceType::CPU)
            device_name_ = "/cpu:" + std::to_string(device_id);
        else if(device_type == DeviceType::GPU)
            device_name_ = "/gpu:" + std::to_string(device_id);
    }
    Device::Device(const std::string& device_name):device_name_(device_name)
    {

    }
    Device Device::CPU(int device_id)
    {
        return Device(DeviceType::CPU,device_id);
    }
    Device Device::GPU(int device_id)
    {
        return Device(DeviceType::GPU,device_id);
    }

    std::string Device::device_name() const
    {
        return device_name_;
    }
}  // namespace open_spiel
