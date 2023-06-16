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
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/file.h"
#include "open_spiel/abseil-cpp/absl/flags/parse.h"
#include "open_spiel/abseil-cpp/absl/flags/flag.h"
#include "open_spiel/games/mpg/mpg.h"
#include <fstream>
#include <filesystem>

namespace open_spiel::algorithms::mpg {
    namespace {
        std::string test_game="mpg(max_moves=20,max_size=20,generator=gnp,generator_params=20 0.5 -1 1)";
        double SolveState( const State& state, absl::flat_hash_map<std::string, int>& cache, std::vector<PVPNetModel::TrainInputs>& train_inputs)
        {
          std::string state_str = state.ToString();
          if (cache.find(state_str) != cache.end()) {
            return train_inputs[cache[state_str]].value;
          }
          if (state.IsTerminal()) {
            return state.PlayerReturn(0);
          }

          bool max_player = state.CurrentPlayer() == 0;
          std::vector<float> environmnet = state.ObservationTensor();
          int state_ = environmnet.back();
            environmnet.pop_back();
          std::vector<Action> legal_actions = state.LegalActions();

          Action best_action = kInvalidAction;
          double best_value = -2;
          for (Action action : legal_actions) {
            double value = SolveState(*state.Child(action), cache, train_inputs);
            if (best_action == kInvalidAction ||
                (max_player ? value > best_value : value < best_value)) {
              best_action = action;
              best_value = value;
            }
          }
          ActionsAndProbs policy({{best_action, 1}});

          cache[state_str] = train_inputs.size();
          train_inputs.push_back(PVPNetModel::TrainInputs{
            environmnet, state_, policy, best_value});
          return best_value;
        }

std::vector<PVPNetModel::TrainInputs> SolveGame() {
  std::shared_ptr<const open_spiel::Game> game =
      open_spiel::LoadGame(test_game);
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  // Store them directly into a vector so they are returned in order so
  // given a static initialization the model trains identically.
  absl::flat_hash_map<std::string, int> cache;
  std::vector<PVPNetModel::TrainInputs> train_inputs;
  train_inputs.reserve(4520);
  SolveState(*state, cache, train_inputs);
  return train_inputs;
}

PVPNetModel BuildModel(const Game& game, const std::string& nn_model,
                      bool create_graph) {
  std::string tmp_dir = open_spiel::file::GetTmpDir();
  std::string filename = absl::StrCat(
      "open_spiel_PVPNet_test_", nn_model, ".pb");

  if (create_graph) {
    SPIEL_CHECK_TRUE(CreateGraphDefMPG(
        game,
        /*learning_rate=*/0.01,
        /*weight_decay=*/0.0001,
        tmp_dir, filename,
        nn_model, /*nn_width=*/32, /*nn_depth=*/2, /*verbose=*/true));
  }

  std::string model_path = absl::StrCat(tmp_dir, "/", filename);
  SPIEL_CHECK_TRUE(file::Exists(model_path));

  PVPNetModel model(game, tmp_dir, filename, "/cpu:0");

  return model;
}




auto runInference(const std::unique_ptr<State> &state, PVPNetModel &model)
{

    std::vector<Action> legal_actions = state->LegalActions();
    auto [environment,_state] = PVPNetModel::InferenceInputs::Extract(state->ObservationTensor());
    PVPNetModel::InferenceInputs inputs = {environment, _state};

    return model.Inference(std::vector{inputs}).front();
}

auto runInferences(const std::shared_ptr<const Game> &game, PVPNetModel &model, int batch_size)
{
    std::vector<PVPNetModel::InferenceInputs> batch_inputs;
    for(int i=0;i<batch_size;i++)
    {
        auto state=game->NewInitialState();
        auto input = PVPNetModel::InferenceInputs::Extract(state->ObservationTensor());
        batch_inputs.push_back(input);
    }

    return model.Inference(batch_inputs);
}

void TestModelCreation(const std::string& nn_model) {
  std::cout << "TestModelCreation: " << nn_model << std::endl;
  std::shared_ptr<const Game> game = LoadGame(test_game);
  PVPNetModel model = BuildModel(*game, nn_model, true);
  runInferences(game, model,100);
}

// Can learn a single trajectory
void TestModelLearnsSimple(const std::string& nn_model) {
  std::cout << "TestModelLearnsSimple: " << nn_model << std::endl;
  std::shared_ptr<const Game> game = LoadGame(test_game);
  PVPNetModel model = BuildModel(*game, nn_model, false);

  std::vector<PVPNetModel::TrainInputs> train_inputs;
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();

  while (!state->IsTerminal()) {
    std::vector<float> environment = state->ObservationTensor();
    auto state_=environment.back();
    environment.pop_back();
    std::vector<Action> legal_actions = state->LegalActions();
    Action action = legal_actions[0];
    ActionsAndProbs policy({{action, 1}});

    train_inputs.emplace_back(PVPNetModel::TrainInputs{
        environment, state_, policy, 1});

    PVPNetModel::InferenceInputs inputs = {environment,state_};
    std::vector<PVPNetModel::InferenceOutputs> out =
        model.Inference(std::vector{inputs});
    SPIEL_CHECK_EQ(out.size(), 1);
    SPIEL_CHECK_EQ(out[0].policy.size(), legal_actions.size());

    state->ApplyAction(action);
  }

  std::cout << "states: " << train_inputs.size() << std::endl;
  std::vector<PVPNetModel::LossInfo> losses;
  const double policy_loss_goal = 0.05;
  const double value_loss_goal = 0.05;
  for (int i = 0; i < 200; i++) {
    PVPNetModel::LossInfo loss = model.Learn(train_inputs);
    std::cout << absl::StrFormat(
        "%d: Losses(total: %.3f, policy: %.3f, value: %.3f, l2: %.3f)\n",
         i, loss.Total(), loss.Policy(), loss.Value(), loss.L2());
    losses.push_back(loss);
    if (loss.Policy() < policy_loss_goal && loss.Value() < value_loss_goal) {
      break;
    }
  }
  SPIEL_CHECK_GT(losses.front().Total(), losses.back().Total());
  SPIEL_CHECK_GT(losses.front().Policy(), losses.back().Policy());
  SPIEL_CHECK_GT(losses.front().Value(), losses.back().Value());
  SPIEL_CHECK_LT(losses.back().Value(), value_loss_goal);
  SPIEL_CHECK_LT(losses.back().Policy(), policy_loss_goal);
}

// Can learn the optimal policy.
void TestModelLearnsOptimal(
    const std::string& nn_model,
    const std::vector<PVPNetModel::TrainInputs>& train_inputs) {
  std::cout << "TestModelLearnsOptimal: " << nn_model << std::endl;
  std::shared_ptr<const Game> game = LoadGame(test_game);
  PVPNetModel model = BuildModel(*game, nn_model, false);

  std::cout << "states: " << train_inputs.size() << std::endl;
  std::vector<PVPNetModel::LossInfo> losses;
  const double policy_loss_goal = 0.1;
  const double value_loss_goal = 0.1;
  for (int i = 0; i < 500; i++) {
    PVPNetModel::LossInfo loss = model.Learn(train_inputs);
    std::cout << absl::StrFormat(
        "%d: Losses(total: %.3f, policy: %.3f, value: %.3f, l2: %.3f)\n",
         i, loss.Total(), loss.Policy(), loss.Value(), loss.L2());
    losses.push_back(loss);
    if (loss.Policy() < policy_loss_goal && loss.Value() < value_loss_goal) {
      break;
    }
  }
  SPIEL_CHECK_GT(losses.front().Total(), losses.back().Total());
  SPIEL_CHECK_GT(losses.front().Policy(), losses.back().Policy());
  SPIEL_CHECK_GT(losses.front().Value(), losses.back().Value());
  SPIEL_CHECK_LT(losses.back().Value(), value_loss_goal);
  SPIEL_CHECK_LT(losses.back().Policy(), policy_loss_goal);
}

    std::unique_ptr<open_spiel::State> LoadStateFromTestData(const std::shared_ptr<const Game> &game, const std::string input_file)
    {
            std::ifstream input(input_file);
            input.exceptions(std::ifstream::failbit | std::ifstream::badbit);
            std::cout << "Reading state from " << input_file << std::endl;
            size_t starting_position;
            std::cout << "Starting position: ";
            input >> starting_position;
            std::cout << starting_position << std::endl;
            open_spiel::mpg::WeightedGraphType graph(open_spiel::mpg::WeightedGraphType::from_stream(input));
            std::shared_ptr<open_spiel::mpg::Environment> environment = std::make_shared<open_spiel::mpg::Environment>(graph,starting_position);
            return std::make_unique<open_spiel::mpg::MPGEnvironmentState>(game, environment);
    }

    void TestModelLoadCheckpoint(const std::string &nn_model,const std::string &path)
    {
        std::cout << "TestModelLoadCheckpoint: " << nn_model << std::endl;
        std::shared_ptr<const Game> game = LoadGame(test_game);
        auto model_name=absl::StrFormat("PVPNet_test_%s.pb",nn_model);
        std::filesystem::path input_path = path;
        input_path = input_path / "test" / "input.txt" ;
        std::filesystem::path model_dir = path;

        std::filesystem::path checkpoint_path = path;
        checkpoint_path = checkpoint_path / "checkpoints" / nn_model;

        if(nn_model == "mlp")
            return;
        if(!std::filesystem::exists(input_path))
            SpielFatalError(absl::StrFormat("Input %s does not exist",input_path.string()));
        if(!std::filesystem::exists(model_dir / model_name))
            SpielFatalError(absl::StrFormat("Model %s does not exist",(model_dir / model_name).string()));
        if(!std::filesystem::exists(checkpoint_path.string()+ ".index"))
            SpielFatalError(absl::StrFormat("Checkpoint %s does not exist",checkpoint_path.string()));

        auto state=LoadStateFromTestData(game,input_path);
        PVPNetModel model = PVPNetModel(*game, model_dir, model_name);

        model.LoadCheckpoint(checkpoint_path);
        auto result=runInference(state, model);
        std::cout << "Value: " << result.value << " Policy: " << result.policy << std::endl;

    }

}  // namespace
}  // namespace open_spiel
ABSL_FLAG(bool, checkpoint_tests, false, "Run tests with checkpoint loading.");
ABSL_FLAG(bool, ignore_creation, false, "Ignore model creation tests.");
ABSL_FLAG(std::string, test_models_path, "test_data", "Path to test models.");

int main(int argc, char** argv) {
    absl::ParseCommandLine(argc, argv);
    auto test_data_path=absl::GetFlag(FLAGS_test_models_path);
    auto checkpoint_tests=absl::GetFlag(FLAGS_checkpoint_tests);
    auto ignore_creation=absl::GetFlag(FLAGS_ignore_creation);
    std::cout << "Setting CUDA_VISIBLE_DEVICES to empty string to ignore GPU" << std::endl;
    setenv("CUDA_VISIBLE_DEVICES", "", 1);
    if (!ignore_creation)
    {
        open_spiel::algorithms::mpg::TestModelCreation("mlp");
        open_spiel::algorithms::mpg::TestModelCreation("gnn");
    }
    if (checkpoint_tests)
    {
        open_spiel::algorithms::mpg::TestModelLoadCheckpoint("mlp",test_data_path);
        open_spiel::algorithms::mpg::TestModelLoadCheckpoint("gnn",test_data_path);
        return 0;
    }
    else
        std::cerr << "No checkpoint path provided. Skipping checkpoint test." << std::endl;
//    open_spiel::algorithms::mpg::TestModelCreation("resnet");

  // Tests below here reuse the graphs created above. Graph creation is slow
  // due to calling a separate python process.
/*
  open_spiel::algorithms::mpg::TestModelLearnsSimple("mlp");
  open_spiel::algorithms::mpg::TestModelLearnsSimple("conv2d");
  open_spiel::algorithms::mpg::TestModelLearnsSimple("resnet");

  auto train_inputs = open_spiel::algorithms::mpg::SolveGame();
  open_spiel::algorithms::mpg::TestModelLearnsOptimal("mlp", train_inputs);
  open_spiel::algorithms::mpg::TestModelLearnsOptimal("conv2d", train_inputs);
  open_spiel::algorithms::mpg::TestModelLearnsOptimal("resnet", train_inputs);
  */
}
