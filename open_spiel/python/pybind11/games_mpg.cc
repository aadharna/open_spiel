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

#include "open_spiel/python/pybind11/games_mpg.h"

#include "open_spiel/games/mpg/mpg.h"

#include "open_spiel/spiel.h"
#include "open_spiel/python/pybind11/pybind11.h"
#include "open_spiel/spiel_bots.h"
#include "open_spiel/games/mpg/bots.h"

namespace py = ::pybind11;

//PYBIND11_SMART_HOLDER_TYPE_CASTERS(ChessBoard);
//PYBIND11_SMART_HOLDER_TYPE_CASTERS(ChessState);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::mpg::MPGEnvironmentState);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::mpg::GreedyBot);
PYBIND11_SMART_HOLDER_TYPE_CASTERS(open_spiel::mpg::EpsilonGreedyBot);


void open_spiel::init_pyspiel_games_mpg(py::module& m) {
  py::module_ mpg = m.def_submodule("mpg");
  using namespace open_spiel::mpg;
  py::enum_<mpg::PlayerIdentifier>(mpg, "PlayerIdentifier")
    .value("PlayerMax", PlayerIdentifier::kMaxPlayer)
    .value("PlayerMin", PlayerIdentifier::kMinPlayer)
      .value("Player1", PlayerIdentifier::kPlayer1)
      .value("Player2", PlayerIdentifier::kPlayer2)
      .export_values();

  py::classh<MPGEnvironmentState, State>(mpg, "MPGEnvironmentState")
      .def("get_current_state", &MPGEnvironmentState::GetCurrentState)
      .def("get_mean_payoff", &MPGEnvironmentState::GetMeanPayoff, py::arg("with_offset")=true)
      .def("mean_payoff", &MPGEnvironmentState::GetMeanPayoff, py::arg("with_offset")=true)
      .def("set_payoff_offset", &MPGEnvironmentState::SetPayoffOffset,py::arg("offset"))
      .def("legal_actions_with_payoffs",&MPGEnvironmentState::LegalActionsWithPayoffs)
      .def("graph_size", &MPGEnvironmentState::GraphSize)
      .def("count_edges", &MPGEnvironmentState::CountEdges)
      .def("get_payoff", &MPGEnvironmentState::GetPayoff)
      .def("current_state", &MPGEnvironmentState::CurrentState)
      // Pickle support
      .def(py::pickle(
          [](const MPGEnvironmentState& state) {  // __getstate__
            return SerializeGameAndState(*state.GetGame(), state);
          },
          [](const std::string& data) {  // __setstate__
            std::pair<std::shared_ptr<const Game>, std::unique_ptr<State>>
                game_and_state = DeserializeGameAndState(data);
            return dynamic_cast<MPGEnvironmentState*>(game_and_state.second.release());
          }));

    py::classh<GreedyBot, Bot>(mpg, "GreedyBot")
            .def(py::init([]() {
              return new GreedyBot;
            }))
            .def("step", py::overload_cast<const MPGEnvironmentState&>(&GreedyBot::Step))
            .def("step_with_policy", py::overload_cast<const MPGEnvironmentState&>(&GreedyBot::StepWithPolicy))
            .def("restart", &GreedyBot::Restart)
            .def("restart_at", &GreedyBot::RestartAt)
            .def("provides_force_action", &GreedyBot::ProvidesForceAction)
            .def("force_action", py::overload_cast<const MPGEnvironmentState&, Action>(&GreedyBot::ForceAction))
            .def("inform_action", &GreedyBot::InformAction)
            .def("inform_actions", &GreedyBot::InformActions)
            .def("provides_policy", &GreedyBot::ProvidesPolicy)
            .def("get_policy", py::overload_cast<const MPGEnvironmentState&>(&GreedyBot::GetPolicy))
            .def("step_with_policy", py::overload_cast<const MPGEnvironmentState&>(&GreedyBot::StepWithPolicy))
            .def("is_clonable", &GreedyBot::IsClonable)
            .def("clone", &GreedyBot::Clone);

    py::classh<EpsilonGreedyBot, Bot>(mpg, "EpsilonGreedyBot")
            .def(py::init([](double epsilon,std::uint64_t seed) {
                return new EpsilonGreedyBot(epsilon,seed);
            }),
                 py::arg("epsilon"),
                py::arg("seed")
                 )
            .def(py::init([](double epsilon) {
                     return new EpsilonGreedyBot(epsilon);
                 }),
                 py::arg("epsilon")
            )
            .def("step", py::overload_cast<const MPGEnvironmentState&>(&EpsilonGreedyBot::Step))
            .def("step_with_policy", py::overload_cast<const MPGEnvironmentState&>(&EpsilonGreedyBot::StepWithPolicy))
            .def("restart", &EpsilonGreedyBot::Restart)
            .def("restart_at", &EpsilonGreedyBot::RestartAt)
            .def("provides_force_action", &EpsilonGreedyBot::ProvidesForceAction)
            .def("force_action", py::overload_cast<const MPGEnvironmentState&, Action>(&EpsilonGreedyBot::ForceAction))
            .def("inform_action", &EpsilonGreedyBot::InformAction)
            .def("inform_actions", &EpsilonGreedyBot::InformActions)
            .def("provides_policy", &EpsilonGreedyBot::ProvidesPolicy)
            .def("get_policy", py::overload_cast<const MPGEnvironmentState&>(&EpsilonGreedyBot::GetPolicy))
            .def("step_with_policy", py::overload_cast<const MPGEnvironmentState&>(&EpsilonGreedyBot::StepWithPolicy))
            .def("is_clonable", &EpsilonGreedyBot::IsClonable)
            .def("clone", &EpsilonGreedyBot::Clone);
  // action_to_move(action: int, board: ChessBoard)
}
