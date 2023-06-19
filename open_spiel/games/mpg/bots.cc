//
// Created by ramizouari on 18/06/23.
//
#include "bots.h"

namespace open_spiel::mpg
{

    Action MPGBot::Step(const State &state)
    {
        auto mpg=dynamic_cast<const MPGEnvironmentState*>(&state);
        return Step(*mpg);
    }
    void MPGBot::ForceAction(const State &state, Action action)
    {
        auto mpg=dynamic_cast<const MPGEnvironmentState*>(&state);
        ForceAction(*mpg,action);
    }
    ActionsAndProbs MPGBot::GetPolicy(const State &state)
    {
        auto mpg=dynamic_cast<const MPGEnvironmentState*>(&state);
        return GetPolicy(*mpg);
    }

    std::pair<ActionsAndProbs, Action> MPGBot::StepWithPolicy(
            const State &state)
    {
        auto mpg=dynamic_cast<const MPGEnvironmentState*>(&state);
        return StepWithPolicy(*mpg);
    }

    Action GreedyBot::Step(const MPGEnvironmentState &state)
    {
        auto legal_actions_payoffs=state.LegalActionsWithPayoffs();
        Action best_action;
        auto current_player=state.CurrentPlayer();
        return std::max_element(legal_actions_payoffs.begin(),legal_actions_payoffs.end(),
                                [current_player](auto &a,auto &b)
                                {
                                    return WeightFromPerspective(a.second,current_player)< WeightFromPerspective(b.second,current_player);
                                })->first;
    }

    // Notifies the bot that it should consider that it took action action in
    // the given state.
    void GreedyBot::ForceAction(const MPGEnvironmentState &state, Action action)
    {
        // Do nothing
    }

    bool GreedyBot::ProvidesPolicy()
    {
        return true;
    }

    ActionsAndProbs GreedyBot::GetPolicy(const MPGEnvironmentState &state)
    {
        auto step=Step(state);
        return {{step,1.0}};
    }

    std::pair<ActionsAndProbs, Action> GreedyBot::StepWithPolicy(
            const MPGEnvironmentState &state)
    {
        auto step=Step(state);
        auto action_prob= GetPolicy(state);
        return {action_prob,step};
    }

    // Creates a clone of the bot with an independent copy of its internal state.
    // The original bot and the clone are completely independent.
    // The Clone method should be as cheap to execute as possible.
    //
    // Important: the cloned bot must sample actions independently and differently
    // from the original bot. I.e. if the bot uses any randomness controlling key,
    // that key *must* be reseeded when cloning the bot.
    // The typical use-case for cloning is generating multiple continuations
    // of a game. The cloned bot should produce the same policy as the original
    // bot, but there *must* be no correllation between action sampling of
    // the original bot and its clone.
    // Note that bot clones must also sample actions independently.
    bool GreedyBot::IsClonable() const
    {
        return true;
    }

    std::unique_ptr<Bot> GreedyBot::Clone()
    {
        return std::make_unique<GreedyBot>();
    }


    Action EpsilonGreedyBot::Step(const MPGEnvironmentState &state)
    {
        if(bernoulli(rng))
        {
            auto legal_actions=state.LegalActions();
            std::uniform_int_distribution<NodeType> uniform(0,legal_actions.size()-1);
            return legal_actions[uniform(rng)];
        }
        else
            return GreedyStep(state);
    }

    Action EpsilonGreedyBot::GreedyStep(const MPGEnvironmentState &state)
    {
        auto legal_actions_payoffs=state.LegalActionsWithPayoffs();
        Action best_action;
        auto current_player=state.CurrentPlayer();
        return std::max_element(legal_actions_payoffs.begin(),legal_actions_payoffs.end(),
                                [current_player](auto &a,auto &b)
                                {
                                    return WeightFromPerspective(a.second,current_player)< WeightFromPerspective(b.second,current_player);
                                })->first;
    }

    // Notifies the bot that it should consider that it took action action in
    // the given state.
    void EpsilonGreedyBot::ForceAction(const MPGEnvironmentState &state, Action action)
    {
        // Do nothing
    }

    bool EpsilonGreedyBot::ProvidesPolicy()
    {
        return true;
    }

    ActionsAndProbs EpsilonGreedyBot::GetPolicy(const MPGEnvironmentState &state)
    {
        auto actions=state.LegalActions();
        ActionsAndProbs policy;
        policy.reserve(actions.size());
        auto greedy_action=GreedyStep(state);
        for(auto a:actions)
        {
            double p=epsilon/actions.size();
            if(a==greedy_action)
                p+=1-epsilon;
            policy.emplace_back(a,p);
        }
        return policy;
    }

    std::pair<ActionsAndProbs, Action> EpsilonGreedyBot::StepWithPolicy(
            const MPGEnvironmentState &state)
    {
        auto step=Step(state);
        auto action_prob= GetPolicy(state);
        return {action_prob,step};
    }

    // Creates a clone of the bot with an independent copy of its internal state.
    // The original bot and the clone are completely independent.
    // The Clone method should be as cheap to execute as possible.
    //
    // Important: the cloned bot must sample actions independently and differently
    // from the original bot. I.e. if the bot uses any randomness controlling key,
    // that key *must* be reseeded when cloning the bot.
    // The typical use-case for cloning is generating multiple continuations
    // of a game. The cloned bot should produce the same policy as the original
    // bot, but there *must* be no correllation between action sampling of
    // the original bot and its clone.
    // Note that bot clones must also sample actions independently.
    bool EpsilonGreedyBot::IsClonable() const
    {
        return true;
    }

    std::unique_ptr<Bot> EpsilonGreedyBot::Clone()
    {
        std::uniform_int_distribution d;
        return std::make_unique<EpsilonGreedyBot>(epsilon,d(rng));
    }

    EpsilonGreedyBot::EpsilonGreedyBot(double _epsilon,std::uint64_t seed):epsilon(_epsilon),rng(seed),bernoulli(_epsilon)
    {

    }
    EpsilonGreedyBot::EpsilonGreedyBot(double _epsilon):epsilon(_epsilon),bernoulli(_epsilon)
    {
        std::random_device rd;
        auto seed=rd();
        rng.seed(seed);
    }
}

#include "bots.h"
