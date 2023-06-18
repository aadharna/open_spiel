//
// Created by ramizouari on 18/06/23.
//

#ifndef OPEN_SPIEL_BOTS_H
#define OPEN_SPIEL_BOTS_H
#include "open_spiel/spiel_bots.h"
#include "open_spiel/games/mpg/mpg.h"

namespace open_spiel::mpg
{
    class GreedyBot:public Bot {
    public:
        Action Step(const State &state) override;

        // Notifies the bot that it should consider that it took action action in
        // the given state.
        void ForceAction(const State &state, Action action) override;

        // Extends a bot to support explicit stochasticity, meaning that it can
        // return a distribution over moves.
        bool ProvidesPolicy() override;

        ActionsAndProbs GetPolicy(const State &state) override;

        std::pair<ActionsAndProbs, Action> StepWithPolicy(
                const State &state) override;

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
        bool IsClonable() const override;

        std::unique_ptr<Bot> Clone() override;
    };

    class EpsilonGreedyBot :public Bot
    {
        double epsilon;
        std::mt19937_64 rng;
        std::bernoulli_distribution bernoulli;
    public:
        EpsilonGreedyBot(double _epsilon,std::uint64_t seed);
        EpsilonGreedyBot(double _epsilon);

        Action Step(const State &state) override;

        // Notifies the bot that it should consider that it took action action in
        // the given state.
        void ForceAction(const State &state, Action action) override;

        // Extends a bot to support explicit stochasticity, meaning that it can
        // return a distribution over moves.
        bool ProvidesPolicy() override;

        ActionsAndProbs GetPolicy(const State &state) override;

        std::pair<ActionsAndProbs, Action> StepWithPolicy(
                const State &state) override;

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
        bool IsClonable() const override;

        std::unique_ptr<Bot> Clone() override;

    protected:
        Action GreedyStep(const State &state);
    };
}


#endif //OPEN_SPIEL_BOTS_H
