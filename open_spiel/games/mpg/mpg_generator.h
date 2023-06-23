//
// Created by ramizouari on 05/06/23.
//

#ifndef OPEN_SPIEL_MPG_GENERATOR_H
#define OPEN_SPIEL_MPG_GENERATOR_H
#include "mpg.h"
#include "generator/meta_factory.h"
#include "generator/graph.h"
#include "generator/weight.h"


namespace open_spiel::mpg
{

    class UniformGnpMetaFactory : public ParametricMetaFactory
    {
        std::shared_ptr<WeightedGraphGenerator> weighted_graph_generator;
        std::mt19937_64 rng;
    public:
        std::shared_ptr<const Game> GameFromArgs(const std::vector<double> &args,const GameParameters &params) override;
        std::size_t MinArgs() const override;
        std::size_t MaxArgs() const override;
    };

    class UniformlyStochasticUniformGnpMetaFactory : public ParametricMetaFactory
    {
        std::shared_ptr<WeightedGraphGenerator> weighted_graph_generator;
        std::mt19937_64 rng;
    public:
        std::shared_ptr<const Game> GameFromArgs(const std::vector<double> &args,const GameParameters &params) override;
        std::size_t MinArgs() const override;
        std::size_t MaxArgs() const override;
    };


    class DatasetMetaFactory : public StringParametricMetaFactory
    {
        std::shared_ptr<WeightedGraphGenerator> weighted_graph_generator;
        std::mt19937_64 rng;
    public:
        std::shared_ptr<const Game> GameFromArgs(const std::vector<std::string> &args,const GameParameters &params) override;
        std::size_t MinArgs() const override;
        std::size_t MaxArgs() const override;
    };

    extern std::unique_ptr<ParserMetaFactory> parserMetaFactory;


}

#endif //OPEN_SPIEL_MPG_GENERATOR_H
