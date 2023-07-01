//
// Created by ramizouari on 23/06/23.
//

#ifndef OPEN_SPIEL_GENERATOR_GRAPH_H
#define OPEN_SPIEL_GENERATOR_GRAPH_H
#include "../mpg.h"
#include "weight.h"
namespace open_spiel::mpg
{
    class GraphGenerator
    {
    public:
        virtual GraphType operator()()=0;
        virtual ~GraphGenerator()=default;
    };

    class GnpGenerator:public GraphGenerator
    {
        std::uint64_t n;
        double p;
        std::mt19937_64 rng;
    public:
        GnpGenerator(std::uint64_t n, double p, std::uint64_t seed = 0);
        GraphType operator()() override;
    };

    class GncGenerator:public GnpGenerator
    {
        double c;
    public:
        GncGenerator(std::uint64_t n, double c, std::uint64_t seed = 0);
    };

    class SinklessGnpGenerator: public GraphGenerator
    {
        std::uint64_t n;
        double p;
        std::mt19937_64 rng;
    public:
        SinklessGnpGenerator(std::uint64_t n, double p, std::uint64_t seed = 0);
        GraphType operator()() override;
    };

    class SinklessGncGenerator: public SinklessGnpGenerator
    {
        std::uint64_t n;
        double c;
        std::mt19937_64 rng;
    public:
        SinklessGncGenerator(std::uint64_t n, double c, std::uint64_t seed = 0);
    };

    class UniformlyStochasticSinklessGnpGenerator: public GraphGenerator
    {
        NodeType n_min,n_max;
        double p_min,p_max;
        std::mt19937_64 rng;
    public:
        UniformlyStochasticSinklessGnpGenerator(NodeType n_min, NodeType n_max, double p_min, double p_max, std::uint64_t seed = 0);
        GraphType operator()() override;
    };

    class UniformlyStochasticSinklessGncGenerator: public GraphGenerator
    {
        NodeType n_min,n_max;
        double c_min,c_max;
        std::mt19937_64 rng;
    public:
        UniformlyStochasticSinklessGncGenerator(NodeType n_min, NodeType n_max, double c_min, double c_max, std::uint64_t seed = 0);
        GraphType operator()() override;

    };
    using USSGnpGenerator = UniformlyStochasticSinklessGnpGenerator;
    using SGnpGenerator=SinklessGnpGenerator;
    using SGncGenerator=SinklessGncGenerator;
    using USSGncGenerator = UniformlyStochasticSinklessGncGenerator;

    class WeightedGraphGenerator
    {
        std::shared_ptr<GraphGenerator> graph_generator;
        std::shared_ptr<WeightGenerator> weight_generator;
    public:
        WeightedGraphGenerator(std::shared_ptr<GraphGenerator> graph_generator, std::shared_ptr<WeightGenerator> weight_generator);
        WeightedGraphType operator()();

    };
}

#endif //OPEN_SPIEL_GRAPH_H
