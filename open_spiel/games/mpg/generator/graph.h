//
// Created by ramizouari on 23/06/23.
//

#ifndef OPEN_SPIEL_GENERATOR_GRAPH_H
#define OPEN_SPIEL_GENERATOR_GRAPH_H
#include "../mpg.h"
#include "weight.h"
namespace open_spiel::mpg
{
    class GraphGenerator: public DefaultSeedable
    {
    public:
        explicit GraphGenerator(std::uint64_t seed);
        GraphGenerator();
        virtual GraphType operator()()=0;
        virtual ~GraphGenerator()=default;
    protected:
        std::mt19937_64 rng;
    };

    class GnpGenerator:public GraphGenerator
    {
        std::uint64_t n;
        double p;
    public:
        GnpGenerator(std::uint64_t n, double p, std::uint64_t seed = 0);
        GnpGenerator(std::uint64_t n, double p);
        GraphType operator()() override;
    };

    class GncGenerator:public GnpGenerator
    {
        double c;
    public:
        GncGenerator(std::uint64_t n, double c, std::uint64_t seed);
        GncGenerator(std::uint64_t n, double c);
    };

    class SinklessGnpGenerator: public GraphGenerator
    {
        std::uint64_t n;
        double p;
    public:
        SinklessGnpGenerator(std::uint64_t n, double p, std::uint64_t seed);
        SinklessGnpGenerator(std::uint64_t n, double p);
        GraphType operator()() override;
    };

    class SinklessGncGenerator: public SinklessGnpGenerator
    {
        std::uint64_t n;
        double c;
    public:
        SinklessGncGenerator(std::uint64_t n, double c, std::uint64_t seed);
        SinklessGncGenerator(std::uint64_t n, double c);
    };

    class UniformlyStochasticSinklessGnpGenerator: public GraphGenerator
    {
        NodeType n_min,n_max;
        double p_min,p_max;
    public:
        UniformlyStochasticSinklessGnpGenerator(NodeType n_min, NodeType n_max, double p_min, double p_max, std::uint64_t seed);
        UniformlyStochasticSinklessGnpGenerator(NodeType n_min, NodeType n_max, double p_min, double p_max);
        GraphType operator()() override;
    };

    class UniformlyStochasticSinklessGncGenerator: public GraphGenerator
    {
        NodeType n_min,n_max;
        double c_min,c_max;
    public:
        UniformlyStochasticSinklessGncGenerator(NodeType n_min, NodeType n_max, double c_min, double c_max, std::uint64_t seed);
        UniformlyStochasticSinklessGncGenerator(NodeType n_min, NodeType n_max, double c_min, double c_max);
        GraphType operator()() override;

    };
    using USSGnpGenerator = UniformlyStochasticSinklessGnpGenerator;
    using SGnpGenerator=SinklessGnpGenerator;
    using SGncGenerator=SinklessGncGenerator;
    using USSGncGenerator = UniformlyStochasticSinklessGncGenerator;

    class WeightedGraphGenerator:public RecursiveSeedable
    {
        std::shared_ptr<GraphGenerator> graph_generator;
        std::shared_ptr<WeightGenerator> weight_generator;
    public:
        WeightedGraphGenerator(std::shared_ptr<GraphGenerator> graph_generator, std::shared_ptr<WeightGenerator> weight_generator);
        WeightedGraphType operator()();
    };
}

#endif //OPEN_SPIEL_GENERATOR_GRAPH_H
