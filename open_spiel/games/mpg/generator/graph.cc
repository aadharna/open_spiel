//
// Created by ramizouari on 23/06/23.
//

#include "graph.h"
#include "choice.h"

namespace open_spiel::mpg
{
    SinklessGnpGenerator::SinklessGnpGenerator(std::uint64_t n, double p, std::uint64_t seed) : n(n),p(p),rng(seed)
    {
    }

    GraphType SinklessGnpGenerator::operator()()
    {
        GraphType G(n);
        std::binomial_distribution<NodeType> degree_distribution(n,p);
        for(NodeType u=0;u<n;u++)
        {
            NodeType d=0;
            while(d==0)
                d=degree_distribution(rng);
            G[u]= generator::choose(n,d,rng);
        }
        return G;
    }




    GnpGenerator::GnpGenerator(std::uint64_t n, double p, std::uint64_t seed): n(n),p(p),rng(seed)
    {

    }

    GraphType GnpGenerator::operator()()
    {
        GraphType G(n);
        std::binomial_distribution<NodeType> degree_distribution(n,p);
        for(NodeType u=0;u<n;u++)
        {
            auto d=degree_distribution(rng);
            G[u]= generator::choose(n,d,rng);
        }
        return G;
    }

    WeightedGraphGenerator::WeightedGraphGenerator(std::shared_ptr<GraphGenerator> graph_generator,
                                                   std::shared_ptr<WeightGenerator> weight_generator) : graph_generator(std::move(graph_generator)),
                                                                                                        weight_generator(std::move(weight_generator))
    {
    }


    WeightedGraphType WeightedGraphGenerator::operator()()
    {
        auto G=graph_generator->operator()();
        WeightedGraphType W(G.size());
        for(size_t u=0;u<G.size();u++) for(auto v:G[u])
                W[u].emplace(v,weight_generator->operator()());
        return W;
    }


    GraphType UniformlyStochasticSinklessGnpGenerator::operator()()
    {
        std::uniform_int_distribution<NodeType> n_distribution(n_min,n_max);
        std::uniform_real_distribution<double> p_distribution(p_min,p_max);
        auto generator=std::make_unique<SinklessGnpGenerator>(n_distribution(rng),p_distribution(rng),rng());
        return (*generator)();
    }

    UniformlyStochasticSinklessGnpGenerator::UniformlyStochasticSinklessGnpGenerator(NodeType n_min, NodeType n_max,
                                                                                     double p_min, double p_max,
                                                                                     std::uint64_t seed):
            n_min(n_min),n_max(n_max),p_min(p_min),p_max(p_max),rng(seed){

    }

    SinklessGncGenerator::SinklessGncGenerator(std::uint64_t n, double c, std::uint64_t seed) : SinklessGnpGenerator(n,std::min<double>(c/n,1),seed),
        n(n),c(c),rng(seed)
    {
    }

    GncGenerator::GncGenerator(std::uint64_t n, double c, std::uint64_t seed) : GnpGenerator(n,std::min<double>(c/n,1),seed),c(c)
    {
    }

    UniformlyStochasticSinklessGncGenerator::UniformlyStochasticSinklessGncGenerator(NodeType n_min, NodeType n_max,
                                                                                     double c_min, double c_max,
                                                                                     std::uint64_t seed):
                                                                                     n_min(n_min),n_max(n_max),c_min(c_min),c_max(c_max),rng(seed)
     {

    }

    GraphType UniformlyStochasticSinklessGncGenerator::operator()()
    {
        std::uniform_int_distribution<NodeType> n_distribution(n_min,n_max);
        std::uniform_real_distribution<double> c_distribution(c_min,c_max);
        auto generator=std::make_unique<SinklessGncGenerator>(n_distribution(rng),c_distribution(rng),rng());
        return (*generator)();
    }
}