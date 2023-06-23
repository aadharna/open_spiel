//
// Created by ramizouari on 23/06/23.
//

#include <filesystem>
#include "environment.h"
#include "graph.h"
#include "mpg/generator/environment.h"
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <fstream>


namespace open_spiel::mpg
{
    UniformlyStochasticUniformGnpEnvironmentFactory::UniformlyStochasticUniformGnpEnvironmentFactory (NodeType n_min,
                                                                                                      NodeType n_max, double p_min, double p_max, WeightType a, WeightType b,
                                                                                                      std::uint64_t seed) : GeneratorEnvironmentFactory
                                                                                                                                    (
                                                                                                                                            std::make_shared<WeightedGraphGenerator>(std::make_shared<UniformlyStochasticSinklessGnpGenerator>(n_min,n_max,p_min,p_max,seed),
                                                                                                                                                                                     std::make_shared<UniformWeightGenerator>(a,b,seed)),
                                                                                                                                            seed
                                                                                                                                    )

    {
        if(n_min>n_max)
            throw std::invalid_argument("UniformlyStochasticUniformGnpEnvironmentFactory: n_min>n_max");
        if(p_min>p_max)
            throw std::invalid_argument("UniformlyStochasticUniformGnpEnvironmentFactory: p_min>p_max");
        if(a>b)
            throw std::invalid_argument("UniformlyStochasticUniformGnpEnvironmentFactory: a>b");

    }


    GeneratorEnvironmentFactory::GeneratorEnvironmentFactory(std::shared_ptr<WeightedGraphGenerator> weighted_graph_generator, std::uint64_t seed) : weighted_graph_generator(std::move(weighted_graph_generator)),
                                                                                                                                                     rng(seed)
    {

    }

    std::shared_ptr<Environment> GeneratorEnvironmentFactory::NewEnvironment(const MPGMetaGame &metaGame)
    {
        auto G=weighted_graph_generator->operator()();
        std::uniform_int_distribution<NodeType> dist(0,G.size()-1);
        return std::make_shared<Environment>(G, dist(rng));
    }

    UniformGnpEnvironmentFactory::UniformGnpEnvironmentFactory(NodeType n, WeightType p, WeightType a, WeightType b, std::uint64_t seed): GeneratorEnvironmentFactory(
            std::make_shared<WeightedGraphGenerator>(std::make_shared<SinklessGnpGenerator>(n,p,seed),
                                                     std::make_shared<UniformWeightGenerator>(a,b,seed)),
            seed
    )
    {

    }

    DatasetEnvironmentFactory::DatasetEnvironmentFactory(std::string dataset_path_, std::uint64_t seed):dataset_path(std::move(dataset_path_)),rng(seed)
    {
        if(!std::filesystem::exists(dataset_path))
            throw std::invalid_argument("DatasetEnvironmentFactory: dataset path does not exist");
        for(auto &p:std::filesystem::directory_iterator(dataset_path))
            if(p.is_regular_file())
                dataset_files.push_back(p.path());
    }

    namespace
    {

        WeightedGraphType read_graph(std::istream &stream)
        {
            std::string line;
            WeightedGraphType G;
            int size=0;
            while(std::getline(stream,line))
            {
                std::istringstream iss(line);
                NodeType u,v;
                WeightType w;
                iss>>u>>v>>w;
                size=std::max<int>({G.size(),u+1,v+1});
                if(G.capacity()<size)
                {
                    G.reserve(2 * size);
                    G.resize(size);
                }
                G[u].emplace(v,w);
            }
            return G;
        }

        WeightedGraphType read_graph(const std::filesystem::path &path)
        {
            std::ifstream file(path);
            if(!file.is_open())
                throw std::invalid_argument("DatasetEnvironmentFactory: could not open file "+path.string());
            return read_graph(file);
        }

        WeightedGraphType read_graph_gz(const std::filesystem::path &path)
        {
            std::ifstream file(path);
            if(!file.is_open())
                throw std::invalid_argument("DatasetEnvironmentFactory: could not open file "+path.string());
            boost::iostreams::filtering_istream in;
            in.push(boost::iostreams::gzip_decompressor());
            in.push(file);
            return read_graph(in);
        }
    }

    std::shared_ptr<Environment> DatasetEnvironmentFactory::NewEnvironment(const MPGMetaGame &metaGame)
    {
        std::uniform_int_distribution<std::size_t> dist(0,dataset_files.size()-1);
        auto path=std::filesystem::path(dataset_path)/dataset_files[dist(rng)];
        if(path.extension()==".gz")
        {
            auto G=read_graph_gz(path);
            std::uniform_int_distribution<NodeType> starting_vertex(0,G.size()-1);
            return std::make_shared<Environment>(G,starting_vertex(rng));
        }
        else
        {
            auto G=read_graph(path);
            std::uniform_int_distribution<NodeType> starting_vertex(0,G.size()-1);
            return std::make_shared<Environment>(G,starting_vertex(rng));
        }
    }

}