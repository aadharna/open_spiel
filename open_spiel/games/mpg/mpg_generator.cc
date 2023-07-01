//
// Created by ramizouari on 06/06/23.
//

#include "mpg_generator.h"
#include "mpg/generator/environment.h"
#include "generator/choice.h"

namespace open_spiel::mpg
{

    std::shared_ptr<Environment> ExampleEnvironmentFactory::NewEnvironment(const MPGMetaGame& metaGame) {
        auto G=WeightedGraphType::from_string(R"(1 2 5
2 3 -7
3 7 0
3 6 5
6 1 -3
1 4 4
4 5 -3
5 6 3
5 7 0
7 1 0
0 1 5)");
        return std::make_shared<Environment>(G, 0);
    }

    std::shared_ptr<const Game> UniformlyStochasticUniformGnpMetaFactory::GameFromArgs(const std::vector<double> &args,
                                                                                       const open_spiel::GameParameters &params)
   {
        if(args.size()<MinArgs() || args.size()>MaxArgs())
            throw std::invalid_argument("UniformGnpEnvironmentFactory::FromVector: invalid number of arguments");
        auto n_min=static_cast<NodeType>(args[0]);
        auto n_max=static_cast<NodeType>(args[1]);
        auto p_min=static_cast<WeightType>(args[2]);
        auto p_max=static_cast<WeightType>(args[3]);
        auto a=static_cast<WeightType>(args[4]);
        auto b=static_cast<WeightType>(args[5]);
        if(args.size()==7)
        {
            auto seed=static_cast<std::uint64_t>(args[6]);
            return std::make_shared<MPGMetaGame>(params, std::make_unique<UniformlyStochasticUniformGnpEnvironmentFactory>(n_min,n_max,p_min,p_max,a,b,seed));
        }
        else return std::make_shared<MPGMetaGame>(params, std::make_unique<UniformlyStochasticUniformGnpEnvironmentFactory>(n_min,n_max,p_min,p_max,a,b));
    }


    std::shared_ptr<const Game> UniformGnpMetaFactory::GameFromArgs(const std::vector<double> &args, const GameParameters &params)
    {
        if(args.size()<4 || args.size()>5)
            throw std::invalid_argument("UniformGnpEnvironmentFactory::FromVector: invalid number of arguments");
        auto n=static_cast<NodeType>(args[0]);
        auto p=static_cast<WeightType>(args[1]);
        auto a=static_cast<WeightType>(args[2]);
        auto b=static_cast<WeightType>(args[3]);
        if(args.size()==5)
        {
            auto seed=static_cast<std::uint64_t>(args[4]);
            return std::make_shared<MPGMetaGame>(params, std::make_unique<UniformGnpEnvironmentFactory>(n,p,a,b,seed));
        }
        else return std::make_shared<MPGMetaGame>(params, std::make_unique<UniformGnpEnvironmentFactory>(n,p,a,b));
    }

    std::size_t DatasetMetaFactory::MinArgs() const {
        return 1;
    }

    std::size_t DatasetMetaFactory::MaxArgs() const {
        return 1;
    }

    std::shared_ptr<const Game>
    DatasetMetaFactory::GameFromArgs(const std::vector<std::string> &args, const GameParameters &params) {
        if(args.size()!=1)
            throw std::invalid_argument("DatasetMetaFactory::FromVector: invalid number of arguments");
        auto path=args[0];
        return std::make_shared<MPGMetaGame>(params, std::make_unique<DatasetEnvironmentFactory>(path));
    }

    std::shared_ptr<const Game>
    UniformGncMetaFactory::GameFromArgs(const std::vector<double> &args, const GameParameters &params) {
        if(args.size()<4 || args.size()>5)
            throw std::invalid_argument("UniformGnpEnvironmentFactory::FromVector: invalid number of arguments");
        auto n=static_cast<NodeType>(args[0]);
        auto c=static_cast<WeightType>(args[1]);
        auto a=static_cast<WeightType>(args[2]);
        auto b=static_cast<WeightType>(args[3]);
        if(args.size()==5)
        {
            auto seed=static_cast<std::uint64_t>(args[4]);
            return std::make_shared<MPGMetaGame>(params, std::make_unique<UniformGncEnvironmentFactory>(n,c,a,b,seed));
        }
        else return std::make_shared<MPGMetaGame>(params, std::make_unique<UniformGncEnvironmentFactory>(n,c,a,b));    }

    std::size_t UniformGncMetaFactory::MinArgs() const {
        return 4;
    }

    std::size_t UniformGncMetaFactory::MaxArgs() const {
        return 5;
    }


    std::shared_ptr<const Game> UniformlyStochasticUniformGncMetaFactory::GameFromArgs(const std::vector<double> &args,
                                                                                       const GameParameters &params) {
        if(args.size()<MinArgs() || args.size()>MaxArgs())
            throw std::invalid_argument("UniformGnpEnvironmentFactory::FromVector: invalid number of arguments");
        auto n_min=static_cast<NodeType>(args[0]);
        auto n_max=static_cast<NodeType>(args[1]);
        auto c_min=static_cast<WeightType>(args[2]);
        auto c_max=static_cast<WeightType>(args[3]);
        auto a=static_cast<WeightType>(args[4]);
        auto b=static_cast<WeightType>(args[5]);
        if(args.size()==7)
        {
            auto seed=static_cast<std::uint64_t>(args[6]);
            return std::make_shared<MPGMetaGame>(params, std::make_unique<UniformlyStochasticUniformGncEnvironmentFactory>(n_min,n_max,c_min,c_max,a,b,seed));
        }
        else return std::make_shared<MPGMetaGame>(params, std::make_unique<UniformlyStochasticUniformGncEnvironmentFactory>(n_min,n_max,c_min,c_max,a,b));
    }

    std::size_t UniformlyStochasticUniformGncMetaFactory::MinArgs() const {
        return 6;
    }

    std::size_t UniformlyStochasticUniformGncMetaFactory::MaxArgs() const {
        return 7;
    }
}