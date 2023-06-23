//
// Created by ramizouari on 23/06/23.
//
#include "choice.h"

namespace open_spiel::mpg::generator
{
    std::vector<NodeType> choose(NodeType n, int k, std::mt19937_64 &rng, bool distinct)
    {
        if(!distinct)
        {
            std::uniform_int_distribution<NodeType> dist(0,n-1);
            std::vector<NodeType> result(n);
            for(int i=0;i<k;i++)
            {
                auto j=dist(rng);
                result[i]=j;
            }
            return result;
        }
        if(k>n)
            throw std::invalid_argument("k must be less than or equal to n for distinct=true");
        else if(k<open_spiel::mpg::generator::choose_parameters::threshold*n)
        {
            std::vector<NodeType> result;
            std::unordered_set<NodeType> v_set;
            while(v_set.size()<k)
            {
                std::uniform_int_distribution<NodeType> dist(0,n-1);
                auto j=dist(rng);
                v_set.insert(j);
            }
            for(auto i:v_set)
                result.push_back(i);
            return result;
        }
        else
        {
            std::vector<NodeType> result(n);
            for(int i=0;i<n;i++)
                result[i]=i;
            return choose(result,k,rng,distinct);
        }
    }

}