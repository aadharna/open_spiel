//
// Created by ramizouari on 23/06/23.
//

#ifndef OPEN_SPIEL_CHOICE_H
#define OPEN_SPIEL_CHOICE_H
#include <vector>
#include <set>
#include <unordered_set>
#include <random>
#include <stdexcept>
#include "../mpg.h"


namespace open_spiel::mpg::generator
{
    /**
    * @brief Choose a random subset of k elements from a set v
    * @tparam T type of elements in v
    * @tparam Compare type of comparison function for elements in v
    * @tparam Allocator type of allocator for elements in v
    * @tparam RNG type of random number generator
    * @param v set of elements to choose from
    * @param k number of elements to choose
    * @param rng random number generator
    * @return a vector of k elements from v
    * */
    template<typename T,typename Compare,typename Allocator,typename RNG>
    std::set<T,Allocator> choose_set(const std::vector<T,Allocator> &v, int k, RNG && rng)
    {
        std::uniform_int_distribution<size_t> dist(0,v.size()-1);
        if (k>v.size())
            throw std::invalid_argument("k must be less than or equal to v.size()");
        std::set<size_t> result;
        while(result.size()<k)
        {
            auto j=dist(rng);
            result.insert(v[j]);
        }
        return result;
    }

    template<typename T,typename Compare,typename Allocator,typename RNG>
    std::set<T,Allocator> choose_set(const std::set<T,Compare,Allocator> &v, int k, RNG && rng)
    {
        std::vector<T,Allocator> v2(v.begin(),v.end());
        return choose_set(v2,k,std::forward<RNG>(rng));
    }

    /**
    * @brief Choose a random subset of k elements from a set v
    * @tparam T type of elements in v
    * @tparam Compare type of comparison function for elements in v
    * @tparam Allocator type of allocator for elements in v
    * @tparam RNG type of random number generator
    * @param v set of elements to choose from
    * @param k number of elements to choose
    * @param rng random number generator
    * @return a vector of k elements from v
    * @note The elements in the returned multiset can be repeated
    * */
    template<typename T,typename Compare,typename Allocator,typename RNG>
    std::multiset<T,Allocator> choose_multiset(const std::vector<T,Allocator> &v, int k, RNG && rng)
    {
        std::uniform_int_distribution<size_t> dist(0,v.size()-1);
        if (k>v.size())
            throw std::invalid_argument("k must be less than or equal to v.size() for distinct=true");
        std::multiset<T,Compare,Allocator> result;
        for(int i=0;i<k;i++)
        {
            auto j=dist(rng);
            result.insert(v[j]);
        }
        return result;
    }

    template<typename T,typename Compare,typename Allocator,typename RNG>
    std::multiset<T,Allocator> choose_multiset(const std::multiset<T,Compare,Allocator> &v, int k, RNG && rng)
    {
        std::vector<T,Allocator> v2(v.begin(),v.end());
        return choose_multiset(v2,k,std::forward<RNG>(rng));
    }

    /**
     * @brief Choose a random subset of k elements from a set v
     * @tparam T type of elements in v
     * @tparam Compare type of comparison function for elements in v
     * @tparam Allocator type of allocator for elements in v
     * @tparam RNG type of random number generator
     * @param v set of elements to choose from
     * @param k number of elements to choose
     * @param rng random number generator
     * @return a vector of k elements from v
     * */
    template<typename T,typename Hash,typename Equality,typename Allocator,typename RNG>
    std::unordered_set<T,Hash,Equality,Allocator> choose_set(const std::vector<T,Allocator> &v, int k, RNG && rng)
    {
        std::uniform_int_distribution<size_t> dist(0,v.size()-1);
        if (k>v.size())
            throw std::invalid_argument("k must be less than or equal to v.size()");
        std::unordered_set<T,Hash,Equality,Allocator> result;
        while(result.size()<k)
        {
            auto j=dist(rng);
            result.insert(v[j]);
        }
        return result;
    }

    template<typename T,typename Hash,typename Equality,typename Allocator,typename RNG>
    std::unordered_set<T,Hash,Equality,Allocator> choose_set(const std::unordered_set<T,Hash,Equality,Allocator> &v, int k, RNG && rng)
    {
        std::vector<T,Allocator> v2(v.begin(),v.end());
        return choose_set(v2,k,std::forward<RNG>(rng));
    }

    /**
    * @brief Choose a random subset of k elements from a set v
    * @tparam T type of elements in v
    * @tparam Compare type of comparison function for elements in v
    * @tparam Allocator type of allocator for elements in v
    * @tparam RNG type of random number generator
    * @param v set of elements to choose from
    * @param k number of elements to choose
    * @param rng random number generator
    * @return a vector of k elements from v
    * @note The elements in the returned multiset can be repeated
    * */
    template<typename T,typename Hash,typename Equality,typename Allocator,typename RNG>
    std::unordered_multiset<T,Hash,Equality,Allocator> choose_multiset(const std::vector<T,Allocator> &v, int k, RNG && rng)
    {
        std::uniform_int_distribution<size_t> dist(0,v.size()-1);
        if (k>v.size())
            throw std::invalid_argument("k must be less than or equal to v.size() for distinct=true");
        std::unordered_multiset<T,Hash,Equality,Allocator> result;
        for(int i=0;i<k;i++)
        {
            auto j=dist(rng);
            result.insert(v[j]);
        }
        return result;
    }

    template<typename T,typename Hash,typename Equality,typename Allocator,typename RNG>
    std::unordered_multiset<T,Hash,Equality,Allocator> choose_multiset(const std::unordered_multiset<T,Hash,Equality,Allocator> &v, int k, RNG && rng)
    {
        std::vector<T,Allocator> v2(v.begin(),v.end());
        return choose_multiset(v2,k,std::forward<RNG>(rng));
    }


    namespace choose_parameters
    {
        inline constexpr float threshold=0.5;
    }

    /**
     * @brief Choose a random subset of k elements from a set v
     * @tparam T type of elements in v
     * @tparam Allocator type of allocator for elements in v
     * @tparam RNG type of random number generator
     * @param v vector of elements to choose from
     * @param k number of elements to choose
     * @param rng random number generator
     * @param distinct if true, the elements in the returned vector are distinct (no index is repeated)
     * @return a vector of k elements from v
     * */
    template<typename T,typename Allocator,typename RNG>
    std::vector<T,Allocator> choose(const std::vector<T,Allocator> &v,int k, RNG && rng,bool distinct=true)
    {

        std::uniform_int_distribution<size_t> dist(0,v.size()-1);
        if(distinct)
        {
            if(k>v.size())
                throw std::invalid_argument("k must be less than or equal to v.size() for distinct=true");
            std::vector<T,Allocator> result;
            std::unordered_set<size_t> v_set;
            if(k<= choose_parameters::threshold * v.size()) while(v_set.size()<k)
                {
                    auto j=dist(rng);
                    v_set.insert(j);
                }
            else
            {
                for(int i=0;i<v.size();i++)
                    v_set.insert(i);
                while(v_set.size() > k)
                {
                    auto j=dist(rng);
                    v_set.erase(j);
                }
            }
            for(auto i:v_set)
                result.push_back(v[i]);
            return result;
        }
        else
        {
            std::vector<T,Allocator> result;
            for(int i=0;i<k;i++)
            {
                auto j=dist(rng);
                result.push_back(v[j]);
            }
            return result;
        }
    }

    std::vector<NodeType> choose(NodeType n,int k, std::mt19937_64 & rng,bool distinct=true);

}

#endif //OPEN_SPIEL_CHOICE_H
