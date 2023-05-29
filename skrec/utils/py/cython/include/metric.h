/*
@author: Zhongchuan Sun
@email: zhongchuansun@gmail.com
*/
#ifndef METRIC_H
#define METRIC_H

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <algorithm>    // std::min, std::max

using std::vector;
using std::unordered_set;
using std::unordered_map;


void precision(const vector<int> &rank, const unordered_set<int> &truth, float *result_pt)
{
    float hits = 0.0;
    for(unsigned int i=0; i<rank.size(); i++)
    {
        if(truth.find(rank[i]) != truth.end())
        {
            hits += 1.0;
        }
        result_pt[i] = hits / (i+1);
    }
}


void recall(const vector<int> &rank, const unordered_set<int> &truth, float *result_pt)
{
    float hits = 0.0;
    float truth_len = std::max(static_cast<int>(truth.size()), 1);
    for(unsigned int i=0; i<rank.size(); i++)
    {
        if(truth.find(rank[i]) != truth.end())
        {
            hits += 1.0;
        }
        result_pt[i] = hits / truth_len;
    }
}


void ap(const vector<int> &rank, const unordered_set<int> &truth, float *result_pt)
{
    float hits = 0.0;
    float pre = 0.0;
    float sum_pre = 0.0;
    float denominator = 1.0;
    int truth_len = std::max(static_cast<int>(truth.size()), 1);
    for(unsigned int i=0; i<rank.size(); i++)
    {
        if(truth.find(rank[i]) != truth.end())
        {
            hits += 1.0;
            pre = hits / (i+1);
            sum_pre += pre;
        }
        denominator = std::min(truth_len, static_cast<int>(i+1));
        result_pt[i] = sum_pre/denominator;
    }
}


void ndcg(const vector<int> &rank, const unordered_set<int> &truth, float *result_pt)
{
    float iDCG = 0.0;
    float DCG = 0.0;
    unsigned int truth_len = std::max(static_cast<int>(truth.size()), 1);
    for(unsigned int i=0; i<rank.size(); i++)
    {
        if(truth.find(rank[i]) != truth.end())
        {
            DCG += 1.0/log2(i+2);
        }
        if(i<truth_len)
        {
            iDCG += 1.0/log2(i+2);
        }
        result_pt[i] = DCG/iDCG;
    }
}


void mrr(const vector<int> &rank, const unordered_set<int> &truth, float *result_pt)
{
    float rr = 0;
    for(unsigned int i=0; i<rank.size(); i++)
    {
        if(truth.find(rank[i]) != truth.end())
        {
            rr = 1.0/(i+1);
            for(auto j=i; j<rank.size(); j++)
            {
                result_pt[j] = rr;
            }
            break;
        }
        else
        {
            rr = 0.0;
            result_pt[i] =rr;
        }
    }
}


typedef void(*metric_fun)(const vector<int> &, const unordered_set<int> &, float *);
unordered_map<int, metric_fun> metric_dict = {{1, precision},
                                              {2, recall},
                                              {3, ap},
                                              {4, ndcg},
                                              {5, mrr}
                                             };

#endif
