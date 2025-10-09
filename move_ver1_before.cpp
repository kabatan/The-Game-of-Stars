#include <pybind11/pybind11.h>

#include <pybind11/stl.h>

#include <iostream>

#include <vector>

#include <cmath>

#include <random>

#include <queue>

#include <chrono>

#include <map>

#include <tuple>

#include <algorithm>

#include <functional>



constexpr float time_out = 0.1;

constexpr u_int16_t s_num = 50;

constexpr u_int16_t p_num = 6;



float alpha = 0.203199487107651f;

float beta_ = 7.331854385889875f;

float gamma_ = 0.675927893550303f;

float rho = 0.8283434619125477f;

float q = 0.5575421046543976f;

float p = 0.5825121310354511f;

u_int16_t simu_num = 14;

float best_bonus = 4.861104130096559f;

float win_bonus = 4.888780475900189f;

float best_rho = 0.9246628156218114f;

int loop_num = 57;

int max_cluster_size = 7;



struct point {

    int r,c;

    bool operator==(const point& other) const {

        return (r==other.r && c==other.c);

    }

    int operator-(const point& other) const {

        return (abs(r-other.r)+abs(c-other.c));

    }

    point operator+(const point& other) const {

        return point{r+other.r,c+other.c};

    }

};



struct bot_state {

    std::vector<point> stars;

    std::vector<std::vector<int>> star_graph;

    std::vector<std::vector<float>> dist_inv;

    std::vector<std::vector<std::vector<float>>> phe;

    std::vector<bool> valid_s;

    int valid_s_num;

    std::vector<float> best_score;

    std::vector<std::queue<int>> best_route;

    float rho_inv;

    std::vector<int> cluster_valid;

    std::vector<int> cluster;

};



struct sim_res {

    std::vector<float> score;

    std::vector<std::queue<int>> route;

};





struct kmean_ref {

    std::vector<std::vector<int>> clusters;

    int max_cluster_size;

};



bot_state bot;

std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());



namespace Random {

    float uniform(float min, float max) {

        static std::uniform_real_distribution<float> dis;

        using param_t = std::uniform_real_distribution<float>::param_type;

        return dis(rng, param_t{min, max});

    }

    int randint(int min, int max) {

        if (min > max) return min;

        static std::uniform_int_distribution<int> dis;

        using param_t = std::uniform_int_distribution<int>::param_type;

        return dis(rng, param_t{min, max});

    }

}



float phe_max(const int& p) {

    return bot.rho_inv*pow(bot.best_score[p], gamma_);

}



float phe_min(const int& p) {

    float tmp = (float)std::max(3, bot.valid_s_num);

    float p_max = phe_max(p);

    float p_05_pow = pow(0.05, 1.0/tmp);

    return (1.0-p_05_pow)*p_max/((tmp/2.0-1.0)*p_05_pow);

}



int choose_v(const int& p, const int& v, const std::vector<bool>& rest_s, const std::vector<int>& cluster_sizes) {

    static thread_local std::vector<int> c_idx;

    static thread_local std::vector<float> c_value;

    c_idx.clear();

    c_value.clear();

    c_idx.reserve(s_num);

    c_value.reserve(s_num);



    double sum = 0.0;

    for (int i=0; i<s_num; ++i) {

        if (rest_s[i]) {

            float tau = bot.phe[p][v][i];

            float value = std::pow(tau, alpha) * bot.dist_inv[v][i] * pow(cluster_sizes[bot.cluster[i]], p);

            c_idx.emplace_back(i);

            c_value.emplace_back(value);

            sum += value;

        }

    }

    if (sum <= 0.0) return -1;



    double r = Random::uniform(0.0f, (float)sum);

    double cumulative_value = 0.0;

    for (size_t i=0; i<c_idx.size(); ++i) {

        cumulative_value += c_value[i];

        if (cumulative_value >= r) return c_idx[i];

    }

    return c_idx.back();

}



std::vector<float> sim_best(const std::vector<point>& players, int p, const std::queue<int>& route) {

    std::vector<float> score(p_num, 0);

    std::vector<std::queue<int>> routes = bot.best_route;

    routes[p] = std::move(route);

    std::vector<bool> rest_s = bot.valid_s;

    std::priority_queue<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>, std::greater<std::tuple<int, int, int>>> game_q;



    for (int i=0; i<p_num; ++i) {

        game_q.emplace(0,i,i+s_num);

    }

   

    std::vector<std::vector<int>> new_positions(s_num+p_num);

    std::vector<std::pair<int, int>> arrivals;

    arrivals.reserve(p_num);



    for (size_t i = 0; i < new_positions.size(); ++i) new_positions[i].reserve(2);



    while (!game_q.empty()) {

        int current = std::get<0>(game_q.top());

       

        arrivals.clear();

        while (!game_q.empty() && std::get<0>(game_q.top()) == current) {

            auto [time, p_idx, v_idx] = game_q.top();

            game_q.pop();

            arrivals.emplace_back(p_idx, v_idx);

        }



        for (const auto& arrival : arrivals) {

            new_positions[arrival.second].emplace_back(arrival.first);

        }



        for (const auto& arrival : arrivals) {

            int v_idx = arrival.second;

            if (v_idx < s_num && rest_s[v_idx] && !new_positions[v_idx].empty()) {

                auto& players_at_v = new_positions[v_idx];

                for (const auto& p_idx : players_at_v) {

                    score[p_idx]+=1.0/players_at_v.size();

                }

                rest_s[v_idx] = false;

                players_at_v.clear();

            }

        }

        for (const auto& arrival : arrivals) {

            new_positions[arrival.second].clear();

        }

        for (const auto& arrival : arrivals) {

            int p_idx = arrival.first;

            int v_from = arrival.second;

            if (!routes[p_idx].empty()) {

                int to = routes[p_idx].front();

                routes[p_idx].pop();

                game_q.emplace(std::make_tuple(current+bot.star_graph[v_from][to], p_idx, to));

            }

        }



    }

    return score;

}



template<typename T>

sim_res sim_com(const T& choose) {

    sim_res result;

    result.score.assign(p_num, 0);

    result.route.resize(p_num);



    std::vector<bool> rest_s = bot.valid_s;

    std::vector<int> cluster_sizes = bot.cluster_valid;

    std::priority_queue<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>, std::greater<std::tuple<int, int, int>>> game_q;



    for (int i = 0; i < p_num; ++i) {

        game_q.emplace(0,i,i+s_num);

    }

   

    std::vector<std::vector<int>> new_positions(s_num+p_num);

    std::vector<std::pair<int, int>> arrivals;

    arrivals.reserve(p_num);



    for (size_t i = 0; i < new_positions.size(); ++i) new_positions[i].reserve(2);



    while (!game_q.empty()) {

        int current = std::get<0>(game_q.top());

       

        arrivals.clear();

        while (!game_q.empty() && std::get<0>(game_q.top()) == current) {

            auto [time, p_idx, v_idx] = game_q.top();

            game_q.pop();

            arrivals.emplace_back(p_idx, v_idx);

        }



        for (const auto& arrival : arrivals) {

            new_positions[arrival.second].emplace_back(arrival.first);

        }



        for (const auto& arrival : arrivals) {

            int v_idx = arrival.second;

            if (v_idx < s_num && rest_s[v_idx] && !new_positions[v_idx].empty()) {

                auto& players_at_v = new_positions[v_idx];

                for (const auto& p_idx : players_at_v) {

                    result.score[p_idx]+=1.0/players_at_v.size();

                }

                rest_s[v_idx] = false;

                cluster_sizes[bot.cluster[v_idx]]--;

                players_at_v.clear();

            }

        }

        for (const auto& arrival : arrivals) {

            new_positions[arrival.second].clear();

        }

        for (const auto& arrival : arrivals) {

            int to = choose(arrival.first, arrival.second, rest_s, cluster_sizes);

            if (to != -1) {

                game_q.emplace(current+bot.star_graph[arrival.second][to], arrival.first, to);

                result.route[arrival.first].emplace(to);

            }

        }

    }

    return result;

}



std::pair<int,int> move_to(const point& pos,const int& s) {

    if (pos.r<bot.stars[s].r) {

        return std::make_pair(pos.r+1,pos.c);

    }

    else if (pos.r>bot.stars[s].r) {

        return std::make_pair(pos.r-1,pos.c);

    }

    else if (pos.c<bot.stars[s].c) {

        return std::make_pair(pos.r,pos.c+1);

    }

    else if (pos.c>bot.stars[s].c) {

        return std::make_pair(pos.r,pos.c-1);

    }

    else {

        return std::make_pair(pos.r,pos.c);

    }

}



kmean_ref kmeans(const int& k) {

    int minx=1000;

    int maxx=-1;

    int miny=1000;

    int maxy=-1;

    for (int i=0;i<s_num;++i) {

        if (bot.stars[i].r<minx) minx=bot.stars[i].r;

        if (bot.stars[i].r>maxx) maxx=bot.stars[i].r;

        if (bot.stars[i].c<miny) miny=bot.stars[i].c;

        if (bot.stars[i].c>maxy) maxy=bot.stars[i].c;

    }



    std::vector<point> centers(k);

    std::vector<int> sizes(k,0);

    std::vector<point> cluster_sum(k);

    std::vector<int> starowner(s_num,0);

    for (int i=0;i<k;++i) {

        centers[i]=point{Random::randint(minx,maxx),Random::randint(miny,maxy)};

    }

    for (int iter=0;iter<loop_num;++iter) {

        std::fill(sizes.begin(), sizes.end(), 0);

        std::fill(cluster_sum.begin(), cluster_sum.end(), point{0, 0});

        std::fill(starowner.begin(), starowner.end(), 0);

        for (int i=0; i<s_num; ++i) {

            int mindist=1000;

            int minj=-1;

            for (int j=0; j<k; ++j) {

                int d = bot.stars[i]-centers[j];

                if (d<mindist) {

                    mindist=d;

                    minj=j;

                }

            }

            starowner[i]=minj;

            sizes[minj]++;

            cluster_sum[minj]=cluster_sum[minj]+bot.stars[i];

        }

        for (int i=0;i<k;++i) {

            if (sizes[i]>0) {

                centers[i]=point{cluster_sum[i].r/sizes[i],cluster_sum[i].c/sizes[i]};

            }

        }

    }

    kmean_ref res;

    res.clusters.assign(k,std::vector<int>());

    res.max_cluster_size=0;

    for (int i=0;i<s_num;++i) {

        res.clusters[starowner[i]].emplace_back(i);

    }

    for (int i=0;i<k;++i) {

        if (res.clusters[i].size()<2) continue;

        for (int j=0;j<res.clusters[i].size()-1;++j) {

            for (int l=j+1;l<res.clusters[i].size();++l) {

                int d = bot.stars[res.clusters[i][j]] - bot.stars[res.clusters[i][l]];

                if (d>res.max_cluster_size) {res.max_cluster_size=d;}

            }

        }

    }

    return res;

}



bool is_new_game(const std::vector<point>& star_list) {

    if (bot.valid_s.empty()) return true;

    for (int i=0; i<star_list.size(); ++i) {

        bool flag=true;

        for (int j=0; j<s_num; ++j) {

            if (bot.valid_s[j] && bot.stars[j]==star_list[i]) {

                flag=false; break;

            }

        }

        if (flag) {return true;}

    }

    return false;

}



void init(const std::vector<point>& players, const std::vector<point>& star_list) {

    bot.stars = star_list;

    bot.star_graph.assign(s_num+p_num,std::vector<int>(s_num+p_num,0));

    bot.dist_inv.assign(s_num+p_num,std::vector<float>(s_num+p_num,0.0));

    bot.phe.assign(p_num,std::vector<std::vector<float>>(s_num+p_num,std::vector<float>(p_num+s_num,0)));

    bot.valid_s.assign(s_num, true);

    bot.valid_s_num =s_num;

    bot.rho_inv=1.0/rho;



   

    for (int i=0; i<s_num+p_num; ++i) {

        point p1 = i<s_num ? star_list[i] : players[i-s_num];

        for (int j=0; j<s_num+p_num; ++j) {

            point p2 = j<s_num ? star_list[j] : players[j-s_num];

            int d= p1 - p2;

            bot.star_graph[i][j]=d;

            bot.dist_inv[i][j]=pow(q/((float)d+0.001),beta_);

        }

    }



    int cluster_num=5;

    kmean_ref res = kmeans(cluster_num);

    while (res.max_cluster_size>max_cluster_size && cluster_num<20) {

        cluster_num++;

        res = kmeans(cluster_num);

    }

    bot.cluster_valid.assign(cluster_num,0);

    bot.cluster.assign(s_num,-1);

    for (int i=0; i<cluster_num; ++i) {

        bot.cluster_valid[i]=res.clusters[i].size();

        for (int j=0; j<res.clusters[i].size(); ++j) {

            bot.cluster[res.clusters[i][j]]=i;

        }

    }



    auto greedy_choose = [&](int p_idx, int v_from, const std::vector<bool>& rest_s, const std::vector<int>&) {

        int to = -1;

        int min_d = std::numeric_limits<int>::max();

        for (int i = 0; i < s_num; ++i) {

            if (rest_s[i] && bot.star_graph[v_from][i] < min_d) {

                min_d = bot.star_graph[v_from][i];

                to = i;

            }

        }

        return to;

    };



    sim_res tmp=sim_com(greedy_choose);

    bot.best_score=std::move(tmp.score);

    bot.best_route=std::move(tmp.route);



    for (int i=0; i<p_num; ++i) {

        float p = phe_min(i);

        for (int j=0; j<p_num+s_num; ++j) {

            for (int k=0; k<s_num+p_num; ++k) {

                bot.phe[i][j][k]=p;

            }

        }

    }

}



void update(const std::vector<point>& players) {

    for (int i=0; i<p_num; ++i) {

        for (int j=0; j<s_num; ++j) {

            int d = players[i]-bot.stars[j];

            bot.star_graph[i+s_num][j]=d;

            bot.star_graph[j][i+s_num]=d;

            bot.dist_inv[i+s_num][j]=pow(q/((float)d+0.001),beta_);

            bot.dist_inv[j][i+s_num]=bot.dist_inv[i+s_num][j];

            if (players[i]==bot.stars[j] && bot.valid_s[j]) {

                bot.valid_s[j] = false;

                bot.valid_s_num--;

                bot.cluster_valid[bot.cluster[j]]--;

            }

        }



        if (!bot.best_route[i].empty() && players[i]==bot.stars[bot.best_route[i].front()]) {

            bot.best_score[i]-=1;

            bot.best_route[i].pop();

        }

    }

    bot.best_score=sim_best(players,0,bot.best_route[0]);

}



std::pair<int,int> move_ver1(const std::vector<std::pair<int,int>>& players_, const std::vector<std::pair<int,int>>& star_list_, const int& p_idx) {

    std::vector<point> players(players_.size());

    std::vector<point> star_list(star_list_.size());

    for (size_t i=0; i<players_.size(); ++i) {

        players[i]=point{players_[i].first,players_[i].second};

    }

    for (size_t i=0; i<star_list.size(); ++i) {

        star_list[i]=point{star_list_[i].first,star_list_[i].second};

    }

   

    if (is_new_game(star_list)) {

        init(players,star_list);

    }

    else {update(players);}



    auto start_time=std::chrono::high_resolution_clock::now();

    auto end_time = start_time + std::chrono::duration<double>(time_out * 0.8);



    auto aco_choose = [&](int p_idx, int v_from, const std::vector<bool>& rest_s, const std::vector<int>& cluster_sizes) {

        return choose_v(p_idx, v_from, rest_s, cluster_sizes);

    };



    int t=0;



   

    while (std::chrono::high_resolution_clock::now() < end_time) {

        std::vector<float> good_score(p_num,0);

        std::vector<std::queue<int>> good_route(p_num);



        for (int i=0; i<simu_num; ++i) {

            sim_res tmp = sim_com(aco_choose);

            float win_score = *std::max_element(tmp.score.begin(), tmp.score.end());

            for (int j=0; j<p_num; ++j) {

                if (tmp.score[j]==win_score && win_score>0) {

                    tmp.score[j] += win_bonus;

                }

                if (tmp.score[j]>good_score[j]) {

                    good_score[j]=tmp.score[j];

                    good_route[j]=tmp.route[j];

                }

            }

            t++;

        }



        for (int i=0; i<p_num; ++i) {

            if (good_route[i].empty()) continue;

            std::vector<float> test_score=sim_best(players,i,good_route[i]);

            float win_score = *std::max_element(test_score.begin(), test_score.end());

            if (test_score[i]==win_score && win_score>0) {

                test_score[i] += win_bonus;

            }

            bool is_best=false;

            if (test_score[i]>bot.best_score[i]) {

                bot.best_score[i]=test_score[i];

                bot.best_route[i]=good_route[i];

                is_best=true;

            }

            if (test_score[i]>0) {

                std::queue<int> route = good_route[i];

                int from_node = i + s_num;

                float delta = pow(test_score[i], gamma_);

                int cnt=0;

                int len = route.size();

                while (!route.empty()) {

                    int to_node = route.front();

                    route.pop();

                    if (is_best) {

                        float delta_=(test_score[i]/(bot.best_score[i]+0.01))*pow(best_bonus,1.0-best_rho*((float)cnt/(float)len));

                        bot.phe[i][from_node][to_node] += delta_*delta;

                        bot.phe[i][to_node][from_node] += delta_*delta;

                    }

                    else {

                        bot.phe[i][from_node][to_node] += delta;

                        bot.phe[i][to_node][from_node] += delta;

                    }

                    bot.phe[i][from_node][to_node] = std::min(phe_max(i), bot.phe[i][from_node][to_node]);

                    bot.phe[i][to_node][from_node] = std::min(phe_max(i), bot.phe[i][to_node][from_node]);

                    from_node = to_node;

                    cnt++;

                }

            }

        }

    }



    for (int i=0; i<p_num; ++i) {

        for (int j=0; j<p_num+s_num; ++j) {

            for (int k=0; k<p_num+s_num; ++k) {

                bot.phe[i][j][k] *= rho;

                bot.phe[i][j][k] = std::max(phe_min(i),bot.phe[i][j][k]);

            }

        }

    }



    //std::cout<<t<<std::endl;



    if (!bot.best_route[p_idx].empty()) {

        return move_to(players[p_idx], bot.best_route[p_idx].front());

    } else {

        return std::make_pair(players[p_idx].r, players[p_idx].c);

    }

}



void set_parameters(const pybind11::dict& params) {

    if (params.contains("alpha")) alpha = params["alpha"].cast<float>();

    if (params.contains("beta_")) beta_ = params["beta_"].cast<float>();

    if (params.contains("gamma_")) gamma_ = params["gamma_"].cast<float>();

    if (params.contains("rho")) rho = params["rho"].cast<float>();

    if (params.contains("q")) q = params["q"].cast<float>();

    if (params.contains("p")) p = params["p"].cast<float>();

    if (params.contains("simu_num")) simu_num = params["simu_num"].cast<u_int16_t>();

    if (params.contains("best_bonus")) best_bonus = params["best_bonus"].cast<float>();

    if (params.contains("win_bonus")) win_bonus = params["win_bonus"].cast<float>();

    if (params.contains("best_rho")) best_rho = params["best_rho"].cast<float>();

    if (params.contains("loop_num")) loop_num = params["loop_num"].cast<int>();

    if (params.contains("max_cluster_size")) max_cluster_size = params["max_cluster_size"].cast<int>();

}



PYBIND11_MODULE(move_ver1, m) {

    m.def("move_ver1", &move_ver1, "",

        pybind11::arg("players_"), pybind11::arg("star_list_"), pybind11::arg("p_idx"));

    m.def("set_parameters", &set_parameters);

}