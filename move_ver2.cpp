#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <queue>
#include <chrono>
#include <map>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <functional>
#include <numeric> 
#include <fstream>
#include <string>

#define ALWAYS_INLINE __attribute__((always_inline))

constexpr float time_out = 1.0f;
constexpr u_int16_t s_num = 50;
constexpr u_int16_t p_num = 6;
constexpr int multi_num = 1;
const std::string state_filename = "bot_state_ver2.dat";

float alpha = 0.3218878580081168f;
float beta_ = 6.495731879583197f;
float gamma_ = 4.887159350622191f;
float rho = 0.9077788573984817f;
float q = 1.6989696503971095f;
float p = 1.5610042732763938f;
int simu_num = 36;
float best_bonus = 1.3397094623377321f;
float win_bonus = 9.796927129775883f;
float best_rho = 0.03901156174468283f;
int loop_num = 192;
int max_cluster_size = 7;
float deposit_budget = 296.58289926911755f;


struct point {
    int r,c;
    inline constexpr bool operator==(const point& other) const {
        return (r==other.r && c==other.c);
    }
    inline constexpr int operator-(const point& other) const {
        return (std::abs(r-other.r)+std::abs(c-other.c));
    }
    inline constexpr point operator+(const point& other) const {
        return point{r+other.r,c+other.c};
    }
};

struct bot_state {
    std::vector<point> stars;
    std::vector<std::vector<int>> star_graph;
    std::vector<std::vector<float>> dist_inv;
    std::vector<std::vector<std::vector<float>>> phe;
    std::vector<uint8_t> valid_s;
    int valid_s_num;
    std::vector<float> best_score;
    std::vector<std::queue<int>> best_route;
    float rho_inv;
    std::vector<int> cluster_valid;
    std::vector<int> cluster;
    std::vector<float> p_p;
};

struct sim_res {
    std::vector<float> score;
    std::vector<std::queue<int>> route;
};

struct kmean_ref {
    std::vector<std::vector<int>> clusters;
    int max_cluster_size;
};

struct XorShift {
    uint32_t x, y, z, w;
    XorShift() : x(123456789), y(362436069), z(521288629), w(88675123) {}
    XorShift(uint32_t seed) : x(seed), y(362436069), z(521288629), w(88675123) {}
    ALWAYS_INLINE uint32_t next() {
        uint32_t t = x ^ (x << 11);
        x = y; y = z; z = w;
        return w = (w ^ (w >> 19)) ^ (t ^ (t >> 8));
    }
    ALWAYS_INLINE float uniform(float min, float max) {
        return min + (max - min) * (next() / 4294967296.0f);
    }
    ALWAYS_INLINE int randint(int min, int max) {
        if (min > max) return min;
        long long range = (long long)max - min + 1;
        return min + (int)((unsigned long long)next() * range >> 32);
    }
};

inline float phe_max(const bot_state& bot, const int& p) {
    return bot.rho_inv * powf(bot.best_score[p], gamma_);
}

inline float phe_min(const bot_state& bot, const int& p) {
    float tmp = (float)std::max(3, bot.valid_s_num);
    float p_max = phe_max(bot, p);
    float p_05_pow = powf(0.05f, 1.0f / tmp);
    return (1.0f - p_05_pow) * p_max / ((tmp / 2.0f - 1.0f) * p_05_pow);
}

int choose_v(const bot_state& bot, XorShift& rng, const int& p, const int& v, const uint8_t* rest_s, const std::vector<int>& cluster_sizes) {
    static thread_local std::vector<int> c_idx;
    static thread_local std::vector<float> c_value;
    c_idx.clear();
    c_value.clear();

    double sum = 0.0;
    const float* phe_ptr = bot.phe[p][v].data();
    const float* dist_inv_ptr = bot.dist_inv[v].data();

    for (int i = 0; i < s_num; ++i) {
        if (rest_s[i]) {
            float value = powf(phe_ptr[i], alpha) * dist_inv_ptr[i] * bot.p_p[cluster_sizes[bot.cluster[i]]];
            c_idx.emplace_back(i);
            c_value.emplace_back(value);
            sum += value;
        }
    }
    if (sum <= 1e-9) return -1; 

    float r = rng.uniform(0.0f, (float)sum);
    float cumulative_value = 0.0f;
    for (size_t i = 0; i < c_idx.size(); ++i) {
        cumulative_value += c_value[i];
        if (cumulative_value >= r) return c_idx[i];
    }
    return c_idx.back();
}

void sim_best(const bot_state& bot, std::vector<float>& score, const std::vector<point>& players, int p, const std::queue<int>& route) {
    score.assign(p_num, 0);
    std::vector<std::queue<int>> routes = bot.best_route;
    routes[p] = route; 
    
    std::vector<uint8_t> rest_s = bot.valid_s;
    
    std::priority_queue<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>, std::greater<std::tuple<int, int, int>>> game_q;

    for (int i = 0; i < p_num; ++i) {
        game_q.emplace(0, i, i+s_num);
    }
    
    static thread_local std::vector<std::vector<int>> new_positions(s_num + p_num);
    static thread_local std::vector<int> used_indices;
    used_indices.clear();

    static thread_local std::vector<std::pair<int, int>> arrivals;
    arrivals.reserve(p_num);

    while (!game_q.empty()) {
        int current = std::get<0>(game_q.top());
        
        arrivals.clear();
        while (!game_q.empty() && std::get<0>(game_q.top()) == current) {
            arrivals.emplace_back(std::get<1>(game_q.top()), std::get<2>(game_q.top()));
            game_q.pop();
        }
        
        for (const auto& arrival : arrivals) {
            auto& pos_vec = new_positions[arrival.second];
            if (pos_vec.empty()) used_indices.push_back(arrival.second);
            pos_vec.emplace_back(arrival.first);
        }

        for (const auto& arrival : arrivals) {
            int v_idx = arrival.second;
            if (v_idx < s_num && rest_s[v_idx]) {
                auto& players_at_v = new_positions[v_idx];
                const size_t num_players_at_v = players_at_v.size();
                if (num_players_at_v > 0) {
                    float score_to_add = 1.0f / num_players_at_v;
                    for (const int p_idx : players_at_v) {
                        score[p_idx] += score_to_add;
                    }
                    rest_s[v_idx] = 0;
                }
            }
        }
        
        for (const int idx : used_indices) new_positions[idx].clear();
        used_indices.clear();

        for (const auto& arrival : arrivals) {
            int p_idx = arrival.first;
            int v_from = arrival.second;
            if (!routes[p_idx].empty()) {
                int to = routes[p_idx].front();
                routes[p_idx].pop();
                game_q.emplace(current + bot.star_graph[v_from][to], p_idx, to);
            }
        }
    }
}

template<typename T>
void sim_com(const bot_state& bot, XorShift& rng, sim_res& result, const T& choose) {
    result.score.assign(p_num, 0);
    for(auto& q : result.route) while(!q.empty()) q.pop();

    std::vector<uint8_t> rest_s = bot.valid_s;
    std::vector<int> cluster_sizes = bot.cluster_valid;
    std::priority_queue<std::tuple<int, int, int>, std::vector<std::tuple<int, int, int>>, std::greater<std::tuple<int, int, int>>> game_q;

    for (int i = 0; i < p_num; ++i) {
        game_q.emplace(0, i, i+s_num);
    }
    
    static thread_local std::vector<std::vector<int>> new_positions(s_num + p_num);
    static thread_local std::vector<int> used_indices;
    used_indices.clear();

    static thread_local std::vector<std::pair<int, int>> arrivals;
    arrivals.reserve(p_num);
    
    while (!game_q.empty()) {
        int current = std::get<0>(game_q.top());
        
        arrivals.clear();
        while (!game_q.empty() && std::get<0>(game_q.top()) == current) {
            arrivals.emplace_back(std::get<1>(game_q.top()), std::get<2>(game_q.top()));
            game_q.pop();
        }

        for (const auto& arrival : arrivals) {
            auto& pos_vec = new_positions[arrival.second];
            if (pos_vec.empty()) used_indices.push_back(arrival.second);
            pos_vec.emplace_back(arrival.first);
        }

        for (const auto& arrival : arrivals) {
            int v_idx = arrival.second;
            if (v_idx < s_num && rest_s[v_idx]) {
                auto& players_at_v = new_positions[v_idx];
                const size_t num_players_at_v = players_at_v.size();
                if (num_players_at_v > 0) {
                    float score_to_add = 1.0f / num_players_at_v;
                    for (const int p_idx : players_at_v) {
                        result.score[p_idx] += score_to_add;
                    }
                    rest_s[v_idx] = 0;
                    cluster_sizes[bot.cluster[v_idx]]--;
                }
            }
        }
        
        for (const int idx : used_indices) new_positions[idx].clear();
        used_indices.clear();

        for (const auto& arrival : arrivals) {
            int to = choose(bot, rng, arrival.first, arrival.second, rest_s.data(), cluster_sizes);
            if (to != -1) {
                game_q.emplace(current + bot.star_graph[arrival.second][to], arrival.first, to);
                result.route[arrival.first].emplace(to);
            }
        }
    }
}

inline std::pair<int,int> move_to(const bot_state& bot, const point& pos,const int& s) {
    const point& target = bot.stars[s];
    if (pos.r < target.r) return {pos.r + 1, pos.c};
    if (pos.r > target.r) return {pos.r - 1, pos.c};
    if (pos.c < target.c) return {pos.r, pos.c + 1};
    if (pos.c > target.c) return {pos.r, pos.c - 1};
    return {pos.r, pos.c};
}

kmean_ref kmeans(const bot_state& bot, XorShift& rng, const int& k) {
    int minx=1000, maxx=-1, miny=1000, maxy=-1;
    for (int i=0; i<s_num; ++i) {
        minx = std::min(minx, bot.stars[i].r);
        maxx = std::max(maxx, bot.stars[i].r);
        miny = std::min(miny, bot.stars[i].c);
        maxy = std::max(maxy, bot.stars[i].c);
    }

    std::vector<point> centers(k);
    for (int i=0; i<k; ++i) {
        centers[i] = {rng.randint(minx, maxx), rng.randint(miny, maxy)};
    }
    
    std::vector<int> starowner(s_num);
    std::vector<point> cluster_sum(k);
    std::vector<int> sizes(k);

    for (int iter=0; iter<loop_num; ++iter) {
        std::fill(cluster_sum.begin(), cluster_sum.end(), point{0, 0});
        std::fill(sizes.begin(), sizes.end(), 0);
        
        for (int i=0; i<s_num; ++i) {
            int mindist = 1000; 
            int minj = -1;
            for (int j=0; j<k; ++j) {
                int d = bot.stars[i] - centers[j];
                if (d < mindist) {
                    mindist = d;
                    minj = j;
                }
            }
            starowner[i] = minj;
            sizes[minj]++;
            cluster_sum[minj] = cluster_sum[minj] + bot.stars[i];
        }
        for (int i=0; i<k; ++i) {
            if (sizes[i] > 0) {
                centers[i] = {cluster_sum[i].r / sizes[i], cluster_sum[i].c / sizes[i]};
            }
        }
    }
    
    kmean_ref res;
    res.clusters.resize(k);
    for(auto& c : res.clusters) c.clear();

    for (int i=0; i<s_num; ++i) {
        res.clusters[starowner[i]].emplace_back(i);
    }
    
    res.max_cluster_size = 0;
    for (int i=0; i<k; ++i) {
        if (res.clusters[i].size() < 2) continue;
        for (size_t j=0; j<res.clusters[i].size(); ++j) {
            for (size_t l=j+1; l<res.clusters[i].size(); ++l) {
                res.max_cluster_size = std::max(res.max_cluster_size, bot.stars[res.clusters[i][j]] - bot.stars[res.clusters[i][l]]);
            }
        }
    }
    return res;
}

bool is_new_game(const bot_state& bot, const std::vector<point>& star_list) {
    if (bot.valid_s.empty()) return true;
    for (const auto& new_star : star_list) {
        bool found = false;
        for (int j = 0; j < s_num; ++j) {
            if (bot.valid_s[j] && bot.stars[j] == new_star) {
                found = true;
                break;
            }
        }
        if (!found) return true;
    }
    return false;
}

void init(bot_state& bot, XorShift& rng, const std::vector<point>& players, const std::vector<point>& star_list) {
    bot.stars = star_list;
    bot.star_graph.assign(s_num+p_num, std::vector<int>(s_num+p_num, 0));
    bot.dist_inv.assign(s_num+p_num, std::vector<float>(s_num+p_num, 0.0f));
    bot.phe.assign(p_num, std::vector<std::vector<float>>(s_num+p_num, std::vector<float>(p_num+s_num, 0.0f)));
    
    bot.valid_s.assign(s_num, 1); 
    bot.valid_s_num = s_num;
    bot.rho_inv = 1.0f / rho;
    bot.p_p.assign(s_num + 1, 0.0f);
    
    for (int i=0; i < s_num+p_num; ++i) {
        point p1 = i < s_num ? star_list[i] : players[i-s_num];
        for (int j=i; j < s_num+p_num; ++j) {
            point p2 = j < s_num ? star_list[j] : players[j-s_num];
            int d = p1 - p2;
            bot.star_graph[i][j] = bot.star_graph[j][i] = d;
            bot.dist_inv[i][j] = bot.dist_inv[j][i] = powf(q / ((float)d + 0.001f), beta_);
        }
    }

    int cluster_num=5;
    kmean_ref res = kmeans(bot, rng, cluster_num);
    while (res.max_cluster_size > max_cluster_size && cluster_num < 20) {
        cluster_num++;
        res = kmeans(bot, rng, cluster_num);
    }
    bot.cluster_valid.assign(cluster_num, 0);
    bot.cluster.assign(s_num, -1);
    for (int i=0; i<cluster_num; ++i) {
        bot.cluster_valid[i] = res.clusters[i].size();
        for (const auto& star_idx : res.clusters[i]) {
            bot.cluster[star_idx] = i;
        }
    }

    auto greedy_choose = [&](const bot_state& bot, XorShift& rng, int, int v_from, const uint8_t* rest_s, const std::vector<int>&) {
        int to = -1;
        int min_d = 10000;
        for (int i = 0; i < s_num; ++i) {
            if (rest_s[i] && bot.star_graph[v_from][i] < min_d) {
                min_d = bot.star_graph[v_from][i];
                to = i;
            }
        }
        return to;
    };
    
    sim_res tmp;
    tmp.route.resize(p_num);
    sim_com(bot, rng, tmp, greedy_choose);
    bot.best_score = std::move(tmp.score);
    bot.best_route = std::move(tmp.route);
    
    for (int i = 0; i<p_num; ++i) {
        float p_min = phe_min(bot, i);
        for (int j=0; j<p_num+s_num; ++j) {
            std::fill(bot.phe[i][j].begin(), bot.phe[i][j].end(), p_min);
        }
    }

    for (int i = 0; i <= s_num; ++i) {
        bot.p_p[i] = powf((float)i, p);
    }
}

void update(bot_state& bot, const std::vector<point>& players) {
    for (int i=0; i<p_num; ++i) {
        for (int j=0; j<s_num; ++j) {
            int d = players[i] - bot.stars[j];
            bot.star_graph[i+s_num][j] = bot.star_graph[j][i+s_num] = d;
            float inv_d = powf(q / ((float)d + 0.001f), beta_);
            bot.dist_inv[i+s_num][j] = bot.dist_inv[j][i+s_num] = inv_d;
            
            if (bot.valid_s[j] && d == 0) {
                bot.valid_s[j] = 0;
                bot.valid_s_num--;
                bot.cluster_valid[bot.cluster[j]]--;
            }
        }
        if (!bot.best_route[i].empty() && players[i] == bot.stars[bot.best_route[i].front()]) {
            bot.best_score[i] -= 1.0f;
            bot.best_route[i].pop();
        }
    }
    static thread_local std::vector<float> temp_scores;
    sim_best(bot,temp_scores, players, 0, bot.best_route[0]);
    bot.best_score = std::move(temp_scores);
}

//for default test_app
template<typename T>
void write_primitive(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

// プリミティブ型を読み込む
template<typename T>
void read_primitive(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

// std::vectorを再帰的に書き込む
template<typename T>
void write_vector(std::ostream& out, const std::vector<T>& vec);

// Tがプリミティブ型の場合の特殊化
template<>
void write_vector<int>(std::ostream& out, const std::vector<int>& vec) {
    size_t size = vec.size();
    write_primitive(out, size);
    out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(int));
}
template<>
void write_vector<float>(std::ostream& out, const std::vector<float>& vec) {
    size_t size = vec.size();
    write_primitive(out, size);
    out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
}
template<>
void write_vector<uint8_t>(std::ostream& out, const std::vector<uint8_t>& vec) {
    size_t size = vec.size();
    write_primitive(out, size);
    out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(uint8_t));
}
template<>
void write_vector<point>(std::ostream& out, const std::vector<point>& vec) {
    size_t size = vec.size();
    write_primitive(out, size);
    out.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(point));
}

// Tがstd::vectorの場合の汎用テンプレート（ネストしたvector用）
template<typename T>
void write_vector(std::ostream& out, const std::vector<T>& vec) {
    size_t size = vec.size();
    write_primitive(out, size);
    for (const auto& elem : vec) {
        write_vector(out, elem); // 再帰呼び出し
    }
}

// std::vectorを再帰的に読み込む
template<typename T>
void read_vector(std::istream& in, std::vector<T>& vec);

// Tがプリミティブ型の場合の特殊化
template<>
void read_vector<int>(std::istream& in, std::vector<int>& vec) {
    size_t size;
    read_primitive(in, size);
    vec.resize(size);
    in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(int));
}
template<>
void read_vector<float>(std::istream& in, std::vector<float>& vec) {
    size_t size;
    read_primitive(in, size);
    vec.resize(size);
    in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
}
template<>
void read_vector<uint8_t>(std::istream& in, std::vector<uint8_t>& vec) {
    size_t size;
    read_primitive(in, size);
    vec.resize(size);
    in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(uint8_t));
}
template<>
void read_vector<point>(std::istream& in, std::vector<point>& vec) {
    size_t size;
    read_primitive(in, size);
    vec.resize(size);
    in.read(reinterpret_cast<char*>(vec.data()), size * sizeof(point));
}


// Tがstd::vectorの場合の汎用テンプレート（ネストしたvector用）
template<typename T>
void read_vector(std::istream& in, std::vector<T>& vec) {
    size_t size;
    read_primitive(in, size);
    vec.resize(size);
    for (auto& elem : vec) {
        read_vector(in, elem); // 再帰呼び出し
    }
}

// std::queue<int>のvectorを書き込む特殊な関数
void write_queues(std::ostream& out, const std::vector<std::queue<int>>& vq) {
    size_t vec_size = vq.size();
    write_primitive(out, vec_size);
    for (const auto& q_const : vq) {
        auto q = q_const; // コピーを作成
        std::vector<int> temp_vec;
        while (!q.empty()) {
            temp_vec.push_back(q.front());
            q.pop();
        }
        write_vector(out, temp_vec);
    }
}

// std::queue<int>のvectorを読み込む特殊な関数
void read_queues(std::istream& in, std::vector<std::queue<int>>& vq) {
    size_t vec_size;
    read_primitive(in, vec_size);
    vq.assign(vec_size, std::queue<int>()); // サイズを合わせて空のキューで初期化
    for (auto& q : vq) {
        std::vector<int> temp_vec;
        read_vector(in, temp_vec);
        for (int val : temp_vec) {
            q.push(val);
        }
    }
}

// bot_state全体をファイルに保存する関数
void save_state(const bot_state& bot, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }

    write_vector(out, bot.stars);
    write_vector(out, bot.star_graph);
    write_vector(out, bot.dist_inv);
    write_vector(out, bot.phe);
    write_vector(out, bot.valid_s);
    write_primitive(out, bot.valid_s_num);
    write_vector(out, bot.best_score);
    write_queues(out, bot.best_route);
    write_primitive(out, bot.rho_inv);
    write_vector(out, bot.cluster_valid);
    write_vector(out, bot.cluster);
    write_vector(out, bot.p_p);
}

// bot_state全体をファイルから読み込む関数
bool load_state(bot_state& bot, const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        // ファイルが存在しない場合は初回実行なので、エラーではなくfalseを返す
        return false;
    }

    read_vector(in, bot.stars);
    read_vector(in, bot.star_graph);
    read_vector(in, bot.dist_inv);
    read_vector(in, bot.phe);
    read_vector(in, bot.valid_s);
    read_primitive(in, bot.valid_s_num);
    read_vector(in, bot.best_score);
    read_queues(in, bot.best_route);
    read_primitive(in, bot.rho_inv);
    read_vector(in, bot.cluster_valid);
    read_vector(in, bot.cluster);
    read_vector(in, bot.p_p);

    return in.good();
}


std::pair<int,int> move_ver2(const std::vector<std::pair<int,int>>& players_, const std::vector<std::pair<int,int>>& star_list_, const int& p_idx) {
    static bot_state bot;
    static std::vector<XorShift> rngs;

    load_state(bot, state_filename);

    if (rngs.empty()) {
        rngs.resize(multi_num);
        std::random_device rd;
        for (int i=0; i<multi_num; ++i) {
            rngs[i] = XorShift(rd());
        }
    }
    omp_set_num_threads(multi_num);

    std::vector<point> players(players_.size());
    std::vector<point> star_list(star_list_.size());
    for (size_t i=0; i<players_.size(); ++i) {
        players[i] = {players_[i].first, players_[i].second};
    }
    for (size_t i=0; i<star_list_.size(); ++i) {
        star_list[i] = {star_list_[i].first, star_list_[i].second};
    }
    
    // Avoid re-loading/saving state on every single call when running in-process.
    // If `bot` is uninitialized, attempt to load persisted state; otherwise reuse in-memory state.
    if (bot.stars.empty()) {
        load_state(bot, state_filename);
    }
    if (is_new_game(bot, star_list)) {
        init(bot, rngs[0], players, star_list);
    } else {
        update(bot, players);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    auto end_time = start_time + std::chrono::duration<double>(time_out * 0.9); // OPTIMIZED: マージンを少し増やす

    auto aco_choose = [&](const bot_state& bot, XorShift& rng, int p_idx, int v_from, const uint8_t* rest_s, const std::vector<int>& cluster_sizes) {
        return choose_v(bot, rng, p_idx, v_from, rest_s, cluster_sizes);
    };

    long long t = 0;

    std::vector<float> good_score(p_num, 0.0f);
    std::vector<std::queue<int>> good_route(p_num);
    std::vector<float> test_score;
    std::vector<std::vector<float>> local_good_score(multi_num, std::vector<float>(p_num, 0.0f));
    std::vector<std::vector<std::queue<int>>> local_good_route(multi_num, std::vector<std::queue<int>>(p_num));

    using Edge = std::pair<int,int>;
    struct EdgeHash { size_t operator()(const Edge& e) const { return ((size_t)e.first<<32) ^ (unsigned)e.second; } };
    std::unordered_map<Edge, float, EdgeHash> deposit_accum[p_num];
    float deposit_total_weight[p_num] = {0};

    while (std::chrono::high_resolution_clock::now() < end_time) {
        std::fill(good_score.begin(), good_score.end(), 0.0f);
        for(auto& q : good_route) {
            std::queue<int> empty_q;
            q.swap(empty_q);
        }
        std::fill(local_good_score.begin(), local_good_score.end(), std::vector<float>(p_num, 0.0f));
        for(auto& lr : local_good_route) {
            for(auto& q : lr) {
                std::queue<int> empty_q;
                q.swap(empty_q);
            }
        }
        
        #pragma omp parallel for reduction(+:t)
        for (int i=0; i<simu_num; ++i) {
            int thread_id = omp_get_thread_num();
            sim_res local_sim_result;
            local_sim_result.route.resize(p_num);
            sim_com(bot, rngs[thread_id], local_sim_result, aco_choose);
            float win_score = *std::max_element(local_sim_result.score.begin(), local_sim_result.score.end());
            for (int j=0; j<p_num; ++j) {
                float current_score = local_sim_result.score[j];
                if (current_score == win_score && win_score > 0) {
                    current_score += win_bonus;
                }

                if (current_score > local_good_score[thread_id][j]) {
                    local_good_score[thread_id][j] = current_score;
                    local_good_route[thread_id][j] = local_sim_result.route[j];
                }
            }
            t++;
        }
        for (int i=0; i<multi_num; ++i) {
            for (int j=0; j<p_num; ++j) {
                if (local_good_score[i][j] > good_score[j]) {
                    good_score[j] = local_good_score[i][j];
                    good_route[j] = local_good_route[i][j];
                }
            }
        }
        for (int i=0; i<p_num; ++i) {
            if (good_route[i].empty()) continue;

            sim_best(bot, test_score, players, i, good_route[i]);
            float win_score = *std::max_element(test_score.begin(), test_score.end());
            
            if (test_score[i] == win_score && win_score > 0) {
                test_score[i] += win_bonus;
            }
            
            bool is_best = false;
            float prev_best = bot.best_score[i];
            if (test_score[i] > prev_best) {
                bot.best_score[i] = test_score[i];
                bot.best_route[i] = good_route[i];
                is_best = true;
            }
            
            if (test_score[i] > 0) {
                std::queue<int> route = good_route[i];
                int from_node = i + s_num;
                const float base = std::pow(test_score[i], gamma_);
                const int len = route.size();
                int cnt = 0;
                float weight_sum = 0.0f;
                while (!route.empty()) {
                    int to_node = route.front();
                    route.pop();
                    float w = is_best ? (test_score[i] / (prev_best + 0.01f)) * std::pow(best_bonus, 1.0f - best_rho * ((float)cnt / (float)len))
                                      : 1.0f;
                    float contrib = base * w;
                    deposit_accum[i][{from_node, to_node}] += contrib;
                    deposit_accum[i][{to_node, from_node}] += contrib; 
                    weight_sum += contrib;
                    from_node = to_node;
                    cnt++;
                }
                deposit_total_weight[i] += weight_sum;
            }
        }
    }
    for (int i=0; i<p_num; ++i) {
        float total_w = deposit_total_weight[i];
        if (total_w > 0.0f) {
            float scale = deposit_budget / total_w;
            const float p_max = phe_max(bot, i);
            for (const auto& kv : deposit_accum[i]) {
                int from_node = kv.first.first;
                int to_node = kv.first.second;
                float add = kv.second * scale;
                float v = bot.phe[i][from_node][to_node] + add;
                float v2 = bot.phe[i][to_node][from_node] + add;
                bot.phe[i][from_node][to_node] = std::min(p_max, v);
                bot.phe[i][to_node][from_node] = std::min(p_max, v2);
            }
        }
    }
    for (int i=0; i<p_num; ++i) {
        const float p_min = phe_min(bot, i);
        for (int j=0; j<p_num+s_num; ++j) {
            for (int k=0; k<p_num+s_num; ++k) {
                bot.phe[i][j][k] = std::max(p_min, bot.phe[i][j][k] * rho);
            }
        }
    }

    //std::cout << "Simulations: " << t << std::endl;
    save_state(bot, state_filename);

    if (!bot.best_route[p_idx].empty()) {
        return move_to(bot, players[p_idx], bot.best_route[p_idx].front());
    } else {
        return {players[p_idx].r, players[p_idx].c};
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
    if (params.contains("deposit_budget")) deposit_budget = params["deposit_budget"].cast<float>();
}

PYBIND11_MODULE(move_ver2, m) {
    m.def("move_ver2", &move_ver2, "", 
        pybind11::arg("players_"), pybind11::arg("star_list_"), pybind11::arg("p_idx"));
    m.def("set_parameters", &set_parameters);
}