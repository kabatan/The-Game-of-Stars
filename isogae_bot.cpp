//
// Created by Kei Isogae on 2025/10/06.
//
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <deque>
#include <cmath>
#include <numeric>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <set>
#include <map>
#include <memory>
#include <functional>
#include <bits/stdc++.h>

// ============================================================================
// Genom Class (GA.py equivalent)
// ============================================================================
struct LayerWeights {
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> biases;
};

class Genom {
private:
    double evaluation;
    LayerWeights layers;

    double av_score;
    int max_score;
    int min_score;
    int win_count;
    double score_diff;

public:
    Genom() : evaluation(0.0) {}

    Genom(const LayerWeights& layer, double eval)
        : layers(layer), evaluation(eval) {}

    const struct LayerWeights& getGenom() const { return layers; }
    void setGenom(const std::vector<std::vector<double>>& genes) { layers.weights = genes; }
    void setGenom(const struct LayerWeights& genes) { layers = genes; }
    void setAvScore(const double& d) { av_score = d; }
    void setMaxScore(const double& d) { max_score = d; }
    void setMinScore(const double& d) { min_score = d; }
    void setWinCount(const int& i) { win_count = i; }
    double getAvScore() const {return av_score;}
    int getMaxScore() const {return max_score;}
    int getMinScore() const {return min_score;}
    int getWinCount() const {return win_count;}
    double getDiff() const {return score_diff;}
    double getEvaluation() const { return evaluation; }
    void setEvaluation(double av, int max, int min, int winc, double dif) {
        av_score = av;
        max_score = max;
        min_score = min;
        win_count = winc;
        score_diff = dif;
        // Blend multiple signals for robust fitness instead of max-only
        evaluation = av * 3.0 + winc * 5.0 + max * 2.0 + min * 1.0 + dif * 1.0;
    }
};

// ============================================================================
// Simulator Class (fast_simulation.py equivalent)
// ============================================================================
class Simulator {
private:
    std::vector<LayerWeights> w;
    int DEFAULT_NB_PLAYERS;
    int ROWS, COLS;
    int STAR_COUNT;
    int MAX_TURNS;

    std::mt19937 gen;

    // Move function type
    using MoveFunction = std::function<std::pair<int,int>(
        const std::vector<std::pair<int,int>>&,  // players
        const std::vector<std::pair<int,int>>&,  // stars
        const LayerWeights&,              // weights
        int                                       // idx
    )>;

    std::vector<MoveFunction> move_functions;

public:
    Simulator(std::mt19937 gn) : DEFAULT_NB_PLAYERS(6), ROWS(20), COLS(30),
                  STAR_COUNT(50), MAX_TURNS(200) {
        gen = gn;

        // Initialize move functions
        initializeMoveFunctions();
    }

    void update(const std::vector<LayerWeights>& weights) {
        w = weights;
    }

    std::pair<std::vector<std::pair<int,int>>, std::vector<std::pair<int,int>>>
    generateInitialPositions(int nb_players) {
        std::vector<std::pair<int,int>> players;
        std::uniform_int_distribution<int>row_dist(0, ROWS - 1);
        std::uniform_int_distribution<int> col_dist(0, COLS - 1);

        // Generate player positions
        while (players.size() < nb_players) {
            int r = row_dist(gen);
            int c = col_dist(gen);

            bool valid = true;
            for (const auto& [pr, pc] : players) {
                if (std::abs(r - pr) + std::abs(c - pc) < 5) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                players.push_back({r, c});
            }
        }

        // Generate star positions
        std::vector<std::pair<int,int>> stars;
        while (stars.size() < STAR_COUNT) {
            int r = row_dist(gen);
            int c = col_dist(gen);

            // Check if position is not already a star
            if (std::find(stars.begin(), stars.end(), std::make_pair(r, c)) != stars.end()) {
                continue;
            }

            // Check distance from players
            bool valid = true;
            for (const auto& [pr, pc] : players) {
                if (std::abs(r - pr) + std::abs(c - pc) <= 4) {
                    valid = false;
                    break;
                }
            }

            if (valid) {
                stars.push_back({r, c});
            }
        }

        return {players, stars};
    }
    std::vector<int> simulateGame() {
    auto [players, stars] = generateInitialPositions(DEFAULT_NB_PLAYERS);
    std::vector<int> scores(DEFAULT_NB_PLAYERS, 0);

    // === 高速アクセス用に1次元インデックス関数 ===
    auto idx = [&](int r, int c) { return r * COLS + c; };

    // === 星マップをboolで保持 ===
    std::vector<bool> star_exists(ROWS * COLS, false);
    for (auto& [r, c] : stars) {
        star_exists[idx(r, c)] = true;
    }

    // === ループ ===
    for (int turn = 0; turn < MAX_TURNS; turn++) {
        bool any_star_left = false;
        for (bool s : star_exists) {
            if (s) { any_star_left = true; break; }
        }
        if (!any_star_left) break; // 星がなくなったら終了

        // 1) 各プレイヤーの移動提案
        std::vector<std::pair<int,int>> proposed(DEFAULT_NB_PLAYERS);
        for (int i = 0; i < DEFAULT_NB_PLAYERS; i++) {
            try {
                if (i == 0 && !w.empty() && w[i].weights.size() > 0)
                    proposed[i] = move_functions[i](players, stars, w[i], i);
                else
                    proposed[i] = move_functions[i](players, stars, LayerWeights(), i);
            } catch (...) {
                proposed[i] = players[i];
            }

            auto [r0, c0] = players[i];
            auto [r1, c1] = proposed[i];
            if (std::abs(r1 - r0) + std::abs(c1 - c0) > 1 ||
                r1 < 0 || r1 >= ROWS || c1 < 0 || c1 >= COLS)
                proposed[i] = players[i]; // 無効なら元位置に戻す
        }

        // 2) 衝突処理 (cellごとにvector<int>保持)
        std::vector<std::vector<int>> cell_players(ROWS * COLS);
        for (int i = 0; i < DEFAULT_NB_PLAYERS; i++) {
            auto [r, c] = proposed[i];
            cell_players[idx(r, c)].push_back(i);
        }

        std::vector<bool> winner(DEFAULT_NB_PLAYERS, false);
        for (int cell = 0; cell < ROWS * COLS; cell++) {
            auto& ps = cell_players[cell];
            if (ps.empty()) continue;
            if (ps.size() == 1) {
                winner[ps[0]] = true;
            } else {
                std::uniform_int_distribution<int>dist(0, ps.size() - 1);
                winner[ps[dist(gen)]] = true;
            }
        }

        // 3) vacating（元の位置を空けるか） and will_move detection
        std::vector<bool> will_move(DEFAULT_NB_PLAYERS, false);
        for (int i = 0; i < DEFAULT_NB_PLAYERS; i++) {
            will_move[i] = winner[i] && (proposed[i] != players[i]);
        }
        std::vector<bool> vacating(ROWS * COLS, false);
        for (int i = 0; i < DEFAULT_NB_PLAYERS; i++) {
            auto [r, c] = players[i];
            vacating[idx(r, c)] = will_move[i];
        }

        // 4) 実際の移動（占有ルールを厳密に適用）
        std::vector<std::pair<int,int>> new_positions = players;
        for (int i = 0; i < DEFAULT_NB_PLAYERS; i++) {
            if (!will_move[i]) continue; // 勝者でも移動しないならスキップ
            auto target = proposed[i];
            int tindex = idx(target.first, target.second);
            // 目標セルが現在占有されており、かつ占有者が退去しないなら移動不可
            bool occupied_now = false;
            for (int j = 0; j < DEFAULT_NB_PLAYERS; j++) {
                if (players[j] == target) { occupied_now = true; break; }
            }
            if (occupied_now && !vacating[tindex]) {
                continue; // 入れ替わり不可
            }
            new_positions[i] = target;
        }

        // 5) 星の取得判定（bool配列を使用）
        for (int i = 0; i < DEFAULT_NB_PLAYERS; i++) {
            auto [r, c] = new_positions[i];
            int pos = idx(r, c);
            if (star_exists[pos]) {
                scores[i]++;
                star_exists[pos] = false; // 星を消す
            }
        }

        players = new_positions;

        // 星リスト更新（次回のmove関数用）
        stars.clear();
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                if (star_exists[idx(r, c)]) {
                    stars.emplace_back(r, c);
                }
            }
        }
    }

    return scores;
}

   /* std::vector<int> simulateGame() {
        auto [players, stars] = generateInitialPositions(DEFAULT_NB_PLAYERS);
        std::vector<int> scores(DEFAULT_NB_PLAYERS, 0);

        for (int turn = 0; turn < MAX_TURNS && !stars.empty(); turn++) {
            // 1) Get proposed moves from each player
            std::vector<std::pair<int,int>> proposed;

            for (int idx = 0; idx < DEFAULT_NB_PLAYERS; idx++) {
                std::pair<int,int> new_coord;
                try {
                    if (idx == 0 && !w.empty() && w[idx].size() > 0) {
                        new_coord = move_functions[idx](players, stars, &w[idx], idx);
                    } else {
                        new_coord = move_functions[idx](players, stars, nullptr, idx);
                    }
                } catch (...) {
                    new_coord = players[idx];
                }

                auto [r0, c0] = players[idx];
                auto [r1, c1] = new_coord;

                // Validate move (max 1 step orthogonally, within bounds)
                if (std::abs(r1 - r0) + std::abs(c1 - c0) <= 1 &&
                    r1 >= 0 && r1 < ROWS && c1 >= 0 && c1 < COLS) {
                    proposed.push_back({r1, c1});
                } else {
                    proposed.push_back({r0, c0});
                }
            }

            std::vector<std::pair<int,int>> old_positions = players;

            // 2) Resolve conflicts - if multiple players target same cell, pick random winner
            std::map<std::pair<int,int>, std::vector<int>> target_to_indices;
            for (int i = 0; i < proposed.size(); i++) {
                target_to_indices[proposed[i]].push_back(i);
            }

            std::set<int> winners;
            for (auto& [tgt, idxs] : target_to_indices) {
                if (idxs.size() == 1) {
                    winners.insert(idxs[0]);
                } else {
                    std::uniform_int_distribution<int> dist(0, idxs.size() - 1);
                    int winner = idxs[dist(gen)];
                    winners.insert(winner);
                }
            }

            // 3) Determine who will actually move
            std::vector<bool> will_move(proposed.size(), false);
            for (int i = 0; i < proposed.size(); i++) {
                will_move[i] = (winners.count(i) > 0) && (proposed[i] != old_positions[i]);
            }

            std::map<std::pair<int,int>, bool> vacating;
            for (int i = 0; i < old_positions.size(); i++) {
                vacating[old_positions[i]] = will_move[i];
            }

            // 4) Apply moves
            std::vector<std::pair<int,int>> final_positions = old_positions;
            for (int i = 0; i < proposed.size(); i++) {
                if (!will_move[i]) continue;

                auto target = proposed[i];
                if (vacating.count(target) && !vacating[target]) {
                    continue;
                }

                final_positions[i] = target;
            }
            // 事前に全セルfalseで初期化
            std::vector<bool> star_exists(ROWS * COLS, false);

            // 星の位置を true に
            for (auto& [r, c] : stars) {
                star_exists[r * COLS + c] = true;
            }

            // 星を取得したか確認
            for (int idx = 0; idx < final_positions.size(); idx++) {
                auto [r, c] = final_positions[idx];
                if (star_exists[r * COLS + c]) {
                    scores[idx]++;
                    star_exists[r * COLS + c] = false; // 星を消す
                }
            }
            // 5) Update scores and remove collected stars
            /*
            std::set<std::pair<int,int>> star_set(stars.begin(), stars.end());
            for (int idx = 0; idx < final_positions.size(); idx++) {
                if (star_set.count(final_positions[idx])) {
                    scores[idx]++;
                }
            }

            // Remove collected stars
            std::vector<std::pair<int,int>> new_stars;
            for (const auto& s : stars) {
                if (std::find(final_positions.begin(), final_positions.end(), s) == final_positions.end()) {
                    new_stars.push_back(s);
                }
            }:


            stars = new_stars;
            players = final_positions;
        }

        return scores;
    }
    */


private:
    void initializeMoveFunctions() {
        // Player 1 uses neural network (defined in main)
        move_functions.push_back([this](const std::vector<std::pair<int,int>>& players,
                                        const std::vector<std::pair<int,int>>& stars,
                                        const LayerWeights& weights,
                                        int idx) {
            return moveWithNN(players, stars, weights, idx);
        });

        // Other players use simple greedy strategy
        for (int i = 1; i < DEFAULT_NB_PLAYERS; i++) {
            move_functions.push_back([this](const std::vector<std::pair<int,int>>& players,
                                            const std::vector<std::pair<int,int>>& stars,
                                            const LayerWeights& weights,
                                            int idx) {
                return moveGreedy(players, stars, idx);
            });
        }
    }

    std::pair<int,int> moveWithNN(const std::vector<std::pair<int,int>>& players,
                                   const std::vector<std::pair<int,int>>& stars,
                                   const LayerWeights& weights,
                                   int idx);

    std::pair<int,int> moveGreedy(const std::vector<std::pair<int,int>>& players,
                                   const std::vector<std::pair<int,int>>& stars,
                                   int idx) {
        auto [r, c] = players[idx];

        // Find closest star
        int min_dist = 10000;
        std::pair<int,int> closest_star = stars[0];
        for (const auto& [sr, sc] : stars) {
            int d = std::abs(sc - c) + std::abs(sr - r);
            if (d < min_dist) {
                min_dist = d;
                closest_star = {sr, sc};
            }
        }

        auto [sr, sc] = closest_star;

        // Move towards closest star
        if (c < sc) return {r, c + 1};
        else if (c > sc) return {r, c - 1};
        else if (r < sr) return {r + 1, c};
        else return {r - 1, c};
    }
};

// ============================================================================
// Neural Network Functions
// ============================================================================
std::vector<double> matmul(const std::vector<double>& vec,
                          const std::vector<std::vector<double>>& matrix) {
    std::vector<double> result(matrix[0].size(), 0.0);
    for (size_t j = 0; j < matrix[0].size(); j++) {
        for (size_t i = 0; i < vec.size(); i++) {
            result[j] += vec[i] * matrix[i][j];
        }
    }
    return result;
}
std::vector<double> vecsum(const std::vector<double>& vec1,const std::vector<double>& vec2) {
    if (vec1.size() != vec2.size()) {
        return std::vector<double>(0, 10);
    }
    std::vector<double> result;
    for (int i = 0; i < vec1.size(); i++) {
        result.push_back(vec1[i] + vec2[i]);
    }
    return result;
}

std::vector<double> applyTanh(const std::vector<double>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = std::tanh(vec[i]);
    }
    return result;
}

std::vector<double> softmax(const std::vector<double>& x) {
    double max_val = *std::max_element(x.begin(), x.end());
    std::vector<double> exp_x(x.size());
    double sum = 0.0;

    for (size_t i = 0; i < x.size(); i++) {
        exp_x[i] = std::exp(x[i] - max_val);
        sum += exp_x[i];
    }

    for (size_t i = 0; i < x.size(); i++) {
        exp_x[i] /= sum;
    }

    return exp_x;
}

std::vector<double> forward(const std::vector<double>& vector,
                           const std::vector<std::vector<double>>& w1,
                           const std::vector<std::vector<double>>& w2,
                           const std::vector<std::vector<double>>& w3,
                           const std::vector<double>& b1,
                           const std::vector<double>& b2,
                           const std::vector<double>& b3) {
    auto layer2 = applyTanh(vecsum(matmul(vector, w1),b1));
    auto layer3 = applyTanh(vecsum(matmul(layer2, w2),b2));
    auto layer4 = softmax(vecsum(matmul(layer3, w3),b3));
    return layer4;
}
double fx(double x) {return (x > 0) ? 10 * exp(-x/10) : -10 * exp(x/10);}

// Reshape flat weight vector into matrix
std::vector<std::vector<double>> reshapeWeights(const std::vector<double>& weights,
                                                 int start, int rows, int cols) {
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    int idx = start;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = weights[idx++];
        }
    }
    return matrix;
}

std::pair<int,int> Simulator::moveWithNN(const std::vector<std::pair<int,int>>& players,
                                         const std::vector<std::pair<int,int>>& stars,
                                         const LayerWeights& layer,
                                         int idx) {
    std::vector<std::vector<double>>weights = layer.weights;
    std::vector<std::vector<double>>biases = layer.biases;
    if (layer.weights.size() < 1 || layer.biases.size() < 1) {
        std::cout << "MoveGreddy" << std::endl;
        return moveGreedy(players, stars, idx);
    }

    auto [pr, pc] = players[idx];

    std::vector<std::pair<int,int>> closest_stars(stars.size());
    std::copy(stars.begin(), stars.end(), closest_stars.begin());
    std::sort(closest_stars.begin(), closest_stars.end(),[pr, pc](auto const& a, auto const& b) {
        int ad = std::abs(a.first - pr) + std::abs(a.second - pc);
        int bd = std::abs(b.first - pr) + std::abs(b.second - pc);
        bool re = ad < bd;
        return ad < bd;
    });

   // for (int i = 0; i  < closest_stars.size(); i++) {
   //     std::cout << std::abs(closest_stars[i].first - pr) + std::abs(closest_stars[i].second - pc)  << "NN";
   // }

   // std::cout << std::endl;
    int min_dist = 10000;
    int min_dist2 = 10000;
    std::vector<std::pair<int,int>> closest_bots(players.size());
    std::copy(players.begin(), players.end(), closest_bots.begin());
    std::sort(closest_bots.begin(), closest_bots.end(),[pr, pc](auto const& a, auto const& b) {
        int ad = std::abs(a.first - pr) + std::abs(a.second - pc);
        int bd = std::abs(b.first - pr) + std::abs(b.second - pc);
        return ad < bd;
    });
    // Find closest bot
    //std::cout << closest_stars.front().first - pr << " " << closest_stars.front().second - pc << std::endl;
    // Create input vector
    const double SCALE = 1.0;
    std::vector<double> input_vector;
    for (int i = 0; i < 10;i++) {
        if (i < (int)closest_stars.size()) {
            // Safe normalization into roughly [-1,1]
            input_vector.emplace_back((closest_stars[i].first - pr) / double(ROWS) / double(i + 1));
            input_vector.emplace_back((closest_stars[i].second - pc) / double(COLS) / double(i + 1));

        } else {
            // No star -> use a large-distance sentinel (or zero)
            //input_vector.emplace_back(fx((ROWS + COLS) * 1.0)); // far away
            //input_vector.emplace_back(fx((ROWS + COLS) * 1.0));
            input_vector.emplace_back(0);
            input_vector.emplace_back(0);
        }
    }
    input_vector.emplace_back(pr / double(ROWS));
    input_vector.emplace_back(pc / double(COLS));
    for (int i = 0; i < 5; i++) {
        int botIndex = i + 1; // you used i+1 in original code: nearest other bots
        if (botIndex < (int)closest_bots.size()) {
            //input_vector.emplace_back(fx((closest_bots[botIndex].first - pr) * 0.5));
            input_vector.emplace_back((closest_bots[botIndex].first - pr) / double(ROWS) / double(i + 1));

            //input_vector.emplace_back(fx((closest_bots[botIndex].second - pc) * 0.5));
            input_vector.emplace_back((closest_bots[botIndex].second - pc) / double(COLS) / double(i + 1));


            //input_vector.emplace_back((closest_bots[botIndex].first - pr) / double(ROWS));
            //input_vector.emplace_back((closest_bots[botIndex].second - pc) / double(COLS));
        } else {
            //input_vector.emplace_back(fx((ROWS + COLS) * 1.0));
            //input_vector.emplace_back(fx((ROWS + COLS) * 1.0));
            input_vector.emplace_back(0);
            input_vector.emplace_back(0);
        }
    }
    //input_vector.clear();
    //input_vector.push_back(closest_stars[0].first - pr);
    //input_vector.push_back(closest_stars[0].second - pc);
    //for (int i = 0; i < 10; i++) input_vector.push_back(0);

    auto w1 = reshapeWeights(weights[0], 0, 32, 10);
    auto w2 = reshapeWeights(weights[1], 0, 10, 10);
    auto w3 = reshapeWeights(weights[2], 0, 10, 4);
    auto b1 = biases[0];
    auto b2 = biases[1];
    auto b3 = biases[2];
    // Replace non-finite inputs with 0 and clip to a reasonable range
    for (auto& v : input_vector) {
        if (!std::isfinite(v)) v = 0.0;
        if (v > 3.0) v = 3.0; else if (v < -3.0) v = -3.0;
    }
    // Forwamrd pass
    auto movement = forward(input_vector, w1, w2, w3, b1, b2, b3);
    bool movement_bad = false;
    for (auto m : movement) { if (!std::isfinite(m) || m < 0) { movement_bad = true; break; } }
    if (movement_bad) {
        // Fallback to greedy if NN outputs are invalid
        return moveGreedy(players, stars, idx);
    }
    //std::cout << "AGStep3D" << std::endl;

    // Sample action from probability distribution
    std::discrete_distribution<int> dist(movement.begin(), movement.end());

    int action = dist(gen);

    // Return new position based on action
    if (action == 0) return {pr, pc + 1};      // right
    else if (action == 1) return {pr - 1, pc}; // up
    else if (action == 2) return {pr, pc - 1}; // left
    else return {pr + 1, pc};                   // down
}

// ============================================================================
// Genetic Algorithm Functions
// ============================================================================
const int GENOM_LENGTH = 460;
const int POPULATIONS = 108;
const int SELECT_ELITES = 10;
const int SELECT_PARENTS = 30;
const double INDIVIDUAL_MUTATION = 0.3;
const double GENOM_MUTATION = 0.4;
const int GENERATIONS = 5000;
const int PLAYERS = 6;
const int GAMES_PER_MATCH = 6;

std::random_device rd;
std::mt19937 gen_main(rd());

Genom createGenom(std::vector<int> w_lengths, std::vector<int> b_lengths) {
    std::normal_distribution<double> dist_w(0.0, 0.1);
    std::normal_distribution<double> dist_b(0.0, 0.01);
    LayerWeights l;
    for (int i = 0; i < w_lengths.size(); i++) {
        std::vector<double> ws;
        for (int j = 0; j < w_lengths[i]; j++) {
            ws.push_back(dist_w(gen_main));
        }
        l.weights.push_back(ws);
    }
    for (int i = 0; i < b_lengths.size(); i++) {
        std::vector<double> bs;
        for (int j = 0; j < b_lengths[i]; j++) {
            bs.push_back(dist_b(gen_main));
        }
        l.biases.push_back(bs);
    }
    return Genom(l, 0);
}

std::vector<Genom> select(std::vector<Genom>& ga, int n) {
    std::sort(ga.begin(), ga.end(),
        [](const Genom& a, const Genom& b) {
            return a.getEvaluation() > b.getEvaluation();
        });

    std::vector<Genom> result;
    for (int i = 0; i < n && i < ga.size(); i++) {
        result.push_back(ga[i]);
    }
    return result;
}

std::vector<Genom> roulette_select(const std::vector<Genom>& population, int num_parents) {
    std::vector<double> fitness;
    for (const auto& g : population) fitness.push_back(g.getEvaluation());
    double min_fit = *std::min_element(fitness.begin(), fitness.end());
    // Shift fitness to be positive
    for (auto& f : fitness) f = f - min_fit + 1e-6;
    double total_fit = std::accumulate(fitness.begin(), fitness.end(), 0.0);
    std::vector<Genom> selected;
    std::uniform_real_distribution<double> dist(0.0, total_fit);
    std::mt19937 rng(std::random_device{}());
    for (int n = 0; n < num_parents; n++) {
        double pick = dist(rng);
        double sum = 0.0;
        for (size_t i = 0; i < population.size(); i++) {
            sum += fitness[i];
            if (sum >= pick) {
                selected.push_back(population[i]);

                break;
            }
        }
    }
    // Ensure we always return the requested number (fallback to best individuals)
    if ((int)selected.size() < num_parents) {
        std::vector<int> order(population.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b){ return population[a].getEvaluation() > population[b].getEvaluation(); });
        for (int id : order) {
            if ((int)selected.size() >= num_parents) break;
            selected.push_back(population[id]);
        }
    }
    return selected;
}

std::vector<Genom> crossover(const Genom& ga_one, const Genom& ga_second) {
    LayerWeights progeny_one, progeny_second;
    bool crossed = false;
    progeny_one.weights.resize(ga_one.getGenom().weights.size());
    progeny_second.weights.resize(ga_second.getGenom().weights.size());
    progeny_one.biases.resize(ga_one.getGenom().biases.size());
    progeny_second.biases.resize(ga_second.getGenom().biases.size());
    for (int i = 0; i < ga_one.getGenom().weights.size(); i++) {
        auto row_one = ga_one.getGenom().weights[i];
        auto row_second = ga_second.getGenom().weights[i];
        int cross_first = std::uniform_int_distribution<int>(0, row_one.size() - 1)(gen_main);
        int cross_second = std::uniform_int_distribution<int>(cross_first, row_one.size() - 1)(gen_main);

        if (0.5 > std::uniform_real_distribution<double>(0, 1.0)(gen_main)) {
            progeny_one.weights[i].insert(progeny_one.weights[i].begin(),row_one.begin(), row_one.begin() + cross_first);
            progeny_one.weights[i].insert(progeny_one.weights[i].end(),row_second.begin() + cross_first, row_second.begin() + cross_second);
            progeny_one.weights[i].insert(progeny_one.weights[i].end(),row_one.begin() + cross_second, row_one.end());
            progeny_second.weights[i].insert(progeny_second.weights[i].begin(),row_second.begin(), row_second.begin() + cross_first);
            progeny_second.weights[i].insert(progeny_second.weights[i].end(),row_one.begin() + cross_first, row_one.begin() + cross_second);
            progeny_second.weights[i].insert(progeny_second.weights[i].end(),row_second.begin() + cross_second, row_second.end());
            crossed = true;
        } else {
            progeny_one.weights[i].insert(progeny_one.weights[i].begin(),row_one.begin(), row_one.end() );
            progeny_second.weights[i].insert(progeny_second.weights[i].begin(), row_second.begin(), row_second.end());
        }

    }
    for (int i = 0; i < ga_one.getGenom().biases.size(); i++) {
        auto row_one = ga_one.getGenom().biases[i];
        auto row_second = ga_second.getGenom().biases[i];
        int cross_first = std::uniform_int_distribution<int>(0, (int)row_one.size() - 1)(gen_main);
        int cross_second = std::uniform_int_distribution<int>(cross_first, (int)row_one.size() - 1)(gen_main);
        if (0.5 > std::uniform_real_distribution<double>(0, 1)(gen_main)) {
            progeny_one.biases[i].insert(progeny_one.biases[i].begin(), row_one.begin(), row_one.begin() + cross_first);
            progeny_one.biases[i].insert(progeny_one.biases[i].end(),row_second.begin() + cross_first, row_second.begin() + cross_second);
            progeny_one.biases[i].insert(progeny_one.biases[i].end(),row_one.begin() + cross_second, row_one.end());
            progeny_second.biases[i].insert(progeny_second.biases[i].begin(),row_second.begin(), row_second.begin() + cross_first);
            progeny_second.biases[i].insert(progeny_second.biases[i].end(),row_one.begin() + cross_first, row_one.begin() + cross_second);
            progeny_second.biases[i].insert(progeny_second.biases[i].end(),row_second.begin() + cross_second, row_second.end());

            crossed = true;
        } else {
            progeny_one.biases[i].insert(progeny_one.biases[i].begin(),row_one.begin(), row_one.end());
            progeny_second.biases[i].insert(progeny_second.biases[i].begin(),row_second.begin(), row_second.end());
        }
    }
    std::vector<Genom> result;
    result.push_back(Genom(progeny_one,0));
    result.push_back(Genom(progeny_second,0));
    return result;
}

std::vector<Genom> mutation(std::vector<Genom>& ga, double individual_mutation, double genom_mutation) {
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::normal_distribution<double> mut_dist(0.0,0.05);
    std::normal_distribution<double> bias_dist(0.0, 0.02);

    std::vector<Genom> ga_list;
    for (auto genom : ga) {
        Genom new_genom;
        auto layers = genom.getGenom();
        if (individual_mutation > prob_dist(gen_main)) {
            std::vector<std::vector<double>> new_w;
            std::vector<std::vector<double>> old_w = layers.weights;
            new_w.resize(old_w.size());
            for (int i = 0; i < old_w.size(); i++) {
                for (int j = 0; j < old_w[i].size(); j++) {
                    new_w[i].push_back(old_w[i][j]
                        + ((genom_mutation > prob_dist(gen_main)) ? mut_dist(gen_main) : 0));
                }
            }
            layers.weights = new_w;
            std::vector<std::vector<double>> new_b;
            std::vector<std::vector<double>> old_b = layers.biases;
            new_b.resize(old_b.size());
            for (int i = 0; i < old_b.size(); i++) {
                for (int j = 0; j < old_b[i].size(); j++) {
                    new_b[i].push_back(old_b[i][j]
                        + ((genom_mutation > prob_dist(gen_main)) ? bias_dist(gen_main) : 0));
                }
            }
            layers.biases = new_b;
        }
        new_genom.setGenom(layers);
        new_genom.setEvaluation(0,0,0,0,0);
        ga_list.push_back(new_genom);
    }
    return ga_list;
}

std::vector<Genom> nextGenerationGeneCreate(
    std::vector<Genom>& ga,
    const std::vector<Genom>& ga_elite,
    const std::vector<Genom>& ga_progeny) {

    std::sort(ga.begin(), ga.end(),
        [](const Genom& a, const Genom& b) {
            return a.getEvaluation() < b.getEvaluation();
        });

    int to_remove = ga_elite.size() + ga_progeny.size();
    ga.erase(ga.begin(), ga.begin() + to_remove);
    ga = mutation(ga, INDIVIDUAL_MUTATION,GENOM_MUTATION);
    ga.insert(ga.end(), ga_elite.begin(), ga_elite.end());
    ga.insert(ga.end(), ga_progeny.begin(), ga_progeny.end());

    return ga;
}

// ============================================================================
// Main Program
// ============================================================================
int main() {
    // Initialize population
    std::vector<Genom> current_generation;
    current_generation.reserve(POPULATIONS);
    // Neuron 30(input) -> 10(hidden) -> 10(hidden) -> 4(output)
    std::vector<int> w_ = {320, 100, 40};
    std::vector<int> b_ = {10, 10, 4};
    for (int i = 0; i < POPULATIONS; i++) {
        current_generation.push_back(createGenom(w_, b_));
    }
    // Main evolution loop
    std::ofstream outputn("output.txt");
    int elite_score = 0; //writing outputn
    std::vector<Genom> elite_gene;
    for (int generation = 1; generation <= GENERATIONS; generation++) {
        std::cout << "gene = " << generation << std::endl;
        auto gen_start = std::chrono::high_resolution_clock::now();

        // Evaluation through tournament

        //std::vector<int> placement(POPULATIONS, 0);
        std::vector<double> each_scores(POPULATIONS, 0);
        std::vector<std::vector<Genom>> group;
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < POPULATIONS; i++) {
            unsigned int thread_seed = rd() ^ (omp_get_thread_num() + 123456);
            std::mt19937 local_gen(thread_seed);
            Simulator sim(local_gen);
            std::vector<LayerWeights> l(PLAYERS);
            l[0] = current_generation[i].getGenom();
           // std::cout << l[0].weights.size() << std::endl;
            //other players uses greedy algorithm
            sim.update(l);
            int total_score = 0;
            int win_count = 0;
            int max_score = 0;
            int min_score = 100;
            double score_diff = 0;
            for (int k = 0; k < GAMES_PER_MATCH; k++) {
                auto game_scores = sim.simulateGame();
                total_score += game_scores[0];
                int winner_score = -1;
                max_score = std::max(max_score, game_scores[0]);
                min_score = std::min(min_score, game_scores[0]);
                double other_sum = 0;
                for (int j = 0; j < game_scores.size(); j++) {
                    winner_score = std::max(winner_score, game_scores[j]);
                    if (j != 0) other_sum += game_scores[j];
                }
                score_diff += game_scores[0] - other_sum / 5;
                if (game_scores[0] == winner_score) {
                    win_count += 1;
                }
            }
            current_generation[i].setEvaluation(total_score / GAMES_PER_MATCH, max_score, min_score, win_count, score_diff / GAMES_PER_MATCH);
        }
        auto gen_ends = std::chrono::high_resolution_clock::now();

        // Select elites
        elite_gene = select(current_generation, SELECT_ELITES);

        std::cout << "Elite scores: " << std::endl;
        std::cout << "Max:   ";
        for (int i = 0; i < SELECT_ELITES; i++) {
            std::cout << elite_gene[i].getMaxScore() << " ";
        }
        std::cout << std::endl << "Min:   ";
        for (int i = 0; i < SELECT_ELITES; i++) {
            std::cout << elite_gene[i].getMinScore() << " ";
        }
        std::cout << std::endl << "Avr:   ";
        for (int i = 0; i < SELECT_ELITES; i++) {
            std::cout << elite_gene[i].getAvScore() << " ";
        }
        std::cout << std::endl << "Win:   ";
        for (int i = 0; i < SELECT_ELITES; i++) {
            std::cout << elite_gene[i].getWinCount() << " ";
        }
        std::cout << std::endl;
        outputn << elite_gene[0].getAvScore() <<"," << elite_gene[0].getMaxScore() << "," << elite_gene[0].getMinScore()<< std::endl;

        // Select parents and create offspring
       //auto parent_pool = select(current_generation, SELECT_PARENTS);
        std::vector<Genom> parent_pool = roulette_select(current_generation, SELECT_PARENTS);

        std::vector<Genom> progeny_gene;
        for (int i = 0; i < SELECT_PARENTS; i++) {
            int parent2_idx = (i == 0) ? SELECT_PARENTS - 1 : i - 1;
            auto offspring = crossover(parent_pool[parent2_idx], parent_pool[i]);
            progeny_gene.insert(progeny_gene.end(), offspring.begin(), offspring.end());
        }
        std::cout << "OK1023" << std::endl;
        // Create next generation
        auto next_generation = nextGenerationGeneCreate(current_generation, elite_gene, progeny_gene);

        // Statistics
        std::vector<double> fits;
        for (const auto& individual : current_generation) {
            fits.push_back(individual.getEvaluation());
        }

        double min_fit = *std::min_element(fits.begin(), fits.end());
        double max_fit = *std::max_element(fits.begin(), fits.end());
        double avg_fit = std::accumulate(fits.begin(), fits.end(), 0.0) / fits.size();

        std::cout << "-----Result of generation " << generation << "-----" << std::endl;
        std::cout << "  Min: " << min_fit << std::endl;
        std::cout << "  Max: " << max_fit << std::endl;
        std::cout << "  Avg: " << avg_fit << std::endl;

        current_generation = next_generation;
        std::cout << "preapration for nest" << generation << std::endl;
        if (generation % 500 == 0) {
            Simulator sim(gen_main);
            std::vector<LayerWeights> w(PLAYERS);
            w[0] = elite_gene[0].getGenom();
            sim.update(w);
            int win_count = 0;

            for (int i =0; i < 1000; i++) {
                auto game_scores = sim.simulateGame();
                if (i > game_scores.size() - 20 ) {
                    for (int k = 0; k < game_scores.size(); k++) {
                        std::cout << game_scores[k] << ",";
                    }
                    std::cout << std::endl;
                }
                if (game_scores[0] >= *std::max_element(game_scores.begin(), game_scores.end())) {
                    win_count++;
                }

            }
            std::cout << "eliteGene eval " << elite_gene[0].getEvaluation()<< std::endl;
            std::cout << "Win: " << win_count << " ( " << float(win_count) / 1000.0 * 100.0  << "% )" << std::endl;


        }
    }

    // Save best individuals
    auto elite_genes = select(current_generation, SELECT_ELITES);
    std::cout << "The best individual has score: " << elite_genes[0].getEvaluation() << std::endl;

    std::ofstream outfile("botdata.txt");
    //elite_genes[0];
    outfile.close();

    return 0;
}