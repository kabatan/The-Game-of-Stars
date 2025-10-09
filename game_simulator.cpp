#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <map>
#include <cmath>

namespace py = pybind11;

// --- Game configuration (Python側と合わせる) ---
constexpr int ROWS = 20;
constexpr int COLS = 30;
constexpr int STAR_COUNT = 50;
constexpr int MAX_TURNS = 200;
constexpr int DEFAULT_NB_PLAYERS = 6;

struct Point {
    int r, c;
    bool operator==(const Point& other) const { return r == other.r && c == other.c; }
};

// Pythonのタプルとの変換
namespace pybind11 { namespace detail {
    template <> struct type_caster<Point> {
    public:
        PYBIND11_TYPE_CASTER(Point, _("Point"));
        bool load(handle src, bool) {
            if (!py::isinstance<py::tuple>(src) || py::len(src) != 2) return false;
            auto t = py::cast<py::tuple>(src);
            value.r = py::cast<int>(t[0]);
            value.c = py::cast<int>(t[1]);
            return true;
        }
        static handle cast(Point src, return_value_policy, handle) {
            return py::make_tuple(src.r, src.c).release();
        }
    };
}}

// 乱数生成器
thread_local std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

std::pair<std::vector<Point>, std::vector<Point>> generate_initial_positions() {
    std::vector<Point> players;
    players.reserve(DEFAULT_NB_PLAYERS);
    while (players.size() < DEFAULT_NB_PLAYERS) {
        int r = std::uniform_int_distribution<int>(0, ROWS - 1)(rng);
        int c = std::uniform_int_distribution<int>(0, COLS - 1)(rng);
        bool far_enough = true;
        for (const auto& p : players) {
            if (std::abs(r - p.r) + std::abs(c - p.c) < 5) {
                far_enough = false;
                break;
            }
        }
        if (far_enough) {
            players.push_back({r, c});
        }
    }

    std::vector<Point> stars;
    stars.reserve(STAR_COUNT);
    while (stars.size() < STAR_COUNT) {
        int r = std::uniform_int_distribution<int>(0, ROWS - 1)(rng);
        int c = std::uniform_int_distribution<int>(0, COLS - 1)(rng);
        Point new_star = {r, c};
        
        bool is_duplicate = false;
        for (const auto& s : stars) { if (s == new_star) { is_duplicate = true; break; } }
        if (is_duplicate) continue;

        bool far_from_players = true;
        for (const auto& p : players) {
            if (std::abs(r - p.r) + std::abs(c - p.c) <= 4) {
                far_from_players = false;
                break;
            }
        }
        if (far_from_players) {
            stars.push_back(new_star);
        }
    }
    return {players, stars};
}

// C++からPythonのmove関数を呼び出し、1ゲームをシミュレートするコア関数
py::dict simulate_single_game_cpp(const py::list& move_funcs_py) {
    auto [players, stars] = generate_initial_positions();
    std::vector<int> scores(DEFAULT_NB_PLAYERS, 0);
    int turn = 0;

    std::vector<py::function> move_functions;
    for (const auto& handle : move_funcs_py) {
        move_functions.push_back(handle.cast<py::function>());
    }

    while (turn < MAX_TURNS && !stars.empty()) {
        // 1. 各プレイヤーの移動提案を取得 (Python関数を呼び出す)
        std::vector<Point> proposed;
        py::list players_py = py::cast(players);
        py::list stars_py = py::cast(stars);

        for (int i = 0; i < DEFAULT_NB_PLAYERS; ++i) {
            Point new_coord;
            try {
                py::object result = move_functions[i](players_py, stars_py);
                new_coord = result.cast<Point>();
            } catch (const py::error_already_set& e) {
                // Python側でエラーが起きたら、その場に留まる
                new_coord = players[i];
            }
            
            // 妥当性チェック
            if (std::abs(new_coord.r - players[i].r) + std::abs(new_coord.c - players[i].c) <= 1 &&
                0 <= new_coord.r && new_coord.r < ROWS && 0 <= new_coord.c && new_coord.c < COLS) {
                proposed.push_back(new_coord);
            } else {
                proposed.push_back(players[i]);
            }
        }

        // 2. 衝突解決
        std::map<std::pair<int, int>, std::vector<int>> target_to_indices;
        for(int i = 0; i < proposed.size(); ++i) {
            target_to_indices[{proposed[i].r, proposed[i].c}].push_back(i);
        }

        std::vector<bool> is_winner(DEFAULT_NB_PLAYERS, false);
        for(auto const& [target, indices] : target_to_indices) {
            if (indices.size() == 1) {
                is_winner[indices[0]] = true;
            } else {
                int winner_idx = std::uniform_int_distribution<int>(0, indices.size() - 1)(rng);
                is_winner[indices[winner_idx]] = true;
            }
        }

        // 3. 占有ルール適用
        std::vector<bool> will_move(DEFAULT_NB_PLAYERS);
        std::map<std::pair<int, int>, bool> vacating;
        for(int i = 0; i < DEFAULT_NB_PLAYERS; ++i) {
            will_move[i] = is_winner[i] && !(players[i] == proposed[i]);
            vacating[{players[i].r, players[i].c}] = will_move[i];
        }

        std::vector<Point> final_positions = players;
        for(int i = 0; i < DEFAULT_NB_PLAYERS; ++i) {
            if (!will_move[i]) continue;
            Point target = proposed[i];
            if (vacating.count({target.r, target.c}) && !vacating.at({target.r, target.c})) {
                continue; // 移動先が塞がっている
            }
            final_positions[i] = target;
        }
        players = final_positions;

        // 4. スコア更新と星の削除
        std::vector<Point> remaining_stars;
        for (const auto& star : stars) {
            bool collected = false;
            for (int i = 0; i < DEFAULT_NB_PLAYERS; ++i) {
                if (players[i] == star) {
                    scores[i]++;
                    collected = true;
                    // 複数のプレイヤーが同じ星に乗った場合でも、星は1つ消える
                    break;
                }
            }
            if (!collected) {
                remaining_stars.push_back(star);
            }
        }
        stars = remaining_stars;
        
        turn++;
    }

    // 5. 勝者判定と結果の返却
    int max_score = 0;
    if (!scores.empty()) {
        max_score = *std::max_element(scores.begin(), scores.end());
    }
    std::vector<int> winners;
    for(int i = 0; i < scores.size(); ++i) {
        if (scores[i] == max_score) {
            winners.push_back(i);
        }
    }
    
    py::dict result;
    result["scores"] = scores;
    result["turns"] = turn;
    result["winner"] = (winners.size() == 1) ? winners[0] : -1;
    result["is_tie"] = winners.size() > 1;
    result["max_score"] = max_score;
    result["total_stars_collected"] = std::accumulate(scores.begin(), scores.end(), 0);
    result["remaining_stars"] = stars.size();
    
    return result;
}

PYBIND11_MODULE(game_simulator_cpp, m) {
    m.doc() = "A fast game simulator implemented in C++";
    m.def("simulate_single_game_cpp", &simulate_single_game_cpp, "Simulates a single game and returns the results");
}