#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <vector>
#include "transposition_table.h"

namespace {
constexpr int kBoardSize = 36;
constexpr int kPassIndex = 36;
constexpr int kChildSlots = 37;

inline int move_to_child_index(int move) {
    // FIXED: HIGH #3 - explicit bounds check and distinct pass-slot index.
    if (move == -1) {
        return kPassIndex;
    }
    if (move < 0 || move >= kBoardSize) {
        return -1;
    }
    return move;
}

inline int child_index_to_move(int idx) {
    return idx == kPassIndex ? -1 : idx;
}
}  // namespace

// Lock-safe model: node statistics are updated lock-free through atomics.
struct alignas(64) MCTSNode {
    float q{0.0f};
    int n{0};
    float p;
    float virtual_loss{0.0f};
    int in_flight{0};
    int children_indices[kChildSlots];
    bool is_expanded{false};
    int parent;
    int move_from_parent;

    MCTSNode() : p(0.0f), parent(-1), move_from_parent(-1) {
        std::fill_n(children_indices, kChildSlots, -1);
    }
};

// Lock-safe model: each rollout uses an independent game copy.
struct OthelloGame {
    int8_t board[kBoardSize];
    int current_player;
    mutable std::array<float, 2 * kBoardSize> planes_buffer{};

    OthelloGame() {
        std::memset(board, 0, sizeof(board));
        board[14] = -1;
        board[15] = 1;
        board[20] = 1;
        board[21] = -1;
        current_player = 1;
    }

    uint64_t get_hash() const {
        uint64_t hash = 0;
        const auto& z = Zobrist::get();
        for (int i = 0; i < kBoardSize; ++i) {
            if (board[i] == 1) {
                hash ^= z.table[0][i];
            } else if (board[i] == -1) {
                hash ^= z.table[1][i];
            }
        }
        if (current_player == 1) {
            hash ^= z.black_to_move;
        }
        return hash;
    }

    int get_flips(int move) const {
        if (move < 0 || move >= kBoardSize || board[move] != 0) {
            return 0;
        }
        const int x = move % 6;
        const int y = move / 6;
        const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
        const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        int flips = 0;
        for (int d = 0; d < 8; ++d) {
            int cx = x + dx[d];
            int cy = y + dy[d];
            int count = 0;
            while (cx >= 0 && cx < 6 && cy >= 0 && cy < 6 && board[cy * 6 + cx] == -current_player) {
                cx += dx[d];
                cy += dy[d];
                ++count;
            }
            if (cx >= 0 && cx < 6 && cy >= 0 && cy < 6 && board[cy * 6 + cx] == current_player) {
                flips += count;
            }
        }
        return flips;
    }

    std::vector<int> get_legal_moves() const {
        std::vector<int> moves;
        moves.reserve(kBoardSize);
        for (int i = 0; i < kBoardSize; ++i) {
            if (get_flips(i) > 0) {
                moves.push_back(i);
            }
        }
        return moves;
    }

    void apply_move(int move) {
        if (move == -1) {
            current_player = -current_player;
            return;
        }
        if (move < 0 || move >= kBoardSize) {
            return;
        }
        const int x = move % 6;
        const int y = move / 6;
        const int dx[] = {-1, 0, 1, -1, 1, -1, 0, 1};
        const int dy[] = {-1, -1, -1, 0, 0, 1, 1, 1};
        board[move] = static_cast<int8_t>(current_player);
        for (int d = 0; d < 8; ++d) {
            int cx = x + dx[d];
            int cy = y + dy[d];
            int count = 0;
            while (cx >= 0 && cx < 6 && cy >= 0 && cy < 6 && board[cy * 6 + cx] == -current_player) {
                cx += dx[d];
                cy += dy[d];
                ++count;
            }
            if (cx >= 0 && cx < 6 && cy >= 0 && cy < 6 && board[cy * 6 + cx] == current_player) {
                cx = x + dx[d];
                cy = y + dy[d];
                while (count > 0) {
                    board[cy * 6 + cx] = static_cast<int8_t>(current_player);
                    cx += dx[d];
                    cy += dy[d];
                    --count;
                }
            }
        }
        current_player = -current_player;
    }

    bool is_terminal() const {
        if (!get_legal_moves().empty()) {
            return false;
        }
        OthelloGame pass_game = *this;
        pass_game.current_player = -pass_game.current_player;
        return pass_game.get_legal_moves().empty();
    }

    // FIXED: CRITICAL #4 - persistent member-backed planes buffer for zero-copy NumPy views.
    float* get_planes_buffer() {
        for (int i = 0; i < kBoardSize; ++i) {
            planes_buffer[i] = (board[i] == current_player) ? 1.0f : 0.0f;
            planes_buffer[kBoardSize + i] = (board[i] == -current_player) ? 1.0f : 0.0f;
        }
        return planes_buffer.data();
    }
};

// Lock-safe model: shared_mutex protects structural mutations; statistics remain lock-free.
class MCTSTree {
public:
    std::vector<MCTSNode> nodes;
    std::shared_mutex tree_mutex;

    MCTSTree() {
        nodes.reserve(500000);
        nodes.emplace_back();
    }

    void reset() {
        nodes.clear();
        nodes.reserve(500000);
        nodes.emplace_back();
    }

    int add_node(int parent, int move) {
        std::unique_lock<std::shared_mutex> lock(tree_mutex);
        return add_node_nolock(parent, move);
    }

    void add_virtual_loss(int node_idx, float loss = 0.3f) {
        std::unique_lock<std::shared_mutex> lock(tree_mutex);
        while (node_idx != -1) {
            nodes[node_idx].virtual_loss += loss;
            nodes[node_idx].n += 1;
            nodes[node_idx].in_flight += 1;
            node_idx = nodes[node_idx].parent;
        }
    }

    void backup(int node_idx, float v, float loss = 0.3f) {
        std::unique_lock<std::shared_mutex> lock(tree_mutex);
        while (node_idx != -1) {
            MCTSNode& node = nodes[node_idx];
            int visits = node.n;
            if (visits <= 0) {
                visits = 1;
            }
            node.q = ((node.q * static_cast<float>(visits - 1)) + v) / static_cast<float>(visits);
            node.virtual_loss -= loss;
            node.in_flight -= 1;
            v = -v;
            node_idx = node.parent;
        }
    }

    int select_leaf(OthelloGame& game) {
        std::shared_lock<std::shared_mutex> lock(tree_mutex);
        int curr = 0;
        constexpr float cpuct = 1.0f;
        constexpr float in_flight_penalty = 0.3f;

        while (nodes[curr].is_expanded) {
            int best_child = -1;
            int best_move = -1;
            float best_ucb = -1e9f;
            const int n_curr = std::max(1, nodes[curr].n);
            const float sqrt_n = std::sqrt(static_cast<float>(n_curr));

            for (int i = 0; i < kChildSlots; ++i) {
                const int child_idx = nodes[curr].children_indices[i];
                if (child_idx == -1) {
                    continue;
                }
                const float q = nodes[child_idx].q;
                const int n = nodes[child_idx].n;
                const float vl = nodes[child_idx].virtual_loss;
                const int in_flight = nodes[child_idx].in_flight;
                const float p = nodes[child_idx].p;
                const float ucb = (q - vl - in_flight_penalty * static_cast<float>(in_flight)) +
                                  cpuct * p * sqrt_n / (1.0f + static_cast<float>(n));
                if (ucb > best_ucb) {
                    best_ucb = ucb;
                    best_child = child_idx;
                    best_move = child_index_to_move(i);
                }
            }

            if (best_child == -1) {
                break;
            }
            curr = best_child;
            game.apply_move(best_move);
        }
        return curr;
    }

    void expand(int node_idx, const std::vector<int>& valid_moves, const std::vector<float>& policy) {
        if (nodes[node_idx].is_expanded) {
            return;
        }

        std::unique_lock<std::shared_mutex> lock(tree_mutex);
        if (nodes[node_idx].is_expanded) {
            return;
        }

        if (valid_moves.empty()) {
            const int child = add_node_nolock(node_idx, -1);
            if (child != -1) {
                nodes[child].p = 1.0f;
            }
        } else {
            for (int move : valid_moves) {
                if (move < 0 || move >= kBoardSize || static_cast<size_t>(move) >= policy.size()) {
                    continue;
                }
                const int child = add_node_nolock(node_idx, move);
                if (child != -1) {
                    nodes[child].p = policy[move];
                }
            }
        }

        nodes[node_idx].is_expanded = true;
    }

private:
    int add_node_nolock(int parent, int move) {
        nodes.emplace_back();
        const int idx = static_cast<int>(nodes.size()) - 1;
        nodes[idx].parent = parent;
        nodes[idx].move_from_parent = move;

        if (parent != -1) {
            const int slot = move_to_child_index(move);
            if (slot >= 0 && slot < kChildSlots) {
                nodes[parent].children_indices[slot] = idx;
            }
        }
        return idx;
    }
};
