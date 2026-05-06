#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <memory>
#include "othello_core.cpp"

namespace py = pybind11;

class BatchedMCTS {
    MCTSTree tree;
    OthelloGame root_game;
    
public:
    BatchedMCTS() {}

    void set_root_board(std::vector<int8_t> board, int current_player) {
        root_game = OthelloGame();
        std::copy(board.begin(), board.end(), root_game.board);
        root_game.current_player = current_player;
        tree.reset();
    }

    // Lock-safe model: GIL is released only for C++ tree traversal and backup.
    // FIXED: CRITICAL #4 - returns zero-copy NumPy view backed by persistent game memory.
    py::tuple select_and_get_leaf() {
        auto game = std::make_shared<OthelloGame>(root_game);
        int leaf_idx = -1;

        {
            py::gil_scoped_release release;
            leaf_idx = tree.select_leaf(*game);
            tree.add_virtual_loss(leaf_idx, 0.3f);
        }

        TTEntry entry;
        const bool hit = TranspositionTable::get().lookup(game->get_hash(), entry);

        // Hold shared_ptr ownership in capsule so NumPy view remains valid after return.
        py::capsule owner(new std::shared_ptr<OthelloGame>(game), [](void* p) {
            delete reinterpret_cast<std::shared_ptr<OthelloGame>*>(p);
        });

        float* planes = game->get_planes_buffer();
        // FIXED: CRITICAL #4 - explicit 3D shape/stride metadata and zero-copy pointer view.
        py::buffer_info info(
            planes,
            sizeof(float),
            py::format_descriptor<float>::format(),
            3,
            {2, 6, 6},
            {
                static_cast<py::ssize_t>(36 * sizeof(float)),
                static_cast<py::ssize_t>(6 * sizeof(float)),
                static_cast<py::ssize_t>(sizeof(float))
            });
        py::array state_tensor(info, owner);

        const auto legal_moves = game->get_legal_moves();
        const bool is_terminal = game->is_terminal();

        float v = 0.0f;
        if (is_terminal) {
            int p1 = 0, p2 = 0;
            for (int i = 0; i < 36; ++i) {
                if (game->board[i] == 1) {
                    ++p1;
                } else if (game->board[i] == -1) {
                    ++p2;
                }
            }
            v = (p1 > p2)
                    ? (game->current_player == 1 ? 1.0f : -1.0f)
                    : ((p2 > p1) ? (game->current_player == -1 ? 1.0f : -1.0f) : 0.0f);
        }

        std::vector<float> p_out;
        if (hit) {
            p_out.assign(entry.p, entry.p + 36);
        }

        return py::make_tuple(leaf_idx, state_tensor, legal_moves, is_terminal, v, hit, p_out, entry.v);
    }

    void expand_and_backup(int leaf_idx, std::vector<int> legal_moves, std::vector<float> policy, float value) {
        py::gil_scoped_release release;
        tree.expand(leaf_idx, legal_moves, policy);
        tree.backup(leaf_idx, value, 0.3f);
    }

    // Returns visit counts (N) for each action at root node (size 37: actions 0-35 + pass).
    std::vector<int> get_root_visit_counts() {
        std::vector<int> counts(kChildSlots, 0);
        const auto& root = tree.nodes[0];
        for (int i = 0; i < kChildSlots; ++i) {
            int child_idx = root.children_indices[i];
            if (child_idx != -1) {
                counts[i] = tree.nodes[child_idx].n;
            }
        }
        return counts;
    }

    int get_current_player() {
        return root_game.current_player;
    }

    std::vector<int8_t> get_board() {
        return std::vector<int8_t>(root_game.board, root_game.board + kBoardSize);
    }
};

PYBIND11_MODULE(othello_cpp, m) {
    py::class_<BatchedMCTS>(m, "BatchedMCTS")
        .def(py::init<>())
        .def("set_root_board", &BatchedMCTS::set_root_board)
        .def("select_and_get_leaf", &BatchedMCTS::select_and_get_leaf)
        .def("expand_and_backup", &BatchedMCTS::expand_and_backup)
        .def("get_root_visit_counts", &BatchedMCTS::get_root_visit_counts)
        .def("get_current_player", &BatchedMCTS::get_current_player)
        .def("get_board", &BatchedMCTS::get_board);
}
