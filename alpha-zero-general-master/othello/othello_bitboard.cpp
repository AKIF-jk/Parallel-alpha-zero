// High-performance 8x8 Othello implementation using bit-boards.
// Build example:
// c++ -O3 -Wall -shared -std=c++17 -fPIC \
//   $(python3 -m pybind11 --includes) \
//   othello/othello_bitboard.cpp \
//   -o othello_bitboard$(python3-config --extension-suffix)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace py = pybind11;

class BitBoard {
private:
  uint64_t black_;
  uint64_t white_;
  int n_;

  static constexpr uint64_t NOT_A_FILE = 0xfefefefefefefefeULL;
  static constexpr uint64_t NOT_H_FILE = 0x7f7f7f7f7f7f7f7fULL;

  static inline uint64_t shift_n(uint64_t b) { return b << 8; }
  static inline uint64_t shift_s(uint64_t b) { return b >> 8; }
  static inline uint64_t shift_e(uint64_t b) { return (b << 1) & NOT_A_FILE; }
  static inline uint64_t shift_w(uint64_t b) { return (b >> 1) & NOT_H_FILE; }
  static inline uint64_t shift_ne(uint64_t b) { return (b << 9) & NOT_A_FILE; }
  static inline uint64_t shift_nw(uint64_t b) { return (b << 7) & NOT_H_FILE; }
  static inline uint64_t shift_se(uint64_t b) { return (b >> 7) & NOT_A_FILE; }
  static inline uint64_t shift_sw(uint64_t b) { return (b >> 9) & NOT_H_FILE; }

  static uint64_t ray_flips(uint64_t move, uint64_t player, uint64_t opponent,
                            uint64_t (*shift)(uint64_t), uint64_t border_mask) {
    uint64_t flips = 0;
    uint64_t cur = shift(move) & border_mask;
    while (cur && (cur & opponent)) {
      flips |= cur;
      cur = shift(cur) & border_mask;
    }
    if (cur & player) {
      return flips;
    }
    return 0;
  }

  static uint64_t legal_dir(uint64_t player, uint64_t opponent,
                            uint64_t (*shift)(uint64_t), uint64_t border_mask) {
    uint64_t empty = ~(player | opponent);
    uint64_t x = shift(player) & opponent & border_mask;
    x |= shift(x) & opponent & border_mask;
    x |= shift(x) & opponent & border_mask;
    x |= shift(x) & opponent & border_mask;
    x |= shift(x) & opponent & border_mask;
    x |= shift(x) & opponent & border_mask;
    return shift(x) & empty & border_mask;
  }

public:
  explicit BitBoard(int size = 8) : black_(0), white_(0), n_(size) {
    if (n_ != 8) {
      throw std::runtime_error("BitBoard only supports 8x8.");
    }
    // Initial position: white=1, black=-1 in Python side.
    white_ = (1ULL << 27) | (1ULL << 36); // (3,3) and (4,4)
    black_ = (1ULL << 28) | (1ULL << 35); // (3,4) and (4,3)
  }

  uint64_t get_legal_moves_mask(int color) const {
    uint64_t player = (color == 1) ? white_ : black_;
    uint64_t opponent = (color == 1) ? black_ : white_;

    uint64_t moves = 0;
    moves |= legal_dir(player, opponent, shift_n, ~0ULL);
    moves |= legal_dir(player, opponent, shift_s, ~0ULL);
    moves |= legal_dir(player, opponent, shift_e, NOT_A_FILE);
    moves |= legal_dir(player, opponent, shift_w, NOT_H_FILE);
    moves |= legal_dir(player, opponent, shift_ne, NOT_A_FILE);
    moves |= legal_dir(player, opponent, shift_nw, NOT_H_FILE);
    moves |= legal_dir(player, opponent, shift_se, NOT_A_FILE);
    moves |= legal_dir(player, opponent, shift_sw, NOT_H_FILE);
    return moves;
  }

  std::vector<std::pair<int, int>> get_legal_moves_list(int color) const {
    uint64_t moves = get_legal_moves_mask(color);
    std::vector<std::pair<int, int>> out;
    while (moves) {
      int bit = __builtin_ctzll(moves);
      out.emplace_back(bit / 8, bit % 8);
      moves &= (moves - 1);
    }
    return out;
  }

  bool has_legal_moves(int color) const { return get_legal_moves_mask(color) != 0; }

  void execute_move(int x, int y, int color) {
    uint64_t move = 1ULL << (x * 8 + y);
    uint64_t player = (color == 1) ? white_ : black_;
    uint64_t opponent = (color == 1) ? black_ : white_;

    uint64_t flips = 0;
    flips |= ray_flips(move, player, opponent, shift_n, ~0ULL);
    flips |= ray_flips(move, player, opponent, shift_s, ~0ULL);
    flips |= ray_flips(move, player, opponent, shift_e, NOT_A_FILE);
    flips |= ray_flips(move, player, opponent, shift_w, NOT_H_FILE);
    flips |= ray_flips(move, player, opponent, shift_ne, NOT_A_FILE);
    flips |= ray_flips(move, player, opponent, shift_nw, NOT_H_FILE);
    flips |= ray_flips(move, player, opponent, shift_se, NOT_A_FILE);
    flips |= ray_flips(move, player, opponent, shift_sw, NOT_H_FILE);

    if (color == 1) {
      white_ |= move | flips;
      black_ &= ~flips;
    } else {
      black_ |= move | flips;
      white_ &= ~flips;
    }
  }

  int count_diff(int color) const {
    uint64_t player = (color == 1) ? white_ : black_;
    uint64_t opponent = (color == 1) ? black_ : white_;
    return __builtin_popcountll(player) - __builtin_popcountll(opponent);
  }

  py::array_t<int8_t> to_numpy() const {
    py::array_t<int8_t> out({8, 8});
    auto buf = out.mutable_unchecked<2>();
    for (int x = 0; x < 8; ++x) {
      for (int y = 0; y < 8; ++y) {
        uint64_t bit = 1ULL << (x * 8 + y);
        if (white_ & bit) {
          buf(x, y) = 1;
        } else if (black_ & bit) {
          buf(x, y) = -1;
        } else {
          buf(x, y) = 0;
        }
      }
    }
    return out;
  }

  void from_numpy(py::array_t<int8_t, py::array::c_style | py::array::forcecast> arr) {
    auto buf = arr.unchecked<2>();
    if (buf.shape(0) != 8 || buf.shape(1) != 8) {
      throw std::runtime_error("from_numpy expects shape (8, 8)");
    }
    white_ = 0;
    black_ = 0;
    for (int x = 0; x < 8; ++x) {
      for (int y = 0; y < 8; ++y) {
        uint64_t bit = 1ULL << (x * 8 + y);
        int8_t p = buf(x, y);
        if (p == 1) {
          white_ |= bit;
        } else if (p == -1) {
          black_ |= bit;
        }
      }
    }
  }

  uint64_t hash() const { return black_ ^ (white_ << 1) ^ (white_ >> 63); }
};

PYBIND11_MODULE(othello_bitboard, m) {
  py::class_<BitBoard>(m, "BitBoard")
      .def(py::init<int>(), py::arg("size") = 8)
      .def("get_legal_moves_list", &BitBoard::get_legal_moves_list)
      .def("has_legal_moves", &BitBoard::has_legal_moves)
      .def("execute_move", &BitBoard::execute_move)
      .def("count_diff", &BitBoard::count_diff)
      .def("to_numpy", &BitBoard::to_numpy)
      .def("from_numpy", &BitBoard::from_numpy)
      .def("hash", &BitBoard::hash);
}
