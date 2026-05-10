// =============================================================================
// othello_bitboard.cpp  –  High-performance 8×8 Othello bitboard (pybind11)
// =============================================================================
//
// Optimisations added over the baseline:
//
//   SIMD / compiler hints
//   ---------------------
//   * `#pragma GCC optimize("O3,unroll-loops")` + `target("avx2,bmi,bmi2,popcnt")`
//     let GCC/Clang auto-vectorise and use hardware POPCNT/BMI2.
//   * `_mm_popcnt_u64` (x86 intrinsic) used directly in count_diff / batch helpers
//     for guaranteed single-instruction popcount when the header is available.
//   * alignas(64) on the BitBoard data members so the compiler can issue aligned
//     load/store instructions when structs are placed in arrays.
//   * `__builtin_expect` (via LIKELY/UNLIKELY macros) on hot-path branches inside
//     ray_flips and legal_dir.
//   * `__restrict__` on pointer parameters of batch helpers to assert no aliasing.
//   * loop-unroll pragma inside to_numpy / from_numpy inner loops.
//
//   OpenMP
//   ------
//   * `#ifdef _OPENMP` guards throughout – the file compiles cleanly without -fopenmp.
//   * Two new batch entry-points that accept a 3-D numpy array (N×8×8, int8):
//       batch_get_legal_moves_mask(boards, color) -> List[int]   (uint64 masks)
//       batch_count_diff          (boards, color) -> List[int]   (score diffs)
//     Both parallelise over boards with `#pragma omp parallel for schedule(dynamic)`.
//   * Thread safety: each thread works on a private BitBoard copy; results are
//     written to pre-allocated per-index slots (no races, no mutex needed).
//
// Build (with OpenMP + native SIMD):
//   c++ -O3 -march=native -fopenmp -Wall -shared -std=c++17 -fPIC \
//       $(python3 -m pybind11 --includes) \
//       othello_bitboard.cpp \
//       -o othello_bitboard$(python3-config --extension-suffix)
//
// Build (without OpenMP):
//   c++ -O3 -march=native -Wall -shared -std=c++17 -fPIC \
//       $(python3 -m pybind11 --includes) \
//       othello_bitboard.cpp \
//       -o othello_bitboard$(python3-config --extension-suffix)
// =============================================================================

// ---- Compiler-level optimisation hints (GCC / Clang) ------------------------
#pragma GCC optimize("O3,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,popcnt")

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _OPENMP
#  include <omp.h>
#endif

// x86 SIMD intrinsics (popcount, etc.) – guarded so non-x86 still compiles.
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  include <immintrin.h>          // AVX2 / BMI2
#  include <nmmintrin.h>          // _mm_popcnt_u64
#  define HW_POPCNT(x) static_cast<int>(_mm_popcnt_u64(x))
#else
#  define HW_POPCNT(x) static_cast<int>(__builtin_popcountll(x))
#endif

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace py = pybind11;

// ---- Branch-prediction helpers ----------------------------------------------
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

// =============================================================================
// BitBoard class
// =============================================================================
class alignas(64) BitBoard {
private:
  // alignas(64) on the whole class; individual members are packed into the
  // first cache line together.
  uint64_t black_;
  uint64_t white_;
  int      n_;

  static constexpr uint64_t NOT_A_FILE = 0xfefefefefefefefeULL;
  static constexpr uint64_t NOT_H_FILE = 0x7f7f7f7f7f7f7f7fULL;

  // ---- Direction shifts (inlined) ------------------------------------------
  static inline uint64_t shift_n (uint64_t b) { return b << 8; }
  static inline uint64_t shift_s (uint64_t b) { return b >> 8; }
  static inline uint64_t shift_e (uint64_t b) { return (b << 1) & NOT_A_FILE; }
  static inline uint64_t shift_w (uint64_t b) { return (b >> 1) & NOT_H_FILE; }
  static inline uint64_t shift_ne(uint64_t b) { return (b << 9) & NOT_A_FILE; }
  static inline uint64_t shift_nw(uint64_t b) { return (b << 7) & NOT_H_FILE; }
  static inline uint64_t shift_se(uint64_t b) { return (b >> 7) & NOT_A_FILE; }
  static inline uint64_t shift_sw(uint64_t b) { return (b >> 9) & NOT_H_FILE; }

  // ---- Ray flip computation ------------------------------------------------
  static uint64_t ray_flips(uint64_t move,
                             uint64_t player, uint64_t opponent,
                             uint64_t (*shift)(uint64_t),
                             uint64_t border_mask)
  {
    uint64_t flips = 0;
    uint64_t cur   = shift(move) & border_mask;
    while (LIKELY(cur) && LIKELY(cur & opponent)) {
      flips |= cur;
      cur    = shift(cur) & border_mask;
    }
    if (LIKELY(cur & player)) return flips;
    return 0;
  }

  // ---- Legal-move generation in one direction ------------------------------
  static uint64_t legal_dir(uint64_t player, uint64_t opponent,
                             uint64_t (*shift)(uint64_t),
                             uint64_t border_mask)
  {
    uint64_t empty = ~(player | opponent);
    uint64_t x = shift(player) & opponent & border_mask;
    // Six unrolled propagation steps (max board width/height - 2 = 6).
    x |= shift(x) & opponent & border_mask;
    x |= shift(x) & opponent & border_mask;
    x |= shift(x) & opponent & border_mask;
    x |= shift(x) & opponent & border_mask;
    x |= shift(x) & opponent & border_mask;
    return shift(x) & empty & border_mask;
  }

public:
  // ---- Constructor ---------------------------------------------------------
  explicit BitBoard(int size = 8) : black_(0), white_(0), n_(size)
  {
    if (UNLIKELY(n_ != 8))
      throw std::runtime_error("BitBoard only supports 8x8.");
    // Standard Othello starting position:  white=+1, black=-1 in Python.
    white_ = (1ULL << 27) | (1ULL << 36); // (3,3) and (4,4)
    black_ = (1ULL << 28) | (1ULL << 35); // (3,4) and (4,3)
  }

  // ---- Legal move mask -----------------------------------------------------
  uint64_t get_legal_moves_mask(int color) const
  {
    uint64_t player   = (color == 1) ? white_ : black_;
    uint64_t opponent = (color == 1) ? black_ : white_;

    uint64_t moves = 0;
    moves |= legal_dir(player, opponent, shift_n,  ~0ULL);
    moves |= legal_dir(player, opponent, shift_s,  ~0ULL);
    moves |= legal_dir(player, opponent, shift_e,  NOT_A_FILE);
    moves |= legal_dir(player, opponent, shift_w,  NOT_H_FILE);
    moves |= legal_dir(player, opponent, shift_ne, NOT_A_FILE);
    moves |= legal_dir(player, opponent, shift_nw, NOT_H_FILE);
    moves |= legal_dir(player, opponent, shift_se, NOT_A_FILE);
    moves |= legal_dir(player, opponent, shift_sw, NOT_H_FILE);
    return moves;
  }

  // ---- Legal move list (Python-facing) ------------------------------------
  std::vector<std::pair<int,int>> get_legal_moves_list(int color) const
  {
    uint64_t moves = get_legal_moves_mask(color);
    std::vector<std::pair<int,int>> out;
    out.reserve(__builtin_popcountll(moves)); // avoid reallocations
    while (LIKELY(moves)) {
      int bit = __builtin_ctzll(moves);
      out.emplace_back(bit / 8, bit % 8);
      moves &= moves - 1;      // clear lowest set bit
    }
    return out;
  }

  // ---- Has any legal move? -------------------------------------------------
  bool has_legal_moves(int color) const
  {
    return get_legal_moves_mask(color) != 0;
  }

  // ---- Execute a move ------------------------------------------------------
  void execute_move(int x, int y, int color)
  {
    uint64_t move     = 1ULL << (x * 8 + y);
    uint64_t player   = (color == 1) ? white_ : black_;
    uint64_t opponent = (color == 1) ? black_ : white_;

    uint64_t flips = 0;
    flips |= ray_flips(move, player, opponent, shift_n,  ~0ULL);
    flips |= ray_flips(move, player, opponent, shift_s,  ~0ULL);
    flips |= ray_flips(move, player, opponent, shift_e,  NOT_A_FILE);
    flips |= ray_flips(move, player, opponent, shift_w,  NOT_H_FILE);
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

  // ---- Score difference (hardware POPCNT) ----------------------------------
  int count_diff(int color) const
  {
    uint64_t player   = (color == 1) ? white_ : black_;
    uint64_t opponent = (color == 1) ? black_ : white_;
    return HW_POPCNT(player) - HW_POPCNT(opponent);
  }

  // ---- Export to numpy (8×8, int8) ----------------------------------------
  py::array_t<int8_t> to_numpy() const
  {
    py::array_t<int8_t> out({8, 8});
    auto buf = out.mutable_unchecked<2>();

    for (int x = 0; x < 8; ++x) {
      #pragma GCC unroll 8
      for (int y = 0; y < 8; ++y) {
        uint64_t bit = 1ULL << (x * 8 + y);
        buf(x, y) = (white_ & bit) ? int8_t(1)
                  : (black_ & bit) ? int8_t(-1)
                  :                  int8_t(0);
      }
    }
    return out;
  }

  // ---- Import from numpy (8×8, int8) --------------------------------------
  void from_numpy(
      py::array_t<int8_t, py::array::c_style | py::array::forcecast> arr)
  {
    auto buf = arr.unchecked<2>();
    if (UNLIKELY(buf.shape(0) != 8 || buf.shape(1) != 8))
      throw std::runtime_error("from_numpy expects shape (8, 8)");

    white_ = 0;
    black_ = 0;
    for (int x = 0; x < 8; ++x) {
      #pragma GCC unroll 8
      for (int y = 0; y < 8; ++y) {
        uint64_t bit = 1ULL << (x * 8 + y);
        int8_t   p   = buf(x, y);
        if      (p ==  1) white_ |= bit;
        else if (p == -1) black_ |= bit;
      }
    }
  }

  // ---- Zobrist-style hash --------------------------------------------------
  uint64_t hash() const
  {
    return black_ ^ (white_ << 1) ^ (white_ >> 63);
  }

  // ==========================================================================
  // Internal helpers used by the batch functions below
  // ==========================================================================

  // Load state directly from a raw int8 pointer (row-major 8×8 board).
  // The pointer must not alias any other live BitBoard – caller's responsibility.
  void load_raw(const int8_t * __restrict__ data)
  {
    white_ = 0;
    black_ = 0;
    for (int i = 0; i < 64; ++i) {
      uint64_t bit = 1ULL << i;
      if      (data[i] ==  1) white_ |= bit;
      else if (data[i] == -1) black_ |= bit;
    }
  }
};

// =============================================================================
// Batch helpers (OpenMP parallelised, new functions – additive only)
// =============================================================================

// Helper: decode one board from the flat 3-D array slice.
static inline void load_board_slice(BitBoard &bb,
                                    const int8_t * __restrict__ base,
                                    py::ssize_t board_idx)
{
  bb.load_raw(base + board_idx * 64);
}

// -----------------------------------------------------------------------------
// batch_get_legal_moves_mask
//   boards : numpy array of shape (N, 8, 8), dtype int8
//   color  : 1 = white, -1 = black
//   returns: list of N uint64_t legal-move masks
// -----------------------------------------------------------------------------
static std::vector<uint64_t>
batch_get_legal_moves_mask(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> boards,
    int color)
{
  auto buf = boards.unchecked<3>();
  if (UNLIKELY(buf.shape(1) != 8 || buf.shape(2) != 8))
    throw std::runtime_error("batch_get_legal_moves_mask: boards must be (N,8,8)");

  const py::ssize_t N    = buf.shape(0);
  const int8_t     *base = buf.data(0, 0, 0);

  std::vector<uint64_t> result(static_cast<std::size_t>(N));

#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 64) default(none) \
      shared(base, result, color) firstprivate(N)
#endif
  for (py::ssize_t i = 0; i < N; ++i) {
    BitBoard bb;
    load_board_slice(bb, base, i);
    result[static_cast<std::size_t>(i)] = bb.get_legal_moves_mask(color);
  }
  return result;
}

// -----------------------------------------------------------------------------
// batch_count_diff
//   boards : numpy array of shape (N, 8, 8), dtype int8
//   color  : 1 = white, -1 = black
//   returns: list of N score differences (player - opponent piece count)
// -----------------------------------------------------------------------------
static std::vector<int>
batch_count_diff(
    py::array_t<int8_t, py::array::c_style | py::array::forcecast> boards,
    int color)
{
  auto buf = boards.unchecked<3>();
  if (UNLIKELY(buf.shape(1) != 8 || buf.shape(2) != 8))
    throw std::runtime_error("batch_count_diff: boards must be (N,8,8)");

  const py::ssize_t N    = buf.shape(0);
  const int8_t     *base = buf.data(0, 0, 0);

  std::vector<int> result(static_cast<std::size_t>(N));

#ifdef _OPENMP
  #pragma omp parallel for schedule(dynamic, 64) default(none) \
      shared(base, result, color) firstprivate(N)
#endif
  for (py::ssize_t i = 0; i < N; ++i) {
    BitBoard bb;
    load_board_slice(bb, base, i);
    result[static_cast<std::size_t>(i)] = bb.count_diff(color);
  }
  return result;
}

// =============================================================================
// pybind11 module
// =============================================================================
PYBIND11_MODULE(othello_bitboard, m)
{
  m.doc() = "High-performance 8x8 Othello bitboard (OpenMP + SIMD edition)";

  // --------------------------------------------------------------------------
  // Existing class – all original bindings preserved exactly as before
  // --------------------------------------------------------------------------
  py::class_<BitBoard>(m, "BitBoard")
      .def(py::init<int>(), py::arg("size") = 8)
      .def("get_legal_moves_list", &BitBoard::get_legal_moves_list,
           py::arg("color"))
      .def("has_legal_moves",      &BitBoard::has_legal_moves,
           py::arg("color"))
      .def("execute_move",         &BitBoard::execute_move,
           py::arg("x"), py::arg("y"), py::arg("color"))
      .def("count_diff",           &BitBoard::count_diff,
           py::arg("color"))
      .def("to_numpy",             &BitBoard::to_numpy)
      .def("from_numpy",           &BitBoard::from_numpy,
           py::arg("arr"))
      .def("hash",                 &BitBoard::hash);

  // --------------------------------------------------------------------------
  // New batch functions (additive – do not change existing bindings above)
  // --------------------------------------------------------------------------
  m.def("batch_get_legal_moves_mask", &batch_get_legal_moves_mask,
        py::arg("boards"), py::arg("color"),
        R"doc(
Compute legal-move bitmasks for a batch of boards in parallel.

Parameters
----------
boards : numpy.ndarray, shape (N, 8, 8), dtype int8
    Batch of board states.  1 = white, -1 = black, 0 = empty.
color  : int
    Player whose moves to compute (1 = white, -1 = black).

Returns
-------
list of int (uint64)
    One 64-bit legal-move mask per board.
)doc");

  m.def("batch_count_diff", &batch_count_diff,
        py::arg("boards"), py::arg("color"),
        R"doc(
Compute score differences (player − opponent piece count) for a batch of boards.

Parameters
----------
boards : numpy.ndarray, shape (N, 8, 8), dtype int8
color  : int  (1 = white, -1 = black)

Returns
-------
list of int
    One score difference per board.
)doc");
}