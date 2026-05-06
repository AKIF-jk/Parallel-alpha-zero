#pragma once
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <shared_mutex>
#include <random>

// Cache note: TTEntry is alignas(64) to reduce false sharing when adjacent buckets are touched.

// Lock-safe model: TT cache read/writes protected by shared_mutex per map.
struct alignas(64) TTEntry {
    float q;
    int n;
    float p[36];
    bool is_terminal;
    float v;
};

class Zobrist {
public:
    uint64_t table[2][36];
    uint64_t black_to_move;

    static const Zobrist& get() {
        static Zobrist instance;
        return instance;
    }

private:
    Zobrist() {
        std::mt19937_64 rng(0x1337);
        for(int c=0; c<2; ++c)
            for(int i=0; i<36; ++i)
                table[c][i] = rng();
        black_to_move = rng();
    }
};

class TranspositionTable {
    std::unordered_map<uint64_t, TTEntry> map;
    mutable std::shared_mutex mtx;

public:
    static TranspositionTable& get() {
        static TranspositionTable instance;
        return instance;
    }

    bool lookup(const uint64_t hash, TTEntry& entry) const {
        std::shared_lock<std::shared_mutex> lock(mtx);
        auto it = map.find(hash);
        if (it != map.end()) {
            entry = it->second;
            return true;
        }
        return false;
    }

    void store(const uint64_t hash, const float q, const int n, const float* const p, const bool is_terminal, const float v = 0.0f) {
        std::unique_lock<std::shared_mutex> lock(mtx);
        TTEntry entry{q, n, {}, is_terminal, v};
        std::memcpy(entry.p, p, sizeof(entry.p));
        map[hash] = entry;
    }
};
