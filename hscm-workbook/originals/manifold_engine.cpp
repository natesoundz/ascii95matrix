/*
 * HSCM Manifold Engine — C++ 5-Layer Exhaustion
 *
 * Algorithm:
 *   L0: seed all 65 primitives → valid_cells[0]
 *   L1: for each L0 cell, expand to valid L1 roles → valid_cells[1] (L0×L1 pairs)
 *   L2: for each (L0,L1) pair, expand via phonetic sequences from word stream
 *       → valid_cells[2] (triples)
 *   L3: for each triple, cross with geometric keys → valid_cells[3]
 *   L4: for each quad, cross with trajectory combos → valid_cells[4]
 *
 *   At each layer: dead paths zeroed out. Only valid intersections survive.
 *   Output: binary index of all valid 5-tuples (the compiled manifold)
 *
 * Communication:
 *   - Reads matrix config from JSON file (written by Python orchestrator)
 *   - Reads word+phoneme stream from stdin (pipe from ingestor)
 *   - Writes checkpoint files after each layer
 *   - Writes progress to stderr (line-buffered for monitoring)
 *   - Writes compiled index to binary file
 *
 * Compile:
 *   g++ -O3 -std=c++17 -o build/manifold_engine manifold_engine.cpp
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <map>
#include <cstdint>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <functional>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using MS    = std::chrono::milliseconds;

// =============================================================================
// TYPES
// =============================================================================

using PrimID  = uint8_t;   // L0: 0-64
using SubPOS  = uint8_t;   // L1: 0-44
using PhonSeq = std::vector<uint8_t>; // L2: ordered sequence of phoneme IDs
using GeoKey  = uint8_t;   // L3: 0-13
using TrajSet = uint8_t;   // L4: bitmask of 10 trajectory bits

struct Coordinate {
    PrimID  l0;
    SubPOS  l1;
    PhonSeq l2;
    GeoKey  l3;
    TrajSet l4;
    std::string surface_form;

    bool operator==(const Coordinate& o) const {
        return l0==o.l0 && l1==o.l1 && l2==o.l2 && l3==o.l3 && l4==o.l4;
    }
};

// Coordinate hash for deduplication
struct CoordHash {
    size_t operator()(const Coordinate& c) const {
        size_t h = std::hash<uint8_t>{}(c.l0);
        h ^= std::hash<uint8_t>{}(c.l1) << 8;
        for (auto p : c.l2) h ^= std::hash<uint8_t>{}(p) * 2654435761ULL;
        h ^= std::hash<uint8_t>{}(c.l3) << 24;
        h ^= std::hash<uint8_t>{}(c.l4) << 16;
        return h;
    }
};

// =============================================================================
// MATRIX DEFINITION (hardcoded — matches matrix.py exactly)
// =============================================================================

// L0 primitive IDs
const std::unordered_map<std::string, PrimID> PRIM_IDS = {
    {"THING",0},{"SOMEBODY",1},{"PEOPLE",2},{"BODY",3},{"KIND",4},{"PART",5},
    {"THIS",6},{"THE_SAME",7},{"OTHER",8},
    {"ONE",9},{"TWO",10},{"SOME",11},{"ALL",12},{"MUCH",13},{"MANY",14},
    {"GOOD",15},{"BAD",16},{"BIG",17},{"SMALL",18},
    {"THINK",19},{"KNOW",20},{"WANT",21},{"FEEL",22},{"SEE",23},{"HEAR",24},
    {"SAY",25},{"WORDS",26},{"TRUE",27},
    {"DO",28},{"HAPPEN",29},{"MOVE",30},{"TOUCH",31},
    {"BE",32},{"THERE_IS",33},{"HAVE",34},
    {"LIVE",35},{"DIE",36},
    {"WHEN",37},{"NOW",38},{"BEFORE",39},{"AFTER",40},
    {"A_LONG_TIME",41},{"A_SHORT_TIME",42},{"FOR_SOME_TIME",43},{"MOMENT",44},
    {"WHERE",45},{"HERE",46},{"ABOVE",47},{"BELOW",48},
    {"FAR",49},{"NEAR",50},{"SIDE",51},{"INSIDE",52},{"OUTSIDE",53},
    {"NOT",54},{"MAYBE",55},{"VERY",56},{"MORE",57},
    {"LIKE",58},{"AS",59},{"IF",60},{"BECAUSE",61},
    {"CAN",62},{"SHOULD",63},{"BECOME",64}
};

// L1 SubPOS IDs
const std::unordered_map<std::string, SubPOS> SUBPOS_IDS = {
    {"NOUN_CONCRETE",0},{"NOUN_ABSTRACT",1},{"NOUN_COLLECTIVE",2},{"NOUN_MASS",3},
    {"NOUN_PROPER",4},{"NOUN_ANIMATE",5},{"NOUN_INANIMATE",6},{"NOUN_PROCESS",7},
    {"NOUN_STATE",8},{"NOUN_EVENT",9},{"NOUN_ROLE",10},
    {"VERB_ACTION",11},{"VERB_STATE",12},{"VERB_PROCESS",13},{"VERB_CAUSATIVE",14},
    {"VERB_MENTAL",15},{"VERB_SPEECH",16},{"VERB_MOTION",17},{"VERB_PERCEPTION",18},
    {"VERB_EXISTENCE",19},{"VERB_POSSESSION",20},{"VERB_CHANGE",21},
    {"ADJ_EVALUATIVE",22},{"ADJ_DESCRIPTIVE",23},{"ADJ_RELATIONAL",24},
    {"ADJ_TEMPORAL",25},{"ADJ_SPATIAL",26},{"ADJ_QUANTITATIVE",27},
    {"ADJ_AFFECTIVE",28},{"ADJ_MODAL",29},
    {"ADV_MANNER",30},{"ADV_DEGREE",31},{"ADV_TEMPORAL",32},{"ADV_SPATIAL",33},
    {"ADV_FREQUENCY",34},{"ADV_MODAL",35},{"ADV_EVALUATIVE",36},
    {"PRONOUN_PERSONAL",37},{"PRONOUN_RELATIVE",38},
    {"PREPOSITION_SPATIAL",39},{"PREPOSITION_TEMPORAL",40},
    {"CONJUNCTION_COORD",41},{"CONJUNCTION_SUB",42},
    {"DETERMINER_DEF",43},{"PARTICLE",44}
};

// L2 phoneme IDs (CMU phoneme set mapped to our operators)
const std::unordered_map<std::string, uint8_t> PHONEME_IDS = {
    // Plosives
    {"P",0},{"B",1},{"T",2},{"D",3},{"K",4},{"G",5},
    // Fricatives
    {"F",6},{"V",7},{"S",8},{"Z",9},{"SH",10},{"ZH",11},{"TH",12},{"DH",13},{"HH",14},
    // Nasals
    {"M",15},{"N",16},{"NG",17},
    // Liquids/Glides
    {"L",18},{"R",19},{"W",20},{"Y",21},
    // Vowels — CMU vowel phonemes mapped to our 4 fold classes
    // HIGH_FRONT=22
    {"IY",22},{"IH",22},{"EY",22},
    // MID=23
    {"EH",23},{"AH",23},{"ER",23},{"AX",23},
    // LOW=24
    {"AE",24},{"AA",24},{"AW",24},{"AY",24},
    // HIGH_BACK=25
    {"OW",25},{"OY",25},{"UH",25},{"UW",25},{"UX",25}
};

// Valid L0→L1 edges (mirrors VALID_L0_L1_EDGES in matrix.py)
const std::unordered_map<PrimID, std::vector<SubPOS>> VALID_EDGES = {
    {30, {17,11,7,9}},   // MOVE
    {28, {11,14,9}},     // DO
    {29, {13,12,9}},     // HAPPEN
    {19, {15,7,8}},      // THINK
    {20, {15,12,8,1}},   // KNOW
    {21, {15,12,8}},     // WANT
    {22, {15,18,8,1}},   // FEEL
    {23, {18,15,9}},     // SEE
    {24, {18,9}},        // HEAR
    {25, {16,9,7}},      // SAY
    {15, {22,28,36,1}},  // GOOD
    {16, {22,28,36,1}},  // BAD
    {17, {23,27}},       // BIG
    {18, {23,27}},       // SMALL
    {0,  {0,1,6}},       // THING
    {1,  {5,37}},        // SOMEBODY
    {34, {20,19}},       // HAVE
    {32, {19,12}},       // BE
    {64, {21,13}},       // BECOME
    {35, {12,11,19}},    // LIVE
    {36, {21,13,9}},     // DIE
    {31, {11,18,9}},     // TOUCH
    {56, {31}},          // VERY
    {57, {31,27}},       // MORE
    {54, {44,35}},       // NOT
    {55, {35,44}},       // MAYBE
    {46, {33,39}},       // HERE
    {50, {26,33,39}},    // NEAR
    {49, {26,33}},       // FAR
    {38, {32}},          // NOW
    {39, {32,40,42}},    // BEFORE
    {40, {32,40,42}},    // AFTER
    {26, {1,0}},         // WORDS
    {27, {22,1}},        // TRUE
    {58, {15,39,41}},    // LIKE
    {61, {42}},          // BECAUSE
    {60, {42}},          // IF
    {62, {29,35}},       // CAN
    {63, {29,35}},       // SHOULD
};

// =============================================================================
// PROGRESS MONITOR
// =============================================================================

struct Monitor {
    std::string phase;
    size_t total = 0;
    size_t processed = 0;
    size_t valid_cells = 0;
    size_t dead_ends = 0;
    Clock::time_point start_time;

    void begin(const std::string& p, size_t t = 0) {
        phase = p;
        total = t;
        processed = 0;
        valid_cells = 0;
        dead_ends = 0;
        start_time = Clock::now();
        log("BEGIN");
    }

    void tick(size_t valid_delta = 0, size_t dead_delta = 0) {
        processed++;
        valid_cells += valid_delta;
        dead_ends   += dead_delta;
        if (processed % 1000 == 0) log("PROGRESS");
    }

    void end() { log("END"); }

    void log(const std::string& event) {
        auto elapsed = std::chrono::duration_cast<MS>(
            Clock::now() - start_time).count();
        std::cerr << "[HSCM|" << phase << "|" << event << "] "
                  << "processed=" << processed
                  << " valid=" << valid_cells
                  << " dead_ends=" << dead_ends
                  << " elapsed_ms=" << elapsed;
        if (total > 0 && processed > 0)
            std::cerr << " pct=" << (100*processed/total) << "%";
        std::cerr << "\n";
        std::cerr.flush();
    }
};

// =============================================================================
// CHECKPOINT I/O
// =============================================================================

void write_checkpoint(const std::string& path,
                      const std::vector<std::pair<PrimID,SubPOS>>& cells,
                      int layer) {
    std::ofstream f(path, std::ios::binary);
    uint32_t n = cells.size();
    f.write(reinterpret_cast<const char*>(&layer), sizeof(layer));
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (auto& [l0, l1] : cells) {
        f.write(reinterpret_cast<const char*>(&l0), 1);
        f.write(reinterpret_cast<const char*>(&l1), 1);
    }
    std::cerr << "[CHECKPOINT] Layer " << layer << " written: "
              << cells.size() << " valid cells → " << path << "\n";
}

void write_coordinate_index(const std::string& path,
                             const std::unordered_set<Coordinate, CoordHash>& coords) {
    std::ofstream f(path, std::ios::binary);
    uint32_t n = coords.size();
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    for (auto& c : coords) {
        f.write(reinterpret_cast<const char*>(&c.l0), 1);
        f.write(reinterpret_cast<const char*>(&c.l1), 1);
        uint8_t seq_len = c.l2.size();
        f.write(reinterpret_cast<const char*>(&seq_len), 1);
        for (auto ph : c.l2)
            f.write(reinterpret_cast<const char*>(&ph), 1);
        f.write(reinterpret_cast<const char*>(&c.l3), 1);
        f.write(reinterpret_cast<const char*>(&c.l4), 1);
        // Write surface form
        uint8_t sf_len = std::min(c.surface_form.size(), (size_t)255);
        f.write(reinterpret_cast<const char*>(&sf_len), 1);
        f.write(c.surface_form.c_str(), sf_len);
    }
}

// =============================================================================
// PHONEME SEQUENCE PARSER (from CMU phoneme strings)
// =============================================================================

PhonSeq parse_cmu_phonemes(const std::vector<std::string>& phonemes) {
    PhonSeq seq;
    for (const auto& ph : phonemes) {
        // Strip stress markers (0,1,2)
        std::string clean = ph;
        if (!clean.empty() && std::isdigit(clean.back()))
            clean.pop_back();
        auto it = PHONEME_IDS.find(clean);
        if (it != PHONEME_IDS.end())
            seq.push_back(it->second);
    }
    return seq;
}

// =============================================================================
// L4 TRAJECTORY COMBINATIONS
// Precompute valid non-contradicting TrajSet bitmasks
// =============================================================================

std::vector<TrajSet> build_valid_trajectory_sets() {
    // 5 axes, 2 poles each
    // Axis 0: VEL_FAST(0), VEL_SLOW(1)
    // Axis 1: MAG_LARGE(2), MAG_SMALL(3)
    // Axis 2: DIR_OUT(4), DIR_IN(5)
    // Axis 3: AFF_POS(6), AFF_NEG(7)
    // Axis 4: PER_STATIC(8), PER_DYNAMIC(9)
    // Valid: at most one pole per axis, any subset of axes

    std::vector<TrajSet> valid;
    for (int mask = 0; mask < 1024; mask++) {
        bool ok = true;
        // Check no axis has both poles
        for (int axis = 0; axis < 5; axis++) {
            int bit0 = (mask >> (axis*2)) & 1;
            int bit1 = (mask >> (axis*2+1)) & 1;
            if (bit0 && bit1) { ok = false; break; }
        }
        if (ok) valid.push_back(static_cast<TrajSet>(mask));
    }
    return valid;
}

// =============================================================================
// L3 GEOMETRIC KEYS — which keys are valid per etymology guess
// (full implementation needs etymology DB; here we allow all 14 for each word
//  and let the ingestor narrow it when etymology is known)
// =============================================================================

const std::vector<GeoKey> ALL_GEO_KEYS = {0,1,2,3,4,5,6,7,8,9,10,11,12,13};

// =============================================================================
// MAIN ENGINE
// =============================================================================

int main(int argc, char* argv[]) {
    std::string checkpoint_dir = "checkpoints";
    std::string index_path     = "index/manifold.bin";
    bool ingest_mode           = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--ingest")         ingest_mode    = true;
        if (arg.find("--checkpoints=") == 0) checkpoint_dir = arg.substr(14);
        if (arg.find("--index=") == 0)       index_path     = arg.substr(8);
    }

    Monitor mon;

    // =========================================================================
    // LAYER 0: Seed all 65 primitives
    // =========================================================================

    mon.begin("L0_SEED", PRIM_IDS.size());
    std::vector<PrimID> l0_cells;
    for (auto& [name, id] : PRIM_IDS) {
        l0_cells.push_back(id);
        mon.tick(1);
    }
    mon.end();
    std::cerr << "[L0] " << l0_cells.size() << " primitive seeds\n";

    // =========================================================================
    // LAYER 1: Expand L0 → valid L0×L1 pairs
    // =========================================================================

    mon.begin("L1_EXPAND", l0_cells.size());
    std::vector<std::pair<PrimID,SubPOS>> l1_cells;

    for (auto l0 : l0_cells) {
        auto it = VALID_EDGES.find(l0);
        if (it == VALID_EDGES.end()) {
            mon.tick(0, 1);
            continue;
        }
        for (auto l1 : it->second) {
            l1_cells.emplace_back(l0, l1);
        }
        mon.tick(it->second.size());
    }
    mon.end();

    write_checkpoint(checkpoint_dir + "/layer1.bin", l1_cells, 1);
    std::cerr << "[L1] " << l1_cells.size() << " valid (L0,L1) pairs\n";

    // =========================================================================
    // LAYER 2: Ingest word stream → assign L2 phoneme sequences
    // Reads from stdin: one word per line, format: WORD L0_NAME L1_NAME PH1 PH2...
    // (Written by Python ingestor)
    // =========================================================================

    mon.begin("L2_INGEST");
    std::unordered_set<Coordinate, CoordHash> l2_cells;

    // Build lookup: (l0,l1) → valid pair exists
    std::set<std::pair<PrimID,SubPOS>> valid_l1_set(l1_cells.begin(), l1_cells.end());

    if (ingest_mode) {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream ss(line);
            std::string word, prim_name, subpos_name;
            ss >> word >> prim_name >> subpos_name;

            auto pit = PRIM_IDS.find(prim_name);
            auto sit = SUBPOS_IDS.find(subpos_name);
            if (pit == PRIM_IDS.end() || sit == SUBPOS_IDS.end()) {
                mon.tick(0, 1);
                continue;
            }

            PrimID l0  = pit->second;
            SubPOS l1  = sit->second;

            if (!valid_l1_set.count({l0, l1})) {
                mon.tick(0, 1);
                continue;
            }

            // Read phonemes
            std::vector<std::string> phoneme_strs;
            std::string ph;
            while (ss >> ph) phoneme_strs.push_back(ph);
            PhonSeq l2 = parse_cmu_phonemes(phoneme_strs);

            if (l2.empty()) {
                mon.tick(0, 1);
                continue;
            }

            // Create coordinate with placeholder L3/L4 (will be set in L3/L4 passes)
            Coordinate c;
            c.l0 = l0; c.l1 = l1; c.l2 = l2;
            c.l3 = 255; c.l4 = 0;
            c.surface_form = word;
            l2_cells.insert(c);
            mon.tick(1);
        }
    } else {
        std::cerr << "[L2] Skipping ingest (no --ingest flag). "
                  << "Run with --ingest and pipe word stream from orchestrator.\n";
    }
    mon.end();
    std::cerr << "[L2] " << l2_cells.size() << " unique (L0,L1,L2) triples\n";

    // =========================================================================
    // LAYER 3: Cross with geometric keys
    // For each (L0,L1,L2) triple, assign all valid L3 keys
    // (narrowed when etymology data is available)
    // =========================================================================

    mon.begin("L3_EXPAND", l2_cells.size());
    std::unordered_set<Coordinate, CoordHash> l3_cells;

    if (!l2_cells.empty()) {
        for (const auto& c2 : l2_cells) {
            for (auto geo : ALL_GEO_KEYS) {
                Coordinate c = c2;
                c.l3 = geo;
                l3_cells.insert(c);
                mon.tick(1);
            }
        }
    } else {
        // Demo mode: generate synthetic coordinates from L1 cells
        std::cerr << "[L3] Demo mode: generating synthetic coordinates\n";
        auto traj_sets = build_valid_trajectory_sets();
        for (auto& [l0, l1] : l1_cells) {
            for (auto geo : ALL_GEO_KEYS) {
                Coordinate c;
                c.l0 = l0; c.l1 = l1;
                c.l2 = {};  // empty phoneme sequence in demo mode
                c.l3 = geo; c.l4 = 0;
                l3_cells.insert(c);
                mon.tick(1);
            }
        }
    }
    mon.end();
    std::cerr << "[L3] " << l3_cells.size() << " (L0,L1,L2,L3) quads\n";

    // =========================================================================
    // LAYER 4: Cross with trajectory sets → full 5-tuple coordinates
    // =========================================================================

    mon.begin("L4_EXPAND", l3_cells.size());
    std::unordered_set<Coordinate, CoordHash> manifold;
    auto traj_sets = build_valid_trajectory_sets();

    for (const auto& c3 : l3_cells) {
        for (auto traj : traj_sets) {
            Coordinate c = c3;
            c.l4 = traj;
            manifold.insert(c);
            mon.tick(1);
        }
    }
    mon.end();
    std::cerr << "[L4] " << manifold.size() << " full coordinates in manifold\n";

    // =========================================================================
    // CONVERGENCE POINT: Write compiled manifold to binary index
    // =========================================================================

    fs::create_directories(fs::path(index_path).parent_path());
    write_coordinate_index(index_path, manifold);

    std::cerr << "[MANIFOLD] Compiled. "
              << manifold.size() << " valid coordinates → " << index_path << "\n";
    std::cerr << "[MANIFOLD] Theoretical space: 65×45×28×14×10 = 11,466,000\n";
    std::cerr << "[MANIFOLD] Occupied: " << manifold.size()
              << " (" << (100.0 * manifold.size() / 11466000.0) << "%)\n";

    // Write summary for Python orchestrator
    std::cout << "MANIFOLD_COMPLETE "
              << manifold.size() << " "
              << index_path << "\n";
    std::cout.flush();

    return 0;
}
