// Isolated benchmark: production headers are compiled without modification.
#include "2dmodel-hysteresis.hpp"
#include <windows.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Query { Vec2 pos; real_t angle; };
using Clock = std::chrono::steady_clock;
volatile double bench_sink = 0;

int interval(const Vec2 &p) {
    return static_cast<int>((length(p) - kLUTLowerBoundRatio * (magnet_number/2)) *
        (kSplineNodes-1) / ((kLUTUpperBoundRatio-kLUTLowerBoundRatio)*(magnet_number/2)));
}

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << 2);
    const std::string kind = argv[2];
    const int count = std::stoi(argv[3]);
    const unsigned seed = static_cast<unsigned>(std::stoul(argv[4]));
    std::vector<LUTNode> lut(kFieldFourierModeCount*kSplineNodes);
    std::ifstream input(argv[1], std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(lut.data()), lut.size()*sizeof(LUTNode))) return 3;
    const auto params = make_solver_params(magnet_number, real_t(1), real_t(1), Eta);
    constexpr real_t dt = 0.0005;
    auto initial = [&] {
        real_t v[kSerializedStateSize]{}; v[1] = magnet_number/2; v[5] = 1;
        if(kind=="trajectory_sliding" || kind=="trajectory_rolling") {
            v[4]=(magnet_number/2)*0.5;
            v[9]=(magnet_number/2)*((kind=="trajectory_sliding" ? 0.2:0.0)-0.5)/params.alpha;
        }
        return read_state_values<kEnableEddyCurrents>(v);
    };
    auto step = [&](auto &state, bool &sliding, Vec2 &dir, bool &valid, FieldCache &cache, FieldResult &field) {
        const auto result = fwd1step(dt, params, state, sliding, dir, valid, &cache, lut.data(), &field);
        if (result != AdvanceStepResult::kStepOk) {
            std::cerr << "step failed: " << static_cast<int>(result) << '\n'; std::exit(4);
        }
    };
    auto init_mode = [&](auto &state, bool &sliding, Vec2 &dir, FieldCache &cache) {
        const Vec2 slip=state.speed+params.alpha*make_Vec2(-state.omega.y,state.omega.x);
        const auto norm=length(slip);
        sliding = norm>1e-10; dir = sliding ? slip/norm : Vec2{};
        if(sliding) return;
        project_pure_rolling(state,params);
        if (resolve_zero_slip_mode(state, params, sliding, dir, &cache, lut.data()) != AdvanceStepResult::kStepOk)
            std::exit(5);
    };
    std::cout << std::setprecision(17);
    if (kind.rfind("trajectory",0)==0) {
        auto state = initial(); bool sliding{}, valid{}; Vec2 dir{};
        FieldCache cache{}; FieldResult field{};
        init_mode(state, sliding, dir, cache);
        const bool chunked=kind!="trajectory";
        auto reset=[&] {
            state=initial(); sliding=false; valid=false; dir={}; cache=FieldCache{};
            init_mode(state,sliding,dir,cache);
        };
        for (int j=0; j<20000; ++j) {
            if(chunked && j%1000==0) reset();
            step(state, sliding, dir, valid, cache, field);
        }
        state=initial(); sliding=false; valid=false; dir={}; cache=FieldCache{};
        init_mode(state, sliding, dir, cache);
        const auto start=Clock::now();
        for (int j=0; j<count; ++j) {
            if(chunked && j%1000==0) reset();
            step(state, sliding, dir, valid, cache, field);
        }
        const double ns=std::chrono::duration<double,std::nano>(Clock::now()-start).count()/count;
        real_t values[kSerializedStateSize]; write_state_values<kEnableEddyCurrents>(state, values);
        std::cout << "{\"ns\":" << ns << ",\"values\":[";
        for (int j=0;j<kSerializedStateSize;++j) std::cout << (j ? ",":"") << values[j];
        std::cout << "]}" << std::endl;
        return 0;
    }
    std::vector<Query> queries(65536);
    std::mt19937 gen(seed); std::uniform_real_distribution<double> uniform(0,1);
    if (kind == "path") {
        auto state=initial(); bool sliding{}, valid{}; Vec2 dir{};
        FieldCache cache{}; FieldResult field{}; init_mode(state,sliding,dir,cache);
        for (auto &q:queries) { q={state.pos,state.down_disk_angle}; step(state,sliding,dir,valid,cache,field); }
    } else {
        for (std::size_t j=0;j<queries.size();++j) {
            const double rho=kind=="same_cell" ? 4+0.002*std::sin(j*0.013) : 2.01+3.98*uniform(gen);
            const double phi=6.283185307179586*uniform(gen);
            queries[j]={{rho*std::cos(phi),rho*std::sin(phi)}, 6.283185307179586*uniform(gen)};
        }
    }
    int hits=0;
    for (std::size_t j=1;j<queries.size();++j) hits+=interval(queries[j].pos)==interval(queries[j-1].pos);
    FieldCache cache{}; double sum=0;
    auto call = [&](int j) {
        const auto &q=queries[static_cast<unsigned>(j)&65535];
        const auto f=eval_mag_field(q.pos,q.angle,&cache,lut.data());
        sum+=f.h.x+f.h.y+f.h.z;
        for (double x:f.hGrad3d.component) sum+=x;
    };
    for (int j=0;j<200000;++j) call(j);
    sum=0; const auto start=Clock::now();
    for (int j=0;j<count;++j) call(j);
    const double ns=std::chrono::duration<double,std::nano>(Clock::now()-start).count()/count;
    bench_sink=sum;
    std::cout << "{\"ns\":" << ns << ",\"hit_rate\":" << double(hits)/(queries.size()-1)
              << ",\"values\":[" << sum << "]}" << std::endl;
}
