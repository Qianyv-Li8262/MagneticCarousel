// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 Qianyv-Li8262

// Standalone comparison of the real advancement/reduction functions against
// 65,536 physical sphere modes. Compile as C++ with AVX2, or as CUDA (-x cu).
// Usage: test_eddy_many_modes <output.csv> [physical_mode_count]
// Override IYPT16_EDDY_TEST_DT at compile time for timestep convergence checks.
// Writes numerical data and a JSON summary, never plots. No solver integration.
#include "eddy_currents.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace {
using Triple = std::array<double, 3>;
using Complex = std::complex<double>;
constexpr double pi = hysteresis_physics::pi;
#ifndef IYPT16_EDDY_TEST_DT
#define IYPT16_EDDY_TEST_DT 1e-5
#endif
constexpr double dt_seconds = IYPT16_EDDY_TEST_DT;
static_assert(dt_seconds>0 && dt_seconds<=1e-3, "test dt must be in (0, 1 ms]");
constexpr int steps = static_cast<int>(.25/dt_seconds+.5);
constexpr int sample_stride = (steps+2499)/2500;
constexpr int dense_steps = static_cast<int>(.002/dt_seconds+.5);
struct Tone { int axis; double hz, amplitude, phase; };
constexpr std::array<Tone, 6> tones{{
    {0, 7, .020, .2}, {0, 193, .004, -.6},
    {1, 19, .016, -.4}, {1, 300, .005, .7},
    {2, 43, .018, .9}, {2, 117, .003, .1}}};

double magnitude(const Triple &v) { return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
Triple difference(const Triple &a, const Triple &b) { return {a[0]-b[0], a[1]-b[1], a[2]-b[2]}; }
Triple field(double t)
{
    Triple h{};
    for (const auto &tone : tones)
        h[tone.axis] += tone.amplitude * std::sin(2*pi*tone.hz*t + tone.phase);
    return h;
}
Complex amplitude(const Tone &tone)
{
    return tone.amplitude * Complex(std::sin(tone.phase), -std::cos(tone.phase));
}

// Stable phase equation equivalent to (mu-1)j1(q)+q*j0(q)=0.
// q=n*pi+delta avoids evaluating sin/cos of very large root arguments.
double physical_root(int n, double mu)
{
    double lo=0, hi=pi/2;
    for (int i=0; i<64; ++i) {
        const double mid=(lo+hi)/2;
        if (mid==lo || mid==hi) break;
        const double q=n*pi+mid;
        const double residual=mid-std::atan((mu-1)*q/(q*q+mu-1));
        if (residual>0) hi=mid; else lo=mid;
    }
    return n*pi+(lo+hi)/2;
}

// Exact discrete transfer for the SAME piecewise-linear input used by the
// native function: y[k+1]=exp(-r)y[k] - ((1-exp(-r))/r)*(h[k+1]-h[k]).
Complex discrete_gain(double r, double angle)
{
    const double one_minus_decay=-std::expm1(-r);
    const Complex z_minus_one(-2*std::pow(std::sin(angle/2),2), std::sin(angle));
    return -(one_minus_decay/r)*z_minus_one/(one_minus_decay+z_minus_one);
}

struct Reference {
    std::vector<double> rates, weights; // Physical time rates, not tau* rates.
    std::vector<Triple> startup;
    std::array<Complex, tones.size()> response{};
    std::vector<double> initial_modes;

    void prepare()
    {
        startup.resize(rates.size());
        initial_modes.resize(3*rates.size());
        // Compensated summation keeps tiny high-mode contributions visible.
        std::array<Complex, tones.size()> correction{};
        for (std::size_t j=0; j<rates.size(); ++j) {
            for (std::size_t f=0; f<tones.size(); ++f) {
                const auto &tone=tones[f];
                const Complex g=discrete_gain(dt_seconds*rates[j], 2*pi*tone.hz*dt_seconds);
                const Complex term=weights[j]*g-correction[f];
                const Complex next=response[f]+term;
                correction[f]=(next-response[f])-term;
                response[f]=next;
                const double initial=(g*amplitude(tone)).real();
                startup[j][tone.axis] += weights[j]*initial;
                initial_modes[tone.axis*rates.size()+j] += initial;
            }
        }
    }

    Triple evaluate(int step, bool steady) const
    {
        if (step==0 && !steady) return {};
        const double t=step*dt_seconds;
        Triple result{};
        for (std::size_t f=0; f<tones.size(); ++f) {
            const auto &tone=tones[f];
            result[tone.axis] += (response[f]*amplitude(tone)*std::polar(1.0,2*pi*tone.hz*t)).real();
        }
        if (!steady) {
            for (std::size_t j=0; j<rates.size(); ++j) {
                const double r=t*rates[j];
                if (r>746) break; // Sorted modes: all subsequent exponentials underflow.
                const double decay=std::exp(-r);
                for (int axis=0; axis<3; ++axis) result[axis] -= decay*startup[j][axis];
            }
        }
        return result;
    }
};

#ifdef __CUDACC__
void cuda_check(cudaError_t error)
{
    if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
#endif

// Both builds call the actual public advancement AND magnetic-moment interface.
DEVICE void native_trace(int which, const real_t *derivatives, const double *initial,
                         const int *samples, int sample_count, double *output)
{
    State<kEddyCurrentModeCount> state{};
    if (which==1) {
#ifndef __CUDACC__
        for (int group=0; group<kEddyCurrentModeCount; ++group) {
            state.yx[group]=_mm256_loadu_pd(initial+4*group);
            state.yy[group]=_mm256_loadu_pd(initial+eddy_coefficients::mode_count+4*group);
            state.yz[group]=_mm256_loadu_pd(initial+2*eddy_coefficients::mode_count+4*group);
        }
#else
        for (int j=0; j<eddy_coefficients::mode_count; ++j) {
            state.yx[j]=static_cast<real_t>(initial[j]);
            state.yy[j]=static_cast<real_t>(initial[eddy_coefficients::mode_count+j]);
            state.yz[j]=static_cast<real_t>(initial[2*eddy_coefficients::mode_count+j]);
        }
#endif
    }
    int next=0;
    for (int step=0; step<=steps; ++step) {
        if (step>0) {
            const Vec3 hb_dot={derivatives[3*(step-1)], derivatives[3*(step-1)+1], derivatives[3*(step-1)+2]};
            advance_eddy_current(state.yx,state.yy,state.yz,
                static_cast<real_t>(dt_seconds*hysteresis_physics::omega_ref),hb_dot,REAL_LITERAL(1.0));
        }
        if (next<sample_count && step==samples[next]) {
            const Vec3 m=eval_eddy_current_moment(state.yx,state.yy,state.yz);
            const int offset=3*(which*sample_count+next);
            output[offset]=static_cast<double>(m.x);
            output[offset+1]=static_cast<double>(m.y);
            output[offset+2]=static_cast<double>(m.z);
            ++next;
        }
    }
}
#ifdef __CUDACC__
__global__ void trace_kernel(const real_t *derivatives, const double *initial,
                             const int *samples, int sample_count, double *output)
{
    if (threadIdx.x==0 && blockIdx.x<2)
        native_trace(static_cast<int>(blockIdx.x),derivatives,initial,samples,sample_count,output);
}
#endif

Complex analytic_response(double omega, double mu, double td)
{
    const Complex x=std::sqrt(Complex(0,-omega*td));
    const Complex j0=std::sin(x)/x;
    const Complex j1=(std::sin(x)-x*std::cos(x))/(x*x);
    return 1.5*((2*mu+1)*j1-x*j0)/((mu-1)*j1+x*j0)-3*(mu-1)/(mu+2);
}
Complex continuous_response(const Reference &reference, double omega)
{
    Complex sum{}, correction{};
    for (std::size_t j=0; j<reference.rates.size(); ++j) {
        const Complex v(0,omega/reference.rates[j]);
        const Complex term=-reference.weights[j]*v/(1.0+v)-correction;
        const Complex next=sum+term;
        correction=(next-sum)-term;
        sum=next;
    }
    return sum;
}

struct Metrics {
    double rms_relative=0, peak_absolute=0, peak_normalized=0, peak_time=0;
    double peak_reference=0, max_angle_degrees=0;
    double max_pointwise_relative=0, max_pointwise_time=0;
};
} // namespace

int main(int argc, char **argv)
try {
    const std::filesystem::path output=argc>1 ? argv[1] : "eddy_many_modes.csv";
    const int count=argc>2 ? std::stoi(argv[2]) : 65536;
    if (count<64 || count%2!=0) throw std::runtime_error("physical mode count must be even and >= 64");
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    const double omega_ref=hysteresis_physics::omega_ref;
    const double mu=eddy_coefficients::mu_r;
    const double td=hysteresis_physics::mu0*mu*eddy_coefficients::sigma_si*
                    eddy_coefficients::reference_radius_m*eddy_coefficients::reference_radius_m;
    Reference compressed, full;
    compressed.rates.resize(eddy_coefficients::mode_count);
    compressed.weights.resize(eddy_coefficients::mode_count);
#ifdef __CUDACC__
    cuda_check(cudaMemcpyFromSymbol(compressed.rates.data(),eddy_coefficients::inv_tau_ref_star,
                                    compressed.rates.size()*sizeof(double)));
    cuda_check(cudaMemcpyFromSymbol(compressed.weights.data(),eddy_coefficients::aj,
                                    compressed.weights.size()*sizeof(double)));
#else
    std::copy_n(eddy_coefficients::inv_tau_ref_star,compressed.rates.size(),compressed.rates.begin());
    std::copy_n(eddy_coefficients::aj,compressed.weights.size(),compressed.weights.begin());
#endif
    for (double &rate : compressed.rates) rate*=omega_ref;
    full.rates.resize(count); full.weights.resize(count);
    for (int j=0; j<count; ++j) {
        const double q=physical_root(j+1,mu);
        full.rates[j]=q*q/td;
        full.weights[j]=9*mu/(q*q+(mu-1)*(mu+2));
    }
    Reference half;
    half.rates.assign(full.rates.begin(),full.rates.begin()+count/2);
    half.weights.assign(full.weights.begin(),full.weights.begin()+count/2);
    compressed.prepare(); full.prepare(); half.prepare();

    std::vector<int> samples{0};
    std::vector<real_t> derivatives(3*steps);
    Triple last=field(0);
    for (int step=1; step<=steps; ++step) {
        const Triple current=field(step*dt_seconds);
        for (int axis=0; axis<3; ++axis)
            derivatives[3*(step-1)+axis]=static_cast<real_t>((current[axis]-last[axis])/(dt_seconds*omega_ref));
        last=current;
        if (step<=dense_steps || step%sample_stride==0 || step==steps) samples.push_back(step);
    }
    std::vector<double> trace(2*samples.size()*3);
#ifdef __CUDACC__
    real_t *d_derivatives; double *d_initial, *d_trace; int *d_samples;
    cuda_check(cudaMalloc(&d_derivatives,derivatives.size()*sizeof(real_t)));
    cuda_check(cudaMalloc(&d_initial,compressed.initial_modes.size()*sizeof(double)));
    cuda_check(cudaMalloc(&d_trace,trace.size()*sizeof(double)));
    cuda_check(cudaMalloc(&d_samples,samples.size()*sizeof(int)));
    cuda_check(cudaMemcpy(d_derivatives,derivatives.data(),derivatives.size()*sizeof(real_t),cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_initial,compressed.initial_modes.data(),compressed.initial_modes.size()*sizeof(double),cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(d_samples,samples.data(),samples.size()*sizeof(int),cudaMemcpyHostToDevice));
    trace_kernel<<<2,1>>>(d_derivatives,d_initial,d_samples,static_cast<int>(samples.size()),d_trace);
    cuda_check(cudaGetLastError());
    cuda_check(cudaMemcpy(trace.data(),d_trace,trace.size()*sizeof(double),cudaMemcpyDeviceToHost));
    cuda_check(cudaFree(d_derivatives)); cuda_check(cudaFree(d_initial));
    cuda_check(cudaFree(d_trace)); cuda_check(cudaFree(d_samples));
    constexpr const char *backend="cuda";
#else
    for (int which=0; which<2; ++which)
        native_trace(which,derivatives.data(),compressed.initial_modes.data(),samples.data(),
                     static_cast<int>(samples.size()),trace.data());
    constexpr const char *backend="cpu_avx2";
#endif

    std::ofstream csv(output);
    if (!csv) throw std::runtime_error("cannot open output CSV");
    csv << "case,time_s,hx,hy,hz,actual_x,actual_y,actual_z,reference_x,reference_y,reference_z,half_x,half_y,half_z,compressed_analytic_x,compressed_analytic_y,compressed_analytic_z\n";
    csv << std::setprecision(17);
    std::array<Metrics,2> metrics{};
    double convergence=0, implementation_error=0, overall_peak=0;
    for (int which=0; which<2; ++which) {
        double error_integral=0, reference_integral=0, previous_e2=0, previous_r2=0;
        for (std::size_t k=0; k<samples.size(); ++k) {
            const int step=samples[k];
            const double t=step*dt_seconds;
            const Triple h=field(t);
            const Triple reference=full.evaluate(step,which==1);
            const Triple half_reference=half.evaluate(step,which==1);
            const Triple analytic=compressed.evaluate(step,which==1);
            const std::size_t offset=3*(which*samples.size()+k);
            const Triple actual{trace[offset],trace[offset+1],trace[offset+2]};
            const Triple error=difference(actual,reference);
            const double en=magnitude(error), rn=magnitude(reference);
            if (!std::isfinite(en)) throw std::runtime_error("non-finite native result");
            if (k>0) {
                const double interval=(step-samples[k-1])*dt_seconds;
                error_integral+=interval*(en*en+previous_e2)/2;
                reference_integral+=interval*(rn*rn+previous_r2)/2;
            }
            previous_e2=en*en; previous_r2=rn*rn;
            auto &m=metrics[which];
            if (en>m.peak_absolute) { m.peak_absolute=en; m.peak_time=t; }
            if (rn>0 && en/rn>m.max_pointwise_relative) {
                m.max_pointwise_relative=en/rn; m.max_pointwise_time=t;
            }
            m.peak_reference=std::max(m.peak_reference,rn);
            const Triple cross{actual[1]*reference[2]-actual[2]*reference[1],
                               actual[2]*reference[0]-actual[0]*reference[2],
                               actual[0]*reference[1]-actual[1]*reference[0]};
            const double dot=actual[0]*reference[0]+actual[1]*reference[1]+actual[2]*reference[2];
            if (rn>0 && magnitude(actual)>0)
                m.max_angle_degrees=std::max(m.max_angle_degrees,std::atan2(magnitude(cross),dot)*180/pi);
            convergence=std::max(convergence,magnitude(difference(reference,half_reference)));
            implementation_error=std::max(implementation_error,magnitude(difference(actual,analytic)));
            overall_peak=std::max(overall_peak,rn);
            csv << (which==0 ? "startup" : "steady") << ',' << t;
            for (const auto &v : {h,actual,reference,half_reference,analytic})
                for (double component : v) csv << ',' << component;
            csv << '\n';
        }
        metrics[which].rms_relative=std::sqrt(error_integral/reference_integral);
        metrics[which].peak_normalized=metrics[which].peak_absolute/metrics[which].peak_reference;
    }
    csv.close();

    auto frequency_path=output; frequency_path.replace_extension("frequency.csv");
    std::ofstream frequency(frequency_path);
    frequency << "frequency_hz,compressed_relative_error,many_mode_relative_error\n" << std::setprecision(17);
    double frequency_error=0, reference_frequency_error=0;
    for (int k=0; k<=128; ++k) {
        const double hz=.1*std::pow(3000.,k/128.);
        const Complex exact=analytic_response(2*pi*hz,mu,td);
        const double e=std::abs(continuous_response(compressed,2*pi*hz)-exact)/std::abs(exact);
        const double er=std::abs(continuous_response(full,2*pi*hz)-exact)/std::abs(exact);
        frequency_error=std::max(frequency_error,e);
        reference_frequency_error=std::max(reference_frequency_error,er);
        frequency << hz << ',' << e << ',' << er << '\n';
    }
    const double convergence_relative=convergence/overall_peak;
    const double implementation_relative=implementation_error/overall_peak;
    std::cout << std::setprecision(17)
        << "{\n  \"backend\":\"" << backend << "\",\n  \"physical_modes\":" << count
        << ",\n  \"compressed_modes\":" << compressed.rates.size()
        << ",\n  \"dt_seconds\":" << dt_seconds << ",\n  \"duration_seconds\":" << steps*dt_seconds
        << ",\n  \"samples_per_case\":" << samples.size()
        << ",\n  \"reference_convergence_relative\":" << convergence_relative
        << ",\n  \"implementation_relative\":" << implementation_relative
        << ",\n  \"frequency_compressed_relative\":" << frequency_error
        << ",\n  \"frequency_reference_relative\":" << reference_frequency_error << ",\n  \"cases\":{\n";
    for (int which=0; which<2; ++which) {
        const auto &m=metrics[which];
        std::cout << "    \"" << (which==0 ? "startup" : "steady") << "\":{\"rms_relative\":" << m.rms_relative
            << ",\"peak_absolute\":" << m.peak_absolute << ",\"peak_normalized\":" << m.peak_normalized
            << ",\"peak_time_seconds\":" << m.peak_time << ",\"max_angle_degrees\":" << m.max_angle_degrees
            << ",\"max_pointwise_relative\":" << m.max_pointwise_relative
            << ",\"max_pointwise_time_seconds\":" << m.max_pointwise_time
            << "}" << (which==0 ? ",\n" : "\n");
    }
    std::cout << "  }\n}\n";
    if (convergence_relative>1e-7 || reference_frequency_error>1e-7 || frequency_error>1e-3)
        throw std::runtime_error("reference convergence or compressed frequency-response check failed");
#ifdef USE_FP64
    if (implementation_relative>1e-10)
        throw std::runtime_error("native advancement/reduction differs from independent compressed solution");
#endif
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
