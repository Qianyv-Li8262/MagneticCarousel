#include <cstdio>
#include <string>
#include <cuda_runtime.h>
#include "scan_initial.cu"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

void check(cudaError_t status) {
    if(status!=cudaSuccess) { std::cerr<<cudaGetErrorString(status)<<'\n'; std::exit(2); }
}
int main(int argc,char**argv) {
    if(argc!=4) return 1;
    constexpr int k=64, statesize=51, metricsize=20, lutsize=5*204*4;
    const int steps=std::stoi(argv[2]);
    std::vector<double> lut(lutsize),axis(k),states(k*k*statesize),metrics(k*k*metricsize);
    std::vector<int> status(k*k);
    std::ifstream input(argv[1],std::ios::binary);
    if(!input.read(reinterpret_cast<char*>(lut.data()),lut.size()*sizeof(double))) return 3;
    for(int j=0;j<k;++j) axis[j]=0.25+0.75*j/(k-1);
    double *daxis,*dlut,*dstate,*dmetrics; int *dstatus;
    check(cudaMalloc(&daxis,axis.size()*sizeof(double)));
    check(cudaMalloc(&dlut,lut.size()*sizeof(double)));
    check(cudaMalloc(&dstate,states.size()*sizeof(double)));
    check(cudaMalloc(&dmetrics,metrics.size()*sizeof(double)));
    check(cudaMalloc(&dstatus,status.size()*sizeof(int)));
    check(cudaMemcpy(daxis,axis.data(),axis.size()*sizeof(double),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dlut,lut.data(),lut.size()*sizeof(double),cudaMemcpyHostToDevice));
    auto launch=[&] {
        scan_initial_cuda<<<dim3(8,4),dim3(8,16)>>>(daxis,daxis,k,0.0005,steps,steps/2,10,1.,1.,
            reinterpret_cast<LUTNode*>(dlut),dstate,dstatus,dmetrics);
        check(cudaGetLastError());
    };
    launch(); check(cudaDeviceSynchronize());
    cudaEvent_t start,stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    std::cout<<std::setprecision(17)<<"{\"ms\":[";
    for(int j=0;j<3;++j) {
        check(cudaEventRecord(start)); launch(); check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop));
        float ms; check(cudaEventElapsedTime(&ms,start,stop)); std::cout<<(j?",":"")<<ms;
    }
    check(cudaMemcpy(states.data(),dstate,states.size()*sizeof(double),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(metrics.data(),dmetrics,metrics.size()*sizeof(double),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(status.data(),dstatus,status.size()*sizeof(int),cudaMemcpyDeviceToHost));
    int failed=0; for(int value:status) failed+=value!=0;
    std::cout<<"],\"failed_trajectories\":"<<failed<<"}"<<std::endl;
    std::ofstream output(argv[3],std::ios::binary);
    output.write(reinterpret_cast<char*>(states.data()),states.size()*sizeof(double));
    output.write(reinterpret_cast<char*>(metrics.data()),metrics.size()*sizeof(double));
    output.write(reinterpret_cast<char*>(status.data()),status.size()*sizeof(int));
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaFree(daxis)); check(cudaFree(dlut)); check(cudaFree(dstate)); check(cudaFree(dmetrics)); check(cudaFree(dstatus));
}
