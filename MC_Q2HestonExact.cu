/**************************************************************
Lokman A. Abbas-Turki code

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <math.h>

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))


/*One-Dimensional Normal Law. Cumulative distribution function. */
double NP(double x) {
	const double p = 0.2316419;
	const double b1 = 0.319381530;
	const double b2 = -0.356563782;
	const double b3 = 1.781477937;
	const double b4 = -1.821255978;
	const double b5 = 1.330274429;
	const double one_over_twopi = 0.39894228;
	double t;

	if (x >= 0.0) {
		t = 1.0 / (1.0 + p * x);
		return (1.0 - one_over_twopi * exp(-x * x / 2.0) * t * (t * (t *
			(t * (t * b5 + b4) + b3) + b2) + b1));
	}
	else {/* x < 0 */
		t = 1.0 / (1.0 - p * x);
		return (one_over_twopi * exp(-x * x / 2.0) * t * (t * (t * (t *
			(t * b5 + b4) + b3) + b2) + b1));
	}
}

// Set the state for each thread
__global__ void init_curand_state_k(curandState* state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}



// Marsaglia-Tsang method for gamma distribution
__device__ float gamma_mt(float alpha, curandState* state) {
	if (alpha < 1.0f) {

		float u = curand_uniform(state);

		return gamma_mt(alpha + 1.0f, state) * powf(u, 1.0f / alpha);

	}
	else {

		float d = alpha - 1.0f / 3.0f;
		float c = 1.0f / sqrtf(9.0f * d);

		while (true) {

			float x = curand_normal(state);
			float v = 1.0f + c * x;

			if (v <= 0) continue;

			v = v * v * v;

			float u = curand_uniform(state);

			if (u < 1.0f - 0.0331f * (x * x) * (x * x)) return d * v;

			if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v))) return d * v;

		}
	}
}


// Monte Carlo simulation kernel
__global__ void Heston_Exact_MC_k(float S_0, float v_0, float r, float kappa, float theta, float sigma, float rho, float dt, int N, curandState* state, float* PayGPU) {

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx];

	float S = S_0;
	float v = v_0;

	float v_next;
	float vI = 0.0f;
	float exp_kappa_dt = expf(-kappa * dt);
	float d = 2.0f * kappa * theta / (sigma * sigma);
	float coeff = sigma * sigma * (1.0f - exp_kappa_dt) / (2.0f * kappa);

	for (int i = 0; i < N; i++) {

		float lambda = (2.0f * kappa * exp_kappa_dt * v) / (sigma * sigma * (1.0f - exp_kappa_dt));

		float Npoisson = curand_poisson(&localState, lambda);

		float alpha = d + Npoisson;

		v_next = coeff * gamma_mt(alpha, &localState);

		vI += 0.5f * (v + v_next) * dt;

		v = v_next;

	}

	float IW = (v - v_0 - kappa * theta + kappa * vI) / sigma;

	float m = -0.5f * vI + rho * IW;
	float Sigma2 = (1.0f - rho * rho) * vI;
	float G = curand_normal(&localState);
	float S1 = expf(m + sqrtf(fmaxf(Sigma2, 0.0f)) * G);

	PayGPU[idx] = fmaxf(S1 - 1.0f, 0.0f);

	state[idx] = localState;
}

int main(void) {

	int NTPB = 512;
	int NB = 512;
	int n = NB * NTPB;
	float S_0 = 1.0f;
	float v_0 = 0.1f;
	float r = 0.0f;
	float kappa = 0.5f;
	float theta = 0.1f;
	float sigma = 0.3f;
	float rho = 0.5f; // not given in project description, I choose it to be 0.5
	float T = 1.0f;
	int N = 1000;
	float dt = T / N;
	float sum = 0.0f;
	float sum2 = 0.0f;
	float* PayCPU, * PayGPU;

	PayCPU = (float*)malloc(n * sizeof(float));
	cudaMalloc(&PayGPU, n * sizeof(float));

	curandState* states;
	cudaMalloc(&states, n * sizeof(curandState));

	init_curand_state_k << <NB, NTPB >> > (states);

	float Tim;
	cudaEvent_t start, stop;			// GPU timer instructions
	cudaEventCreate(&start);			// GPU timer instructions
	cudaEventCreate(&stop);				// GPU timer instructions
	cudaEventRecord(start, 0);			// GPU timer instructions

	Heston_Exact_MC_k << <NB, NTPB >> > (S_0, v_0, r, kappa, theta, sigma, rho, dt, N, states, PayGPU);

	cudaEventRecord(stop, 0);			// GPU timer instructions
	cudaEventSynchronize(stop);			// GPU timer instructions
	cudaEventElapsedTime(&Tim,			// GPU timer instructions
		start, stop);					// GPU timer instructions
	cudaEventDestroy(start);			// GPU timer instructions
	cudaEventDestroy(stop);				// GPU timer instructions

	testCUDA(cudaMemcpy(PayCPU, PayGPU, n * sizeof(float), cudaMemcpyDeviceToHost));

	// Reduction performed on the host
	for (int i = 0; i < n; i++) {
		sum += PayCPU[i] / n;
		sum2 += PayCPU[i] * PayCPU[i] / n;
	};
	for (int i = 0; i < 5; i++) {
		printf("PayCPU[%d] = %f\n", i, PayCPU[i]);
	};
	printf("The estimated price of E[(S1-1)+] is equal to %f\n", sum);
	printf("error associated to a confidence interval of 95%% = %f\n",
		1.96 * sqrt((double)(1.0f / (n - 1)) * (n * sum2 - (sum * sum))) / sqrt((double)n));
	printf("Execution time %f ms\n", Tim);

	free(PayCPU);
	cudaFree(PayGPU);
	cudaFree(states);

	return 0;
}