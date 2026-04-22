/*
 * GPU Programming - Project 1 : Monte Carlo simulation of the Heston model
 *
 * Authors : Thibault Charbonnier & Arsen Pidburachynskyi
 * Initial code : Lokman A. Abbas-Turki
 *
 * The project is separated into three questions :
 *
 *   Q1 — Euler discretisation
 *
 *   Q2 — Exact scheme (according to reference [4] : Broadie & Kaya 2006 and [8] : Marsaglia & Wai 2000)
 *
 *   Q3 — Almost exact scheme (according to reference [10] : Van Haastrecht & Pelsser 2010)
 *
 * NB : The Q3 outputs are written in results.csv that can be visualized with the provided python file.
 */

#include <stdio.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Helper and utility functions
// ---------------------------------------------------------------------------
void testCUDA(cudaError_t error, const char *file, int line)
{
	if (error != cudaSuccess)
	{
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}
#define testCUDA(error) (testCUDA(error, __FILE__, __LINE__))

double NP(double x)
{
	const double p = 0.2316419;
	const double b1 = 0.319381530;
	const double b2 = -0.356563782;
	const double b3 = 1.781477937;
	const double b4 = -1.821255978;
	const double b5 = 1.330274429;
	const double one_over_twopi = 0.39894228;
	double t;

	if (x >= 0.0)
	{
		t = 1.0 / (1.0 + p * x);
		return (1.0 - one_over_twopi * exp(-x * x / 2.0) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
	}
	else
	{ /* x < 0 */
		t = 1.0 / (1.0 - p * x);
		return (one_over_twopi * exp(-x * x / 2.0) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
	}
}

__global__ void init_curand_state_k(curandState *state)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}

// Gamma generator [8] (Marsaglia & Wai 2000).
__device__ float gamma_mt(float alpha, curandState *state)
{
	if (alpha < 1.0f)
	{
		float u = curand_uniform(state);
		return gamma_mt(alpha + 1.0f, state) * powf(u, 1.0f / alpha);
	}
	float d = alpha - 1.0f / 3.0f;
	float c = 1.0f / sqrtf(9.0f * d);
	while (true)
	{
		float x = curand_normal(state);
		float v = 1.0f + c * x;
		if (v <= 0.0f)
			continue;
		v = v * v * v;
		float u = curand_uniform(state);
		if (u < 1.0f - 0.0331f * x * x * x * x)
			return d * v;
		if (logf(u) < 0.5f * x * x + d * (1.0f - v + logf(v)))
			return d * v;
	}
}

// ---------------------------------------------------------------------------
// Q1 — Euler discretisation kernel
// ---------------------------------------------------------------------------
__global__ void Heston_Euler_MC_k(float S_0, float v_0, float r,
								  float kappa, float theta, float sigma, float rho,
								  float dt, int N,
								  curandState *state, float *sum, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx];

	extern __shared__ float A[];
	float *R1s = A;
	float *R2s = R1s + blockDim.x;

	float S = S_0, v = v_0;

	for (int i = 0; i < N; i++)
	{
		float G1 = curand_normal(&localState);
		float G2 = curand_normal(&localState);
		float Z = rho * G1 + sqrtf(1.0f - rho * rho) * G2;
		S = S + r * S * dt + sqrtf(v) * S * sqrtf(dt) * Z;
		v = v + kappa * (theta - v) * dt + sigma * sqrtf(v) * sqrtf(dt) * G1;
		v = fmaxf(v, 0.0f);
	}

	float payoff = expf(-r * dt * N) * fmaxf(S - 1.0f, 0.0f);
	R1s[threadIdx.x] = payoff / n;
	R2s[threadIdx.x] = payoff * payoff / n;

	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			R1s[threadIdx.x] += R1s[threadIdx.x + s];
			R2s[threadIdx.x] += R2s[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)
	{
		atomicAdd(sum, R1s[0]);
		atomicAdd(sum + 1, R2s[0]);
	}

	state[idx] = localState;
}

// ---------------------------------------------------------------------------
// Q2 — Exact scheme (reference [4] : Broadie & Kaya 2006)
// ---------------------------------------------------------------------------
__global__ void Heston_Exact_MC_k(float S_0, float v_0, float r,
								  float kappa, float theta, float sigma, float rho,
								  float dt, int N,
								  curandState *state, float *sum, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx];

	extern __shared__ float A[];
	float *R1s = A;
	float *R2s = R1s + blockDim.x;

	float v = v_0;
	float vI = 0.0f;

	float exp_kd = expf(-kappa * dt);
	float d = 2.0f * kappa * theta / (sigma * sigma);
	float coeff = sigma * sigma * (1.0f - exp_kd) / (2.0f * kappa);

	for (int i = 0; i < N; i++)
	{
		float lambda = (2.0f * kappa * exp_kd * v) / (sigma * sigma * (1.0f - exp_kd));
		float poisson = (float)curand_poisson(&localState, (double)lambda);
		float v_next = coeff * gamma_mt(d + poisson, &localState);
		vI += 0.5f * (v + v_next) * dt;
		v = v_next;
	}

	float T = dt * N;
	float IW = (v - v_0 - kappa * theta * T + kappa * vI) / sigma;
	float m = r * T - 0.5f * vI + rho * IW;
	float Sigma2 = (1.0f - rho * rho) * vI;
	float G = curand_normal(&localState);
	float S1 = S_0 * expf(m + sqrtf(fmaxf(Sigma2, 0.0f)) * G);

	float payoff = expf(-r * T) * fmaxf(S1 - 1.0f, 0.0f);
	R1s[threadIdx.x] = payoff / n;
	R2s[threadIdx.x] = payoff * payoff / n;

	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			R1s[threadIdx.x] += R1s[threadIdx.x + s];
			R2s[threadIdx.x] += R2s[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)
	{
		atomicAdd(sum, R1s[0]);
		atomicAdd(sum + 1, R2s[0]);
	}

	state[idx] = localState;
}

// ---------------------------------------------------------------------------
// Q3 — Almost exact scheme (reference [10] : Van Haastrecht & Pelsser 2010)
// ---------------------------------------------------------------------------
__global__ void Heston_AlmostExact_MC_k(float S_0, float v_0, float r,
										float kappa, float theta, float sigma, float rho,
										float dt, int N,
										curandState *state, float *sum, int n)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx];

	extern __shared__ float A[];
	float *R1s = A;
	float *R2s = R1s + blockDim.x;

	float log_S = logf(S_0);
	float v = v_0;

	float exp_kd = expf(-kappa * dt);
	float d = 2.0f * kappa * theta / (sigma * sigma);
	float coeff = sigma * sigma * (1.0f - exp_kd) / (2.0f * kappa);

	float k0 = (r - rho / sigma * kappa * theta) * dt;
	float k1 = (rho * kappa / sigma - 0.5f) * dt - rho / sigma;
	float k2 = rho / sigma;
	float var_coeff = (1.0f - rho * rho) * dt;

	for (int i = 0; i < N; i++)
	{
		float lambda = (2.0f * kappa * exp_kd * v) / (sigma * sigma * (1.0f - exp_kd));
		float poisson = (float)curand_poisson(&localState, (double)lambda);
		float v_next = coeff * gamma_mt(d + poisson, &localState);

		float G = curand_normal(&localState);
		log_S += k0 + k1 * v + k2 * v_next + sqrtf(var_coeff * v) * G;
		v = v_next;
	}

	float T = dt * N;
	float payoff = expf(-r * T) * fmaxf(expf(log_S) - 1.0f, 0.0f);
	R1s[threadIdx.x] = payoff / n;
	R2s[threadIdx.x] = payoff * payoff / n;

	__syncthreads();
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			R1s[threadIdx.x] += R1s[threadIdx.x + s];
			R2s[threadIdx.x] += R2s[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0)
	{
		atomicAdd(sum, R1s[0]);
		atomicAdd(sum + 1, R2s[0]);
	}

	state[idx] = localState;
}

// ===========================================================================
// Run & time a MC kernel function
// ===========================================================================

typedef enum
{
	EULER = 0,
	EXACT = 1,
	ALMOST_EXACT = 2
} KernelChoice;

static void run_and_time(KernelChoice kc,
						 float S_0, float v_0, float r,
						 float kappa, float theta, float sigma, float rho,
						 float dt, int N,
						 int NB, int NTPB, int n,
						 curandState *states, float *SumGPU,
						 float *out_price, float *out_ci, float *out_ms)
{
	testCUDA(cudaMemset(SumGPU, 0, 2 * sizeof(float)));

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	int smem = 2 * NTPB * (int)sizeof(float);
	switch (kc)
	{
	case EULER:
		Heston_Euler_MC_k<<<NB, NTPB, smem>>>(
			S_0, v_0, r, kappa, theta, sigma, rho, dt, N, states, SumGPU, n);
		break;
	case EXACT:
		Heston_Exact_MC_k<<<NB, NTPB, smem>>>(
			S_0, v_0, r, kappa, theta, sigma, rho, dt, N, states, SumGPU, n);
		break;
	case ALMOST_EXACT:
		Heston_AlmostExact_MC_k<<<NB, NTPB, smem>>>(
			S_0, v_0, r, kappa, theta, sigma, rho, dt, N, states, SumGPU, n);
		break;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(out_ms, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	float s[2];
	testCUDA(cudaMemcpy(s, SumGPU, 2 * sizeof(float), cudaMemcpyDeviceToHost));
	*out_price = s[0];
	*out_ci = 1.96f * sqrtf((s[1] - s[0] * s[0]) / (n - 1));
}

// ===========================================================================
// main
// ===========================================================================
int main(void)
{

	int NTPB = 512, NB = 512, n = NB * NTPB;
	float S_0 = 1.0f, v_0 = 0.1f, r = 0.0f, rho = 0.5f, T = 1.0f;

	float *SumGPU;
	testCUDA(cudaMalloc(&SumGPU, 2 * sizeof(float)));

	curandState *states;
	testCUDA(cudaMalloc(&states, n * sizeof(curandState)));
	init_curand_state_k<<<NB, NTPB>>>(states);
	testCUDA(cudaDeviceSynchronize());

	// -----------------------------------------------------------------------
	// Q1 / Q2 - fixed params
	// -----------------------------------------------------------------------
	{
		float kappa = 0.5f, theta = 0.1f, sigma = 0.3f;
		int N = 1000;
		float dt = T / N;
		float price, ci, ms;

		printf("=== Q1 / Q2 — kappa=%.1f  theta=%.2f  sigma=%.2f  rho=%.2f"
			   "  dt=1/%d  n=%d ===\n",
			   kappa, theta, sigma, rho, N, n);

		run_and_time(EULER, S_0, v_0, r, kappa, theta, sigma, rho, dt, N,
					 NB, NTPB, n, states, SumGPU, &price, &ci, &ms);
		printf("  Euler : price = %.6f  95%%CI = %.6f  time = %.1f ms\n",
			   price, ci, ms);

		run_and_time(EXACT, S_0, v_0, r, kappa, theta, sigma, rho, dt, N,
					 NB, NTPB, n, states, SumGPU, &price, &ci, &ms);
		printf("  Exact : price = %.6f  95%%CI = %.6f  time = %.1f ms\n\n",
			   price, ci, ms);
	}

	// -----------------------------------------------------------------------
	// Q3 — params grid + CSV output
	// -----------------------------------------------------------------------
	float kappas[] = {0.5f, 1.0f, 2.0f, 5.0f, 10.0f};
	float thetas[] = {0.05f, 0.1f, 0.2f, 0.5f};
	float sigmas[] = {0.1f, 0.3f, 0.5f, 0.7f, 1.0f};
	int Nk = 5, Nt = 4, Ns = 5;

	int N_fine = 1000, N_coarse = 30;
	float dt_fine = T / N_fine, dt_coarse = T / N_coarse;

	FILE *f = fopen("results.csv", "w");
	if (!f)
	{
		printf("Cannot open results.csv\n");
		return 1;
	}

	fprintf(f,
			"kappa,theta,sigma,"
			"price_euler1000,ci_euler1000,time_euler1000,"
			"price_euler30,ci_euler30,time_euler30,"
			"price_exact1000,ci_exact1000,time_exact1000,"
			"price_exact30,ci_exact30,time_exact30,"
			"price_ae1000,ci_ae1000,time_ae1000,"
			"price_ae30,ci_ae30,time_ae30\n");

	printf("=== Q3 — parameter grid  (condition: 20*kappa*theta > sigma^2) ===\n");
	printf("%-6s %-6s %-5s | "
		   "%-24s | %-24s | %-24s | %-24s | %-24s | %-24s\n",
		   "kappa", "theta", "sigma",
		   "Euler dt=1/1000",
		   "Euler dt=1/30",
		   "Exact dt=1/1000",
		   "Exact dt=1/30",
		   "AE dt=1/1000",
		   "AE dt=1/30");

	printf("%-6s %-6s %-5s | "
		   "%-8s %-8s %-6s | "
		   "%-8s %-8s %-6s | "
		   "%-8s %-8s %-6s | "
		   "%-8s %-8s %-6s | "
		   "%-8s %-8s %-6s | "
		   "%-8s %-8s %-6s\n",
		   "", "", "",
		   "price", "95%CI", "ms",
		   "price", "95%CI", "ms",
		   "price", "95%CI", "ms",
		   "price", "95%CI", "ms",
		   "price", "95%CI", "ms",
		   "price", "95%CI", "ms");

	float pe1000, ce1000, te1000;
	float pe30, ce30, te30;
	float px1000, cx1000, tx1000;
	float px30, cx30, tx30;
	float pae1000, cae1000, tae1000;
	float pae30, cae30, tae30;

	for (int ik = 0; ik < Nk; ik++)
	{
		for (int it = 0; it < Nt; it++)
		{
			for (int is = 0; is < Ns; is++)
			{
				float kappa = kappas[ik];
				float theta = thetas[it];
				float sigma = sigmas[is];

				if (20.0f * kappa * theta <= sigma * sigma)
					continue;

				run_and_time(EULER, S_0, v_0, r, kappa, theta, sigma, rho,
							 dt_fine, N_fine, NB, NTPB, n, states, SumGPU,
							 &pe1000, &ce1000, &te1000);

				run_and_time(EULER, S_0, v_0, r, kappa, theta, sigma, rho,
							 dt_coarse, N_coarse, NB, NTPB, n, states, SumGPU,
							 &pe30, &ce30, &te30);

				run_and_time(EXACT, S_0, v_0, r, kappa, theta, sigma, rho,
							 dt_fine, N_fine, NB, NTPB, n, states, SumGPU,
							 &px1000, &cx1000, &tx1000);

				run_and_time(EXACT, S_0, v_0, r, kappa, theta, sigma, rho,
							 dt_coarse, N_coarse, NB, NTPB, n, states, SumGPU,
							 &px30, &cx30, &tx30);

				run_and_time(ALMOST_EXACT, S_0, v_0, r, kappa, theta, sigma, rho,
							 dt_fine, N_fine, NB, NTPB, n, states, SumGPU,
							 &pae1000, &cae1000, &tae1000);

				run_and_time(ALMOST_EXACT, S_0, v_0, r, kappa, theta, sigma, rho,
							 dt_coarse, N_coarse, NB, NTPB, n, states, SumGPU,
							 &pae30, &cae30, &tae30);

				printf("%-6.2f %-6.3f %-5.2f | "
					   "%-8.5f %-8.5f %-6.1f | "
					   "%-8.5f %-8.5f %-6.1f | "
					   "%-8.5f %-8.5f %-6.1f | "
					   "%-8.5f %-8.5f %-6.1f | "
					   "%-8.5f %-8.5f %-6.1f | "
					   "%-8.5f %-8.5f %-6.1f\n",
					   kappa, theta, sigma,
					   pe1000, ce1000, te1000,
					   pe30, ce30, te30,
					   px1000, cx1000, tx1000,
					   px30, cx30, tx30,
					   pae1000, cae1000, tae1000,
					   pae30, cae30, tae30);

				fprintf(f,
						"%.3f,%.3f,%.3f,"
						"%.6f,%.6f,%.3f,"
						"%.6f,%.6f,%.3f,"
						"%.6f,%.6f,%.3f,"
						"%.6f,%.6f,%.3f,"
						"%.6f,%.6f,%.3f,"
						"%.6f,%.6f,%.3f\n",
						kappa, theta, sigma,
						pe1000, ce1000, te1000,
						pe30, ce30, te30,
						px1000, cx1000, tx1000,
						px30, cx30, tx30,
						pae1000, cae1000, tae1000,
						pae30, cae30, tae30);
			}
		}
	}

	fclose(f);

	cudaFree(SumGPU);
	cudaFree(states);
	return 0;
}
