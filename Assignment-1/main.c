#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);

	int rank, P;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &P);

	if (argc != 6) {
		if (rank == 0)
			printf("Usage: %s M D1 D2 T seed\n", argv[0]);
		MPI_Finalize();
		return 0;
	}

	int M = atoi(argv[1]);
	int D1 = atoi(argv[2]);
	int D2 = atoi(argv[3]);
	int T = atoi(argv[4]);
	int seed = atoi(argv[5]);

	int hasD1 = (rank + D1 <= P - 1);
	int hasD2 = (rank + D2 <= P - 1);
	int preD1 = (rank - D1 >= 0);
	int preD2 = (rank - D2 >= 0);

	double *sendD1 = hasD1 ? malloc(M * sizeof(double)) : NULL;
	double *sendD2 = hasD2 ? malloc(M * sizeof(double)) : NULL;

	double *recvD1 = preD1 ? malloc(M * sizeof(double)) : NULL;
	double *recvD2 = preD2 ? malloc(M * sizeof(double)) : NULL;

	srand(seed);
	for (int i = 0; i < M; i++) {
		double v = (double)rand() * (rank + 1) / 10000.0;
		if (hasD1) sendD1[i] = v;
		if (hasD2) sendD2[i] = v;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double time_taken = MPI_Wtime();

	for (int iter = 0; iter < T; iter++)
	{
		if (hasD1)
			MPI_Send(sendD1, M, MPI_DOUBLE, rank+D1, 1, MPI_COMM_WORLD);
		if (hasD2)
			MPI_Send(sendD2, M, MPI_DOUBLE, rank+D2, 2, MPI_COMM_WORLD);

		if (preD1)
			MPI_Recv(recvD1, M, MPI_DOUBLE, rank-D1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (preD2)
			MPI_Recv(recvD2, M, MPI_DOUBLE, rank-D2, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		if (preD1)
			for (int i = 0; i < M; i++)
				recvD1[i] = recvD1[i] * recvD1[i];

		if (preD2)
			for (int i=0;i<M;i++)
				recvD2[i] = log(recvD2[i]);

		if (preD1)
			MPI_Send(recvD1, M, MPI_DOUBLE, rank-D1, 3, MPI_COMM_WORLD);
		if (preD2)
			MPI_Send(recvD2, M, MPI_DOUBLE, rank-D2, 4, MPI_COMM_WORLD);

		if (hasD1)
			MPI_Recv(sendD1, M, MPI_DOUBLE, rank+D1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (hasD2)
			MPI_Recv(sendD2, M, MPI_DOUBLE, rank+D2, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for (int i=0;i<M;i++) {
			double combined = 0.0;
			if (hasD1 && hasD2) combined = sendD1[i] + sendD2[i];
			else if (hasD1) combined = sendD1[i];

			if (hasD1)
				sendD1[i] = (unsigned long long)combined % (unsigned long long)100000;
			if (hasD2)
				sendD2[i] = combined * 100000.0;
		}
	}

	double pair[2] = {-INFINITY, -INFINITY};

	if (hasD1)
		for (int i=0;i<M;i++)
			if (sendD1[i] > pair[0]) pair[0] = sendD1[i];

	if (hasD2)
		for (int i=0;i<M;i++)
			if (sendD2[i] > pair[1]) pair[1] = sendD2[i];

	if (rank != 0) {
		MPI_Send(pair, 2, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
	}

	double globalMaxD1 = pair[0];
	double globalMaxD2 = pair[1];

	if (rank == 0) {
		for (int r=1; r<P; r++) {
			double pair[2];
			MPI_Recv(pair, 2, MPI_DOUBLE, r, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if (pair[0] > globalMaxD1) globalMaxD1 = pair[0];
			if (pair[1] > globalMaxD2) globalMaxD2 = pair[1];
		}
		time_taken = MPI_Wtime() - time_taken;
	}

	if (rank == 0)
		printf("%lf %lf %lf\n", globalMaxD1, globalMaxD2, time_taken);

	free(sendD1);
	free(sendD2);
	free(recvD1);
	free(recvD2);

	MPI_Finalize();
	return 0;
}
