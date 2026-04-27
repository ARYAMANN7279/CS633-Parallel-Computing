#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 3D->1D array index (layout [x][y][z])
#define IDX(a,b,c) ((a)*NY*NZ + (b)*NZ + (c))
// 1D->3D array index (layout [x][y][z])
#define IX(a,y,z) ((a)/((y)*(z)))
#define IY(a,y,z) (((a)/(z))%(y))
#define IZ(a,z) ((a)%(z))

int d, r, rank, P, // Neighbours, Radius, Rank, total Processes
    px, py, pz,    // 3D processes(px*py*pz = P)
    nx, ny, nz,    // 3D data
    NX, NY, NZ,    // N<> = n<> + 2*r (extra for data transfer)
    T, seed, F;    // Time steps, data generator Seed, num of Fields

double isovalue;
double *data, *new_data;

// Isovalue counts per field
long long *local_count, *count;

// MPI datatypes for non-contiguous slices in Y and Z directions
MPI_Datatype faceY, faceZ;
// face X already contiguous using the current data placement

// Flags indicating if neighbors exist in that direction
// 1 denotes positive, 2 denotes negative
int hasX1, hasX2,
    hasY1, hasY2,
    hasZ1, hasZ2;

// Pointers to start index data for data send/recv
double *recvX1, *recvX2,
       *recvY1, *recvY2,
       *recvZ1, *recvZ2,
       *sendX1, *sendX2,
       *sendY1, *sendY2,
       *sendZ1, *sendZ2;

// <=12 non-blocking send/recv operations (6 faces * 2)
MPI_Request reqs[12];
int req_num;

void debugCount() {
	for(int i = 0; i < T; ++i) {
		for(int j = 0; j < F; ++j)
			printf("%lld ", count[j + i * F]);
		printf("\n");
	}
}

void debugData() {
	printf("rank: %d\n", rank);
	for(int i = 0; i < F; ++i)
		for(int x = 0; x < NX; ++x) {
			for(int y = 0; y < NY; ++y) {
				for(int z = 0; z < NZ; ++z)
					printf("%lf ", data[i + F * IDX(x,y,z)]);
				printf("\n");
			}
			printf("\n");
		}
	printf("\n");
}

int initArgs(int argc, char** argv) {
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &P);

	if(argc != 13) {
		if(!rank)
			printf("Usage: %s d ppn px py pz nx ny nz T seed F isovalue\n", argv[0]);
		MPI_Finalize();
		return 1;
	}
	d = atoi(argv[1]);
	px = atoi(argv[3]);
	py = atoi(argv[4]);
	pz = atoi(argv[5]);
	nx = atoi(argv[6]);
	ny = atoi(argv[7]);
	nz = atoi(argv[8]);
	T = atoi(argv[9]);
	seed = atoi(argv[10]);
	F = atoi(argv[11]);
	isovalue = atof(argv[12]);

	// d = 6n+1, else not clearly defined neighbors for avg
	if((d-1) % 6) {
		if(!rank)
			printf("d needs to be 6n+1\n");
		MPI_Finalize();
		return 1;
	}
	// P = px*py*pz
	if(P != px * py * pz) {
		if(!rank)
			printf("total processes not matching\n");
		MPI_Finalize();
		return 1;
	}

	r = (d - 1) / 6;
	// Allocate data with extra r on both sides
	NX = nx + 2*r;
	NY = ny + 2*r;
	NZ = nz + 2*r;
	// All data in contigous memory region, to improve cache hits
	data = malloc(F*NX*NY*NZ*sizeof(double));
	// Extra for average computation
	new_data = malloc(F*NX*NY*NZ*sizeof(double));

	local_count = malloc(F * sizeof(long long));
	// Only root rank collects data
	if(!rank) count = malloc(F * T * sizeof(long long));
	else count = NULL;

	// Data generation as provided in question
	srand(seed);

	for(int i = 0; i < F; i++)
		for(int j = 0; j < nx*ny*nz; j++) {
			// Offseting index for extra r offset in data
			int jx = IX(j, ny, nz) + r;
			int jy = IY(j, ny, nz) + r;
			int jz = IZ(j, nz) + r;
			// Memory layout: [x][y][z][field]
			data[i + IDX(jx,jy,jz) * F] = (double)rand() * (rank + 1) / (110426.0 + i + j);
		}

	// NX slabs of NX*r size, for all fields
	MPI_Type_vector(NX, F*NZ*r, F*NZ*NY, MPI_DOUBLE, &faceY);
	MPI_Type_commit(&faceY);

	// NY*NX rows of length r, for all fields
	MPI_Type_vector(NY*NX, F*r, F*NZ, MPI_DOUBLE, &faceZ);
	MPI_Type_commit(&faceZ);

	// Boundary procceses doesn't have corresponding neighbour
	hasX1 = (IX(rank, py, pz) + 1 < px);
	hasX2 = (IX(rank, py, pz) - 1 >= 0);
	hasY1 = (IY(rank, py, pz) + 1 < py);
	hasY2 = (IY(rank, py, pz) - 1 >= 0);
	hasZ1 = (IZ(rank, pz) + 1 < pz);
	hasZ2 = (IZ(rank, pz) - 1 >= 0);

	return 0;
}

void cleanup() {
	MPI_Type_free(&faceY);
	MPI_Type_free(&faceZ);

	free(data);
	free(new_data);
	MPI_Finalize();
}

void transferData() {
	req_num = 0;

	// Reinitializing pointers each time as pointer to data changes after averaging
	// Recieving pointer at outer layer
	recvX1 = data + F * IDX(0,0,0);
	recvX2 = data + F * IDX(r+nx,0,0);
	recvY1 = data + F * IDX(0,0,0);
	recvY2 = data + F * IDX(0,r+ny,0);
	recvZ1 = data + F * IDX(0,0,0);
	recvZ2 = data + F * IDX(0,0,r+nz);

	// Sending pointer at inner layer
	sendX1 = data + F * IDX(nx,0,0);
	sendX2 = data + F * IDX(r,0,0);
	sendY1 = data + F * IDX(0,ny,0);
	sendY2 = data + F * IDX(0,r,0);
	sendZ1 = data + F * IDX(0,0,nz);
	sendZ2 = data + F * IDX(0,0,r);

	if(hasX2)MPI_Irecv(recvX1, F*NZ*NY*r, MPI_DOUBLE, rank - py*pz, 0, MPI_COMM_WORLD, &reqs[req_num++]);
	if(hasX1)MPI_Irecv(recvX2, F*NZ*NY*r, MPI_DOUBLE, rank + py*pz, 1, MPI_COMM_WORLD, &reqs[req_num++]);
	if(hasY2)MPI_Irecv(recvY1, 1, faceY, rank - pz, 2, MPI_COMM_WORLD, &reqs[req_num++]);
	if(hasY1)MPI_Irecv(recvY2, 1, faceY, rank + pz, 3, MPI_COMM_WORLD, &reqs[req_num++]);
	if(hasZ2)MPI_Irecv(recvZ1, 1, faceZ, rank - 1, 4, MPI_COMM_WORLD, &reqs[req_num++]);
	if(hasZ1)MPI_Irecv(recvZ2, 1, faceZ, rank + 1, 5, MPI_COMM_WORLD, &reqs[req_num++]);

	if(hasX2)MPI_Isend(sendX2, F*NZ*NY*r, MPI_DOUBLE, rank - py*pz, 1, MPI_COMM_WORLD, &reqs[req_num++]);
	if(hasX1)MPI_Isend(sendX1, F*NZ*NY*r, MPI_DOUBLE, rank + py*pz, 0, MPI_COMM_WORLD, &reqs[req_num++]);
	if(hasY2)MPI_Isend(sendY2, 1, faceY, rank - pz, 3, MPI_COMM_WORLD, &reqs[req_num++]);
	if(hasY1)MPI_Isend(sendY1, 1, faceY, rank + pz, 2, MPI_COMM_WORLD, &reqs[req_num++]);
	if(hasZ2)MPI_Isend(sendZ2, 1, faceZ, rank - 1, 5, MPI_COMM_WORLD, &reqs[req_num++]);
	if(hasZ1)MPI_Isend(sendZ1, 1, faceZ, rank + 1, 4, MPI_COMM_WORLD, &reqs[req_num++]);

	// Waitall done later to optimize averaging and counting inner isovalues that do not require data transfer
}

// Averaging only inner ones that have no dependency on data transfer
void avgInnerData() {
	for(int i = 0; i < F; ++i)
		for(int x = r; x < nx - r; x++)
			for(int y = r; y < ny - r; y++)
				for(int z = r; z < nz - r; z++) {
					// Center point
					double tot = data[i+F*IDX(x+r,y+r,z+r)];

					// Add neighbors up to distance r in each direction
					for(int deep = 1; deep <= r; deep++) {
						// X-direction
						tot += data[i+F*IDX(x+r+deep,y+r,z+r)];
						tot += data[i+F*IDX(x+r-deep,y+r,z+r)];
						// Y-direction
						tot += data[i+F*IDX(x+r,y+r+deep,z+r)];
						tot += data[i+F*IDX(x+r,y+r-deep,z+r)];
						// Z-direction
						tot += data[i+F*IDX(x+r,y+r,z+r+deep)];
						tot += data[i+F*IDX(x+r,y+r,z+r-deep)];
					}
					new_data[i+F*IDX(x+r,y+r,z+r)] = tot / d;
				}
}

// Averaging only outer ones that have dependency on data transfer
void avgOuterData() {
	// Loop over all fields
	for(int i = 0; i < F; ++i)
		// Iterate over entire local domain
		for(int x = 0; x < nx; x++)
			for(int y = 0; y < ny; y++)
				for(int z = 0; z < nz; z++) {
					// Skip cells that were already handled by avgInnerData
					int inside = 1;
					if(x < r) inside = 0;
					if(y < r) inside = 0;
					if(z < r) inside = 0;
					if(x >= nx-r) inside = 0;
					if(y >= ny-r) inside = 0;
					if(z >= nz-r) inside = 0;
					if(inside) {
						if(z == r) z = nz - r;
						else continue;
					}
					// Initialize with center value
					int count = 1;
					double tot = data[i+F*IDX(x+r,y+r,z+r)];

					// Apply stencil up to radius r in all 6 directions
					/* +X direction:
					   If (x < (nx-deep)): neighbor lies within this process's local subdomain
					   Else if (hasX1): neighbor lies in adjacent process (+X direction),
					   and has already been received into halo via MPI
					   If neither is true: neighbor does not exist then skip
					   Similar Checks for every other 5 directions
					*/
					for(int deep = 1; deep <= r; deep++) {
						// +X direction :
						if(x < (nx-deep) || hasX1) {
							tot += data[i+F*IDX(x+r+deep,y+r,z+r)];
							count++;
						}
						// +Y direction :
						if(y < (ny-deep) || hasY1) {
							tot += data[i+F*IDX(x+r,y+r+deep,z+r)];
							count++;
						}
						// +Z direction :
						if(z < (nz-deep) || hasZ1) {
							tot += data[i+F*IDX(x+r,y+r,z+r+deep)];
							count++;
						}

						// -X direction :
						if(x >= deep || hasX2) {
							tot += data[i+F*IDX(x+r-deep,y+r,z+r)];
							count++;
						}
						// -Y direction :
						if(y >= deep || hasY2) {
							tot += data[i+F*IDX(x+r,y+r-deep,z+r)];
							count++;
						}
						// -Z direction :
						if(z >= deep || hasZ2) {
							tot += data[i+F*IDX(x+r,y+r,z+r-deep)];
							count++;
						}
					}
					// store average value over computed stencil size(count)
					new_data[i+F*IDX(x+r,y+r,z+r)] = tot / count;
				}

	// Swap pointers to avoid memcpy
	double *tmp = data;
	data = new_data;
	new_data = tmp;
}

// Counting only inner ones that have no dependency on data transfer
void countInnerIsoValue(int idx) {
	memset(local_count, 0, F * sizeof(long long));
	for(int i = 0; i < F; i++) {
		for(int x = 0; x < nx - 1; x++) {
			for(int y = 0; y < ny - 1; y++) {
				for(int z = 0; z < nz - 1; z++) {

					double v = data[i + IDX(x+r,y+r,z+r) * F];

					double comp[3];
					// adding offset of r to every coordinate to skip past the halo-margins in the array

					comp[0] = data[i + IDX(x+r+1,y+r,z+r) * F];
					comp[1] = data[i + IDX(x+r,y+r+1,z+r) * F];
					comp[2] = data[i + IDX(x+r,y+r,z+r+1) * F];

					for(int c = 0; c < 3; ++c)
						if((v < isovalue && comp[c] > isovalue) ||
								(v > isovalue && comp[c] < isovalue))
							local_count[i]++;
				}
			}
		}
	}
}

// Counting only outer ones that have dependency on data transfer
void countOuterIsoValue(int idx) {
	for(int i = 0; i < F; i++) {
		for(int x = 0; x < nx; x++) {
			for(int y = 0; y < ny; y++) {
				for(int z = 0; z < nz; z++) {

					// Skip cells fully in the interior, already counted
					int inside = 1;
					if(x == nx-1) inside = 0;
					if(y == ny-1) inside = 0;
					if(z == nz-1) inside = 0;
					if(inside) {
						if(!z) z = nz - 1;
						else continue;
					}

					double v = data[i + IDX(x+r,y+r,z+r) * F];

					double comp[3];
					int compCount = 0;

					if(x != (nx-1) || hasX1)comp[compCount++] = data[i + IDX(x+r+1,y+r,z+r) * F];
					if(y != (ny-1) || hasY1)comp[compCount++] = data[i + IDX(x+r,y+r+1,z+r) * F];
					if(z != (nz-1) || hasZ1)comp[compCount++] = data[i + IDX(x+r,y+r,z+r+1) * F];

					for(int c = 0; c < compCount; ++c)
						if((v < isovalue && comp[c] > isovalue) ||
								(v > isovalue && comp[c] < isovalue))
							local_count[i]++;
				}
			}
		}
	}

	// Gather totals from all ranks to root rank
	MPI_Reduce(local_count,rank == 0 ? count + idx * F : NULL, F, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
	// Initialize MPI, parse arguments, allocate memory, setup grid
	if(initArgs(argc, argv))
		return 1;

	//starting the Timing calc
	double sTime = MPI_Wtime();

	// --- Initial phase ---

	// Start halo exchange for timestep 0
	transferData();
	// Calculating average asynchrounosly with no dependency to optimize averaging time
	avgInnerData();

	// Wait for halo data to arrive before computing boundary values
	MPI_Waitall(req_num, reqs, MPI_STATUSES_IGNORE);

	for(int i = 0; i < T; ++i) {
		avgOuterData();

		// Start communication for next iteration
		transferData();

		// Not required if last iteration
		if(i != T-1)
			avgInnerData();

		// Calculating average asynchrounosly with no dependency to optimize averaging time
		countInnerIsoValue(i);
		// Calculating count asynchrounosly with no dependency to optimize counting time
		MPI_Waitall(req_num, reqs, MPI_STATUSES_IGNORE);

		countOuterIsoValue(i);
	}
	//ending the timing calc
	double eTime = MPI_Wtime();
	if(!rank) debugCount();
	if(!rank) printf("%lf\n", eTime - sTime);

	cleanup();
	return 0;
}
