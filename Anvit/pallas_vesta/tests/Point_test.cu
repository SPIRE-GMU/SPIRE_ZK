#ifndef __POINT_UNIT_TEST
#define __POINT_UNIT_TEST

#include "test_header.cuh"

#define NUM_TEST 10

void write_limbs_to_file(std::ofstream &out, uint64_t *data, int limit = 4)
{
    bool printStarted = false;
    for (int i = limit - 1; i >= 0; i--)
    {
        if (printStarted)
        {
            out << std::hex << std::uppercase << std::setfill('0') << std::setw(16) << data[i];
        }
        else
        {
            if (data[i])
            {
                out << std::hex << std::uppercase << data[i];
                printStarted = true;
            }
        }
    }
    out << " ";
}
__global__ void get_point_to_array(uint64_t *a, uint64_t *b, uint64_t *c, Point *p)
{
    p->X.decode_montgomery();
    p->Y.decode_montgomery();
    p->Z.decode_montgomery();
    for (int i = 0; i < 4; i++)
        a[i] = p->X.data[i];

    for (int i = 0; i < 4; i++)
        b[i] = p->Y.data[i];

    for (int i = 0; i < 4; i++)
        c[i] = p->Z.data[i];
    p->X.encode_montgomery();
    p->Y.encode_montgomery();
    p->Z.encode_montgomery();
}
void write_point_to_file(Point *p, const char *filename, size_t len = 1)
{
    std::ofstream out(filename);
    uint64_t x[4], y[4], z[4];
    for (int i = 0; i < len; i++)
    {
        uint64_t *a, *b, *c;
        CUDA_CHECK(cudaMalloc(&a, sizeof(uint64_t) * 4));
        CUDA_CHECK(cudaMalloc(&b, sizeof(uint64_t) * 4));
        CUDA_CHECK(cudaMalloc(&c, sizeof(uint64_t) * 4));
        get_point_to_array<<<1, 1>>>(a, b, c, p + i);
        CUDA_CHECK(cudaMemcpy(x, a, sizeof(uint64_t) * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(y, b, sizeof(uint64_t) * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(z, c, sizeof(uint64_t) * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaDeviceSynchronize());
        write_limbs_to_file(out, x);
        write_limbs_to_file(out, y);
        write_limbs_to_file(out, z);
    }

    out.close();
}

bool compare_two_files_output(const char *file1, const char *file2)
{
    std::ifstream in_cuda(file1);
    std::ifstream in_sage(file2);
    std::string cu_out;
    std::string sage_out;
    while (in_cuda >> cu_out && in_sage >> sage_out)
    {
        if (cu_out.size() != sage_out.size())
            return false;
        for (int i = 0; i < cu_out.size(); i++)
        {
            if (cu_out[i] != sage_out[i])
                return false;
        }
    }
    in_cuda.close();
    in_sage.close();
    return true;
}

__global__ void add_kernel(Point *a, Point *b, Point *res)
{
    *res = *a + *b;
    res->to_affine();
}
__global__ void multiplication_kernel(Point *a, Scalar *s, Point *res)
{
    *res = (*a) * (*s);
    res->to_affine();
}

__global__ void multiplication_kernel2(Point *a, uint64_t s, Point *res)
{
    *res = (*a) * s;
    res->to_affine();
}
__global__ void double_kernel(Point *a, Point *res)
{
    *res = a->dbl();
    res->to_affine();
}
__global__ void mixed_add_kernel(Point *a, Point *b, Point *res)
{
    b->to_affine();
    *res = a->mixed_add(*b);
    res->to_affine();
}
__global__ void check_wellformed_kernel(Point *a, bool *result)
{
    *result = a->is_well_formed();
}
__global__ void check_affine_kernel(Point *a, bool *result)
{
    *result = (a->Z.data[0] == 1);
    for (int i = 1; i < 4; i++)
    {
        *result &= (a->Z.data[i] == 0);
    }
}
__global__ void is_equal_kernel(Point *a, Point *b, bool *result)
{
    *result = (*a) == (*b);
}
__global__ void is_not_equal_kernel(Point *a, Point *b, bool *result)
{
    *result = ((*a) != (*b));
}

__global__ void generation_kernel(Point *p, size_t num_points)
{
    size_t idx = threadIdx.x + blockDim.x* blockIdx.x;
    if(idx<= num_points)
    {
        p[idx] = p[idx].one() * (idx+1);
        p[idx].to_affine();
    }
}

__global__ void init_one(Point *a)
{
    *a = a->one();
}

const char *file_sage_add = "point_output_add.txt";
const char *file_sage_multi = "point_output_multi.txt";
const char *file_sage_dbl = "point_output_dbl.txt";
const char *file_sage_generate = "point_output_generate.txt";
const char *file_cuda_add = "cuda_point_output_add.txt";
const char *file_cuda_multi = "cuda_point_output_multi.txt";
const char *file_cuda_dbl = "cuda_point_output_dbl.txt";
const char *file_cuda_generate = "cuda_point_output_generate.txt";

class PointTests : public ::testing::Test
{
public:
    PointTests()
    {
    }
    ~PointTests()
    {
    }
    void SetUp() override
    {
    }
    void TearDown() override
    {
    }
};

TEST_F(PointTests, check_Point_generation)
{
    Point *p;

    size_t num_points = rand() % 100000; // to make sure there is no out of memory error
    num_points = num_points > 4096 ? num_points : 4096;
    CUDA_CHECK(cudaMalloc(&p, sizeof(Point) * num_points));

    size_t blockSize = 128;
    size_t gridSize = (num_points + blockSize -1 )/ blockSize;
    generation_kernel<<< gridSize, blockSize>>>(p, num_points);
    std::string cmd = "sage -python point_sage.py pallas generate ";
    cmd = cmd + std::to_string(num_points) + " 1 1";

    int ret = std::system(cmd.c_str());
    if (ret != 0)
    {
        ASSERT_TRUE(false);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    write_point_to_file(p, file_cuda_generate, num_points);
    ASSERT_TRUE(compare_two_files_output(file_cuda_generate, file_sage_generate));

    CUDA_CHECK(cudaFree(p));
}

TEST_F(PointTests, check_Point_addition)
{
    Point *a, *b, *res, *one;
    cudaMalloc(&one, sizeof(Point));
    cudaMalloc(&a, sizeof(Point));
    cudaMalloc(&b, sizeof(Point));
    cudaMalloc(&res, sizeof(Point));

    for (size_t i = 0; i < NUM_TEST; i++)
    {
        size_t multiplier1 = rand() + 1;
        size_t multiplier2 = rand() + 1;

        init_one<<<1, 1>>>(one);
        cudaDeviceSynchronize();
        multiplication_kernel2<<<1, 1>>>(one, multiplier1, a);
        multiplication_kernel2<<<1, 1>>>(one, multiplier2, b);
        cudaDeviceSynchronize();
        add_kernel<<<1, 1>>>(a, b, res);
        std::string cmd = "sage -python point_sage.py pallas add 1 ";
        cmd = cmd + std::to_string(multiplier1) + " " + std::to_string(multiplier2);

        int ret = std::system(cmd.c_str());
        if (ret != 0)
        {
            ASSERT_TRUE(false);
        }
        cudaDeviceSynchronize();
        write_point_to_file(res, file_cuda_add);

        ASSERT_TRUE(compare_two_files_output(file_cuda_add, file_sage_add));
    }
}
TEST_F(PointTests, check_point_uint64_multiplication)
{
    Point *a, *b, *res, *one;
    CUDA_CHECK(cudaMalloc(&one, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&a, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&b, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&res, sizeof(Point)));

    for (size_t i = 0; i < NUM_TEST; i++)
    {
        size_t multiplier1 = rand() + 1;
        init_one<<<1, 1>>>(one);
        cudaDeviceSynchronize();
        multiplication_kernel2<<<1, 1>>>(one, multiplier1, a);

        std::string cmd = "sage -python point_sage.py pallas multi 1 1 ";
        cmd = cmd + std::to_string(multiplier1);

        int ret = std::system(cmd.c_str());
        if (ret != 0)
        {
            ASSERT_TRUE(false);
        }
        cudaDeviceSynchronize();
        write_point_to_file(a, file_cuda_multi);

        ASSERT_TRUE(compare_two_files_output(file_cuda_multi, file_sage_multi));
    }
}
__global__ void set_scalar_value(Scalar *sc, uint64_t val)
{
    *sc = Scalar(val);
}
TEST_F(PointTests, check_Point_Scalar_Multiplication)
{
    Point *a, *b, *res, *one;
    CUDA_CHECK(cudaMalloc(&one, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&a, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&b, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&res, sizeof(Point)));

    Scalar *s;
    CUDA_CHECK(cudaMalloc(&s, sizeof(Scalar)));

    init_one<<<1, 1>>>(one);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t i = 0; i < NUM_TEST; i++)
    {
        size_t multiplier1 = rand() + 1;

        set_scalar_value<<<1, 1>>>(s, multiplier1);
        CUDA_CHECK(cudaDeviceSynchronize());
        multiplication_kernel<<<1, 1>>>(one, s, a);

        std::string cmd = "sage -python point_sage.py pallas multi 1 1 ";
        cmd = cmd + std::to_string(multiplier1);

        int ret = std::system(cmd.c_str());
        if (ret != 0)
        {
            ASSERT_TRUE(false);
        }
        cudaDeviceSynchronize();
        write_point_to_file(a, file_cuda_multi);

        ASSERT_TRUE(compare_two_files_output(file_cuda_multi, file_sage_multi));
    }
}

TEST_F(PointTests, check_double)
{
    Point *a, *res, *one;
    CUDA_CHECK(cudaMalloc(&a, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&res, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&one, sizeof(Point)));

    init_one<<<1, 1>>>(one);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t i = 0; i < NUM_TEST; i++)
    {
        size_t multiplier = rand() + 1;

        multiplication_kernel2<<<1, 1>>>(one, multiplier, a);
        CUDA_CHECK(cudaDeviceSynchronize());
        double_kernel<<<1, 1>>>(a, res);
        std::string cmd = "sage -python point_sage.py pallas double 1 ";
        cmd = cmd + std::to_string(multiplier) + " 1";

        int ret = std::system(cmd.c_str());
        if (ret != 0)
        {
            ASSERT_TRUE(false);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        write_point_to_file(res, file_cuda_dbl);
        ASSERT_TRUE(compare_two_files_output(file_cuda_dbl, file_sage_dbl));
    }
}

TEST_F(PointTests, check_Mixed_add)
{
    Point *a, *b, *res, *one;
    cudaMalloc(&one, sizeof(Point));
    cudaMalloc(&a, sizeof(Point));
    cudaMalloc(&b, sizeof(Point));
    cudaMalloc(&res, sizeof(Point));

    init_one<<<1, 1>>>(one);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < NUM_TEST; i++)
    {

        size_t multiplier1 = rand() + 1;
        size_t multiplier2 = rand() + 1;

        multiplication_kernel2<<<1, 1>>>(one, multiplier1, a);
        multiplication_kernel2<<<1, 1>>>(one, multiplier2, b);
        cudaDeviceSynchronize();
        mixed_add_kernel<<<1, 1>>>(a, b, res);
        std::string cmd = "sage -python point_sage.py pallas add 1 ";
        cmd = cmd + std::to_string(multiplier1) + " " + std::to_string(multiplier2);

        int ret = std::system(cmd.c_str());
        if (ret != 0)
        {
            ASSERT_TRUE(false);
        }
        cudaDeviceSynchronize();
        write_point_to_file(res, file_cuda_add);

        ASSERT_TRUE(compare_two_files_output(file_cuda_add, file_sage_add));
    }
}
__global__ void get_point_multiplied(Point *a, uint64_t m)
{
    *a = a->one() * m;
}
TEST_F(PointTests, check_wellformed)
{
    Point *a;
    bool *res;
    bool *result = new bool;
    size_t multi1;
    CUDA_CHECK(cudaMalloc(&res, sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&a, sizeof(Point)));
    for (size_t i = 0; i < NUM_TEST; i++)
    {
        multi1 = rand() + 1;
        get_point_multiplied<<<1, 1>>>(a, multi1);
        CUDA_CHECK(cudaDeviceSynchronize());
        check_wellformed_kernel<<<1, 1>>>(a, res);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(result, res, sizeof(bool), cudaMemcpyDeviceToHost));
        ASSERT_TRUE(*result);
    }
}
__global__ void make_point_to_affine(Point *a)
{
    a->to_affine();
}

TEST_F(PointTests, check_affine)
{
    Point *a;
    bool *d_res, *res;
    res = new bool;
    size_t multi1;
    CUDA_CHECK(cudaMalloc(&a, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(bool)));
    for (size_t i = 0; i < NUM_TEST; i++)
    {
        multi1 = rand() + 1;
        get_point_multiplied<<<1, 1>>>(a, multi1);
        CUDA_CHECK(cudaDeviceSynchronize());
        make_point_to_affine<<<1, 1>>>(a);
        CUDA_CHECK(cudaDeviceSynchronize());
        check_affine_kernel<<<1, 1>>>(a, d_res);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(res, d_res, sizeof(bool), cudaMemcpyDeviceToHost));
        ASSERT_TRUE(res);
    }
}
TEST_F(PointTests, check_equal)
{
    Point *a, *b;
    bool *res, *d_res;
    res = new bool;
    size_t multi;
    CUDA_CHECK(cudaMalloc(&a, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&b, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(bool)));
    for (size_t i = 0; i < NUM_TEST; i++)
    {
        multi = rand() + 1;
        get_point_multiplied<<<1, 1>>>(a, multi);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(b, a, sizeof(Point), cudaMemcpyDeviceToDevice));
        make_point_to_affine<<<1, 1>>>(b);
        CUDA_CHECK(cudaDeviceSynchronize());
        is_equal_kernel<<<1, 1>>>(a, b, d_res);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(res, d_res, sizeof(bool), cudaMemcpyDeviceToHost));
        ASSERT_TRUE(*res);
    }
}
TEST_F(PointTests, check_unequal)
{
    Point *a, *b;
    bool *res, *d_res;
    res = new bool;
    size_t multi, multi1;
    CUDA_CHECK(cudaMalloc(&a, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&b, sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_res, sizeof(bool)));
    for (size_t i = 0; i < NUM_TEST; i++)
    {
        multi = rand() + 1;
        multi1 = rand() + 1;
        while (multi == multi1)
        {
            multi += rand();
            multi1 += rand();
        }
        get_point_multiplied<<<1, 1>>>(a, multi);
        get_point_multiplied<<<1, 1>>>(b, multi1);
        CUDA_CHECK(cudaDeviceSynchronize());
        is_not_equal_kernel<<<1, 1>>>(a, b, d_res);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(res, d_res, sizeof(bool), cudaMemcpyDeviceToHost));
        ASSERT_TRUE(*res);
    }
}

#endif