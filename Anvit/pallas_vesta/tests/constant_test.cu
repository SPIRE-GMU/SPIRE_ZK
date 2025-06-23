#ifndef __CONSTANT_UNIT_TEST
#define __CONSTANT_UNIT_TEST

#include "test_header.cuh"

void write_project_output_to_file(uint64_t *data, int limit = 4) {

    std::ofstream out("cu_constant_output.txt");
    bool printStarted = false;
    for(int i= limit -1;i>=0; i--)
    {
        if(printStarted) 
        {
            out<< std::hex << std::uppercase << std::setfill('0') << std::setw(16) <<data[i];
        }
        else 
        {
            if(data[i]) 
            {
                out << std::hex << std::uppercase <<data[i];
                printStarted = true;
            }
        }
    }
        
}

bool compare_two_files_output ()
{
    std::ifstream in_cuda("cu_constant_output.txt");
    std::ifstream in_sage("sage_constant_output.txt");
    std::string cu_out;
    in_cuda>>cu_out;
    std::string sage_out;
    in_sage>>sage_out;
    
    if(cu_out.size() != sage_out.size())
        return false;
    for(int i=0; i< cu_out.size(); i++)
    {
        if(cu_out[i] != sage_out[i])
            return false;
    }
    in_cuda.close();
    in_sage.close();
    return true;
}


class CurveConstants: public ::testing::Test
{
public:
    CurveConstants()
    {

    }
    ~CurveConstants() override
    {

    }
    void SetUp() override
    {

    }
    void TearDown() override
    {

    }
};



TEST_F(CurveConstants, check_Modulus_Pallas) 
{
    int ret = std::system("sage -python constant_sage.py modulus pallas");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, pallas::MODULUS, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    write_project_output_to_file(data);
    ASSERT_TRUE(compare_two_files_output());
}


TEST_F(CurveConstants, check_R_Pallas) 
{
    int ret = std::system("sage -python constant_sage.py r pallas");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, pallas::R, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_project_output_to_file(data);
    ASSERT_TRUE(compare_two_files_output());
}


TEST_F(CurveConstants, check_R2_Pallas) 
{
    int ret = std::system("sage -python constant_sage.py r2 pallas");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, pallas::R2, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_project_output_to_file(data);
    ASSERT_TRUE(compare_two_files_output());
}


TEST_F(CurveConstants, check_R3_Pallas) 
{
    int ret = std::system("sage -python constant_sage.py r3 pallas");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, pallas::R3, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_project_output_to_file(data);
    ASSERT_TRUE(compare_two_files_output());
}


TEST_F(CurveConstants, check_Generator_Pallas) 
{
    int ret = std::system("sage -python constant_sage.py generator pallas");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, pallas::GENERATOR, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_project_output_to_file(data);
    ASSERT_TRUE(compare_two_files_output());
}


TEST_F(CurveConstants, check_Inv_Pallas) 
{
    int ret = std::system("sage -python constant_sage.py inv pallas");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, &pallas::INV, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    write_project_output_to_file(data, 1);
    ASSERT_TRUE(compare_two_files_output());
}


TEST_F(CurveConstants, check_Modulus_Vesta) 
{
    int ret = std::system("sage -python constant_sage.py modulus vesta");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, vesta::MODULUS, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_project_output_to_file(data);
    ASSERT_TRUE(compare_two_files_output());
}


TEST_F(CurveConstants, check_R_Vesta) 
{
    int ret = std::system("sage -python constant_sage.py r vesta");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, vesta::R, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_project_output_to_file(data);
    ASSERT_TRUE(compare_two_files_output());
}


TEST_F(CurveConstants, check_R2_Vesta) 
{
    int ret = std::system("sage -python constant_sage.py r2 vesta");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, vesta::R2, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_project_output_to_file(data);
    ASSERT_TRUE(compare_two_files_output());
}


TEST_F(CurveConstants, check_R3_Vesta) 
{
    int ret = std::system("sage -python constant_sage.py r3 vesta");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, vesta::R3, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_project_output_to_file(data);
    ASSERT_TRUE(compare_two_files_output());
}


TEST_F(CurveConstants, check_Generator_Vesta) 
{
    int ret = std::system("sage -python constant_sage.py generator vesta");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, vesta::GENERATOR, sizeof(uint64_t)*4, cudaMemcpyDeviceToHost);
    write_project_output_to_file(data);
    ASSERT_TRUE(compare_two_files_output());
}


TEST_F(CurveConstants, check_Inv_Vesta) 
{
    int ret = std::system("sage -python constant_sage.py inv vesta");
    if (ret != 0) 
    {
        ASSERT_TRUE(false);
    }
    uint64_t data[4];
    cudaMemcpy(data, &vesta::INV, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    write_project_output_to_file(data, 1);
    ASSERT_TRUE(compare_two_files_output());
}



#endif