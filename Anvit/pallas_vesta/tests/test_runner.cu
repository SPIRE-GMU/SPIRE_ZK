#include "constant_test.cu"
#include "Point_test.cu"
#include "Field_test.cu"

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
