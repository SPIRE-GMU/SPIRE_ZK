#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <iostream>
#include <cassert>

class ThrustSet {
private:
    thrust::device_vector<int> d_set;

public:
    void insert(const int value) {
        /*
         * Inserts a value into the set.
         * If the value already exists, it does nothing (no duplicates allowed).
         */
        if (!contains(value)) {
            d_set.push_back(value);
        }
    }

    void remove(const int value) {
        /*
         * Removes the specified value from the set.
         * If the value is not present, it does nothing.
         */
        auto end = thrust::remove(d_set.begin(), d_set.end(), value);
        d_set.erase(end, d_set.end());
    }

    bool contains(const int value) {
        /*
         * Checks if the set contains the given value.
         * Returns true if the value is found, false otherwise.
         */
        return thrust::find(d_set.begin(), d_set.end(), value) != d_set.end();
    }

    void print() {
        /*
         * Prints the contents of the set for debugging purposes.
         */
        thrust::host_vector<int> h_set = d_set;
        std::cout << "Set contents: ";
        for (int val : h_set) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
    int count() {
        /*
         * Returns the number of elements in the set.
         */
        return d_set.size();
    }
};

void testInsert() {
    ThrustSet mySet;
    mySet.insert(5);
    assert(mySet.count() == 1);
    mySet.insert(2);
    assert(mySet.count() == 2);
    mySet.insert(8);
    assert(mySet.count() == 3);
    mySet.insert(5);
    assert(mySet.count() == 3); // Inserting duplicate should not change the count
    assert(mySet.contains(5));
    assert(mySet.contains(2));
    assert(mySet.contains(8));
}

void testRemove() {
    ThrustSet mySet;
    mySet.insert(5);
    mySet.insert(2);
    mySet.insert(8);
    assert(mySet.count() == 3);
    mySet.remove(2);
    assert(mySet.count() == 2);
    assert(!mySet.contains(2));
    assert(mySet.contains(5));
    mySet.remove(5); // Remove another element
    assert(mySet.count() == 1); // Now only 8 should be left
    assert(!mySet.contains(5));
    assert(mySet.contains(8)); // Ensure 8 is still in the set
    mySet.remove(8); // Remove the last element 
    assert(mySet.count() == 0); // Now the set should be empty
    assert(!mySet.contains(8)); // Ensure 8 is no longer in the set
    mySet.remove(10); // Removing a non-existent element should not change the count
    assert(mySet.count() == 0); // Still should be 0
}

void testContains() {
    ThrustSet mySet;
    mySet.insert(3);
    mySet.insert(7);
    mySet.insert(9);
    assert(mySet.contains(3));
    assert(mySet.contains(7));
    assert(!mySet.contains(5));
}

int main() {
    testInsert();
    testRemove();
    testContains();
    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
