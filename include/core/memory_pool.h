#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <vector>
#include <cstddef>

/**
 * High-performance fixed-size block allocator.
 * Optimized for cache locality and reduced fragmentation.
 */
class MemoryPool {
private:
    struct Block {
        Block* next;
    };
    
    Block* freeHead;
    std::vector<void*> chunks;
    size_t blockSize;

public:
    MemoryPool(size_t size) : blockSize(size), freeHead(nullptr) {}

    void* allocate() {
        if (!freeHead) {
            expand();
        }
        Block* block = freeHead;
        freeHead = freeHead->next;
        return block;
    }

    void deallocate(void* ptr) {
        Block* block = static_cast<Block*>(ptr);
        block->next = freeHead;
        freeHead = block;
    }

private:
    void expand() {
        // Expansion logic omitted for brevity
    }
};

#endif // MEMORY_POOL_H
