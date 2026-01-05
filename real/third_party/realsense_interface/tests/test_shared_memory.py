#!/usr/bin/env python3
"""
共享内存模块测试
"""

import unittest
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from realsense_interface.shared_memory import (
    SharedNDArray, 
    SharedMemoryQueue, 
    SharedMemoryRingBuffer,
    SharedAtomicCounter
)


class TestSharedMemory(unittest.TestCase):
    
    def setUp(self):
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
    
    def tearDown(self):
        self.shm_manager.shutdown()
    
    def test_shared_ndarray(self):
        """测试SharedNDArray"""
        # 创建测试数组
        test_array = np.random.rand(10, 20).astype(np.float32)
        
        # 创建共享数组
        shared_array = SharedNDArray.create_from_array(
            self.shm_manager, test_array)
        
        # 验证数据
        np.testing.assert_array_equal(shared_array.get(), test_array)
        self.assertEqual(shared_array.shape, test_array.shape)
        self.assertEqual(shared_array.dtype, test_array.dtype)
    
    def test_atomic_counter(self):
        """测试原子计数器"""
        counter = SharedAtomicCounter(self.shm_manager)
        
        # 测试基本操作
        self.assertEqual(counter.load(), 0)
        
        counter.store(42)
        self.assertEqual(counter.load(), 42)
        
        counter.add(8)
        self.assertEqual(counter.load(), 50)
    
    def test_shared_memory_queue(self):
        """测试共享内存队列"""
        examples = {
            'data': np.zeros((5, 5), dtype=np.float32),
            'timestamp': 0.0
        }
        
        queue = SharedMemoryQueue.create_from_examples(
            self.shm_manager, examples, buffer_size=10)
        
        # 测试空队列
        self.assertTrue(queue.empty())
        self.assertEqual(queue.qsize(), 0)
        
        # 添加数据
        test_data = {
            'data': np.random.rand(5, 5).astype(np.float32),
            'timestamp': 1.0
        }
        queue.put(test_data)
        
        self.assertFalse(queue.empty())
        self.assertEqual(queue.qsize(), 1)
        
        # 获取数据
        result = queue.get()
        np.testing.assert_array_equal(result['data'], test_data['data'])
        self.assertEqual(result['timestamp'], test_data['timestamp'])
        
        self.assertTrue(queue.empty())
    
    def test_shared_memory_ring_buffer(self):
        """测试共享内存环形缓冲区"""
        examples = {
            'image': np.zeros((100, 100, 3), dtype=np.uint8),
            'timestamp': 0.0
        }
        
        buffer = SharedMemoryRingBuffer.create_from_examples(
            self.shm_manager, examples, 
            get_max_k=5, get_time_budget=0.1, put_desired_frequency=30)
        
        # 添加数据
        for i in range(3):
            test_data = {
                'image': np.full((100, 100, 3), i, dtype=np.uint8),
                'timestamp': float(i)
            }
            buffer.put(test_data)
        
        # 获取最新数据
        result = buffer.get()
        np.testing.assert_array_equal(
            result['image'], np.full((100, 100, 3), 2, dtype=np.uint8))
        
        # 获取最近k个数据
        results = buffer.get_last_k(2)
        self.assertEqual(results['image'].shape[0], 2)


if __name__ == '__main__':
    unittest.main()