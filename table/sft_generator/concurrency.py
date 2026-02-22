#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发处理模块
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Union
import queue
import time


class ConcurrentProcessor:
    """
    并发处理类，用于并行处理多个样本
    """

    def __init__(self, max_workers: int = 5):
        """
        初始化并发处理器

        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_concurrently(self, items: List[Dict[str, Any]], process_func,
                             progress_callback=None) -> List[Dict[str, Any]]:
        """
        并发处理项目

        Args:
            items: 待处理项目列表
            process_func: 处理单个项目的函数
            progress_callback: 进度回调函数

        Returns:
            处理结果列表
        """
        results = []
        futures = []

        for i, item in enumerate(items):
            future = self.executor.submit(process_func, item)
            futures.append((i, future))

        for i, future in futures:
            try:
                result = future.result()
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(items))

            except Exception as e:
                print(f"处理第 {i+1} 个项目时出错: {e}")
                results.append(None)

        return results

    def process_with_timeout(self, process_func, timeout: int = 60,
                             *args, **kwargs) -> Union[Any, None]:
        """
        带超时的处理方法

        Args:
            process_func: 处理函数
            timeout: 超时时间（秒）
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            处理结果或 None（超时）
        """
        try:
            return self.executor.submit(process_func, *args, **kwargs).result(timeout=timeout)
        except asyncio.TimeoutError:
            print(f"处理超时 ({timeout} 秒)")
            return None
        except Exception as e:
            print(f"处理出错: {e}")
            return None

    def shutdown(self, wait: bool = True):
        """
        关闭执行器

        Args:
            wait: 是否等待未完成的任务
        """
        self.executor.shutdown(wait=wait)


class RateLimiter:
    """
    速率限制器，用于控制 API 调用频率
    """

    def __init__(self, max_requests: int = 10, per_seconds: int = 60):
        """
        初始化速率限制器

        Args:
            max_requests: 最大请求数
            per_seconds: 时间窗口（秒）
        """
        self.max_requests = max_requests
        self.per_seconds = per_seconds
        self.request_times = queue.Queue()

    def acquire(self):
        """
        获取令牌，可能会阻塞直到有可用令牌
        """
        now = time.time()

        # 删除超时的请求时间
        while not self.request_times.empty() and (now - self.request_times.queue[0] > self.per_seconds):
            self.request_times.get()

        # 如果已达到限制，等待
        if self.request_times.qsize() >= self.max_requests:
            wait_time = self.per_seconds - (now - self.request_times.queue[0])
            if wait_time > 0:
                print(f"已达到速率限制，等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)

        # 添加新请求时间
        self.request_times.put(time.time())

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AsyncProcessor:
    """
    异步处理器，用于支持异步任务处理
    """

    @staticmethod
    async def process_async(items: List[Dict[str, Any]], process_func,
                            max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        异步处理项目

        Args:
            items: 待处理项目列表
            process_func: 处理单个项目的异步函数
            max_workers: 最大工作线程数

        Returns:
            处理结果列表
        """
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = []

            for item in items:
                task = loop.run_in_executor(executor, process_func, item)
                tasks.append(task)

            return await asyncio.gather(*tasks)

    @staticmethod
    async def process_with_timeout_async(process_func, timeout: int = 60,
                                        *args, **kwargs) -> Union[Any, None]:
        """
        异步带超时的处理方法

        Args:
            process_func: 异步处理函数
            timeout: 超时时间（秒）
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            处理结果或 None（超时）
        """
        try:
            return await asyncio.wait_for(process_func(*args, **kwargs), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"处理超时 ({timeout} 秒)")
            return None
        except Exception as e:
            print(f"处理出错: {e}")
            return None
