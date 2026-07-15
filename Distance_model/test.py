import time
from functools import wraps

class FunctionTimeoutError(Exception):
    """自定义异常，当函数运行时间超过阈值时抛出"""
    pass

def measure_time(threshold):
    """装饰器，用于检测函数运行时间是否超过阈值
    
    Args:
        threshold (float): 时间阈值（秒）
    
    Returns:
        function: 装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)  # 执行原函数
            elapsed_time = time.time() - start_time
            
            if elapsed_time > threshold:
                raise FunctionTimeoutError(
                    f"Function '{func.__name__}' exceeded time threshold. "
                    f"Elapsed time: {elapsed_time:.2f}s, Threshold: {threshold}s"
                )
            return result
        return wrapper
    return decorator

# 使用示例
@measure_time(threshold=1.5)  # 设置阈值为1.5秒
def my_function():
    """模拟一个耗时函数"""
    time.sleep(2)  # 模拟耗时2秒的操作
    return "Done"

if __name__ == "__main__":
    try:
        result = my_function()
        #print("Function completed:", result)
    except FunctionTimeoutError as e:
        #print("Error:", e) 